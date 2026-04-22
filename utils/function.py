from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from torch import nn

import os
import pandas as pd
import numpy as np
import torch

def load_data(s_model_name, batch_size=32, num_workers=4, device='cuda', random_state=42):
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载数据
    option_data = pd.read_csv(os.path.join(project_root, 'data/train_pinn_data.csv')).to_numpy()
    if s_model_name == 'Heston':
        model_data = pd.read_csv(os.path.join(project_root, f'data/calibration_params/{s_model_name}/Heston_params.csv')).to_numpy()
    else:
        model_data = pd.read_csv(os.path.join(project_root, f'data/calibration_params/{s_model_name}/FVSJ_params.csv')).to_numpy()

    y = option_data[:, -1].reshape(-1, 1)
    option_data = option_data[:, :-1]
    
    # 检查数据一致性
    assert len(option_data) == len(model_data), "数据行数不匹配"
    
    # 合并数据
    X = np.concatenate([option_data, model_data], axis=1)

    # 划分训练测试集 (添加随机打乱)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pinn_data = X_train[:, :len(option_data[0])]
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    # 分别标准化特征
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)  # 使用训练集的参数
    
    # 标准化目标值
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    
    # 修改这部分：不要将数据直接移动到CUDA
    X_train_t = torch.tensor(X_train, dtype=torch.float32)  # 移除 .to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    pinn_data_t = torch.tensor(pinn_data, dtype=torch.float32)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    pinn_dataset = TensorDataset(pinn_data_t)
    
    # 修改DataLoader参数
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # 保持pin_memory以加速CPU到GPU的传输
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    pinn_loader = DataLoader(
        pinn_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    input_size = X_train.shape[1]
    return train_loader, test_loader, pinn_loader, input_size


def black_scholes_price(cp, spot, strike, maturity, vol, r):
    # 计算d1和d2（向量化操作）
    d1 = (torch.log(spot / strike) + (r + 0.5 * vol**2) * maturity) / (vol * torch.sqrt(maturity))
    d2 = d1 - vol * torch.sqrt(maturity)
    
    # 创建标准正态分布（自动匹配输入设备）
    dist = torch.distributions.normal.Normal(0, 1)
    
    # 计算累积分布函数
    cdf_d1 = dist.cdf(d1)
    cdf_d2 = dist.cdf(d2)
    cdf_neg_d1 = dist.cdf(-d1)
    cdf_neg_d2 = dist.cdf(-d2)
    
    # 计算看涨和看跌期权价格
    call_price = spot * cdf_d1 - strike * torch.exp(-r * maturity) * cdf_d2
    put_price = strike * torch.exp(-r * maturity) * cdf_neg_d2 - spot * cdf_neg_d1
    
    # 根据cp标志选择看涨或看跌价格
    option_price = torch.where(cp == 1, call_price, put_price)
    option_price = torch.where(option_price == 0, torch.tensor(0.02), option_price)
    
    return option_price

def nth_derivative(f, wrt, n):
    if n == 0:
        return f
    
    df_dx = grad(f, wrt, 
                 grad_outputs=torch.ones_like(f), 
                 create_graph=True, 
                 retain_graph=True,
                 allow_unused=True)[0]
    
    if df_dx is None:
        # 返回与输入相同形状的零张量
        return torch.zeros_like(wrt)
    
    if n == 1:
        return df_dx
    
    # 递归计算高阶导数
    return nth_derivative(df_dx, wrt, n-1)

# Helper function for PI-ConvTF Pinn Loss
def BS_PDE(params):
    # 顺序 vol cp T spot  strike， r 

    vol = params[:, 0].view(-1, 1).requires_grad_(True)
    cp = params[:, 1].view(-1, 1).requires_grad_(True)
    maturity = params[:, 2].view(-1, 1).requires_grad_(True)
    strike = params[:, 3].view(-1, 1).requires_grad_(True)
    spot = params[:, 4].view(-1, 1).requires_grad_(True)
    r = params[:, 5].view(-1, 1).requires_grad_(True)
    
    c = black_scholes_price(cp, spot, strike, maturity, vol, r)

    c_t = nth_derivative(c, maturity, 1)
    c_s = nth_derivative(c, spot, 1)
    c_ss = nth_derivative(c, spot, 2)

    f = c_t + r*c - r*spot*c_s - torch.square(vol*spot)*c_ss*0.5
    # f = c_t + r * Y_hat - r * spot * c_s - torch.square(vol * spot) * c_ss * 0.5
    return f


def loss_function(Y_hat, Y, params, lambda_weight):
    """
    计算数据损失和物理损失。

    参数:
        Y_hat: 预测的隐含波动率（物理空间）
        Y: 真实的隐含波动率（物理空间）
        params: 物理参数 [iv, cp, maturity, strike, spot, r]
        lambda_weight: 权重参数

    返回:
        loss_pred: 数据损失
        loss_pi: 物理损失（PDE Loss）
    """
    loss_fn = nn.MSELoss()

    # 1. 数据损失 - 预测波动率与真实波动率的差异
    loss_pred = loss_fn(Y_hat, Y)

    # 2. 物理损失 - 使用PDE约束
    # 从params中提取参数
    iv = params[:, 0].view(-1, 1)  # 隐含波动率
    cp = params[:, 1].view(-1, 1)  # 期权类型
    maturity = params[:, 2].view(-1, 1)  # 到期时间
    strike = params[:, 3].view(-1, 1)  # 行权价
    spot = params[:, 4].view(-1, 1)  # 标的资产价格
    r = params[:, 5].view(-1, 1)  # 无风险利率

    # 使用预测的隐含波动率计算期权价格
    option_price_pred = black_scholes_price(cp, spot, strike, maturity, Y_hat, r)

    # 使用真实的隐含波动率计算期权价格
    option_price_true = black_scholes_price(cp, spot, strike, maturity, Y, r)

    # PDE Loss：预测期权价格与真实期权价格的差异
    # 这实际上是在检查预测的波动率是否正确
    loss_pi = loss_fn(option_price_pred, option_price_true)

    # 返回数据损失和PDE损失
    return loss_pred, loss_pi
    # scale = torch.sqrt(torch.square(term1) + torch.square(term2) + 
    #                   torch.square(term3) + torch.square(term4) + 1e-8)
    # # 显式归一化各PDE项
    # # 计算各特征的最大值用于归一化
    # C_MAX = torch.max(torch.abs(option_price)) + 1e-8
    # T_MAX = torch.max(maturity) + 1e-8
    # Y_MAX = torch.max(Y_hat) + 1e-8
    # S_MAX = torch.max(spot) + 1e-8

    # # 归一化各项，确保各项量级一致
    # c_t_norm = c_t / (C_MAX / T_MAX)
    # term2_norm = term2 / C_MAX
    # term3_norm = term3 / C_MAX
    # # 对扩散项进行更保守的归一化，避免其主导整个残差
    # diffusion_norm = term4 / (C_MAX * torch.clamp(Y_MAX**2, min=0.1, max=10.0))

    # # 计算归一化的PDE残差
    # f_normalized = c_t_norm + term2_norm - term3_norm - diffusion_norm

    # # 打印调试信息（只在第一个batch打印）
    # if torch.rand(1).item() < 0.01:  # 约1%的概率打印
    #     print(f"PDE残差各部分量级:")
    #     print(f"  c_t: {torch.mean(torch.abs(term1)).item():.6e} (归一化: {torch.mean(torch.abs(c_t_norm)).item():.6e})")
    #     print(f"  r*option_price: {torch.mean(torch.abs(term2)).item():.6e} (归一化: {torch.mean(torch.abs(term2_norm)).item():.6e})")
    #     print(f"  r*spot*c_s: {torch.mean(torch.abs(term3)).item():.6e} (归一化: {torch.mean(torch.abs(term3_norm)).item():.6e})")
    #     print(f"  0.5*(Y_hat*spot)^2*c_ss: {torch.mean(torch.abs(term4)).item():.6e} (归一化: {torch.mean(torch.abs(diffusion_norm)).item():.6e})")
    #     print(f"  PDE残差f: {torch.mean(torch.abs(f)).item():.6e}")
    #     print(f"  归一化PDE残差: {torch.mean(torch.abs(f_normalized)).item():.6e}")
    #     print(f"  Y_hat范围: [{torch.min(Y_hat).item():.6e}, {torch.max(Y_hat).item():.6e}]")
    #     print(f"  option_price范围: [{torch.min(option_price_center).item():.6e}, {torch.max(option_price_center).item():.6e}]")

    # # 处理可能的 NaN 值
    # f_normalized = torch.where(torch.isnan(f_normalized), torch.zeros_like(f_normalized), f_normalized)

    # # PDE Loss是残差与0的MSE
    # pi_loss_target = torch.zeros_like(f_normalized)
    # loss_pi = loss_fn(f_normalized, pi_loss_target)
    
    # return loss_pred, loss_pi

if __name__ == "__main__":
    train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name='Heston', batch_size=32, num_workers=0)
