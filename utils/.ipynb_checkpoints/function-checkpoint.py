from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.autograd import grad
from torch import nn

import pandas as pd
import numpy as np
import torch

def load_data(s_model_name, batch_size=32, num_workers=4, device='cuda', random_state=42):
    # 加载数据
    option_data = pd.read_csv('../data/train_pinn_data.csv').to_numpy()
    if s_model_name == 'Heston':
        model_data = pd.read_csv(f'../data/calibration_params/{s_model_name}/Heston_params.csv').to_numpy()
    else:
        model_data = pd.read_csv(f'../data/calibration_params/{s_model_name}/FVSJ_params_all.csv').to_numpy()

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
    
    # 转换为PyTorch张量并转移到设备
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    pinn_data_t = torch.tensor(pinn_data, dtype=torch.float32).to(device)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    pinn_dataset = TensorDataset(pinn_data_t)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pinn_loader = DataLoader(pinn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
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
    loss_fn = nn.MSELoss()
    loss_pred = loss_fn(Y_hat, Y)

    price_pde = BS_PDE(params)
    price_pde = torch.where(torch.isnan(price_pde), torch.zeros_like(price_pde), price_pde)
    pi_loss_target = torch.zeros_like(price_pde)
    loss_pi = loss_fn(price_pde, pi_loss_target)

    loss_pred += loss_pi * lambda_weight

    return loss_pred

if __name__ == "__main__":
    train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name='Heston', batch_size=32, num_workers=0)
