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
        model_data = pd.read_csv(f'../data/calibration_params/{s_model_name}/FVSJ_params.csv').to_numpy()
    y = option_data[:, -1].reshape(-1, 1)
    option_data = option_data[:, :-1]
    
    # 检查数据一致性
    assert len(option_data) == len(model_data), "数据行数不匹配"
    
    # 合并数据
    X = np.concatenate([option_data, model_data], axis=1)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pinn_data = X_train[:, :len(option_data[0])]
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]
    
    # 分别标准化特征
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    
    # 标准化目标值
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)
    
    # 获取标准化参数
    maturity_mean = torch.tensor(x_scaler.mean_[0], dtype=torch.float32)
    maturity_std = torch.tensor(x_scaler.scale_[0], dtype=torch.float32)
    spot_mean = torch.tensor(x_scaler.mean_[1], dtype=torch.float32)
    spot_std = torch.tensor(x_scaler.scale_[1], dtype=torch.float32)
    target_mean = torch.tensor(y_scaler.data_min_[0], dtype=torch.float32)
    target_std = torch.tensor(y_scaler.data_range_[0], dtype=torch.float32)
    
    # 转换为张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    pinn_data_t = torch.tensor(pinn_data, dtype=torch.float32)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    pinn_dataset = TensorDataset(pinn_data_t)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
    
    # 返回标准化参数
    scaler_params = {
        'maturity_mean': maturity_mean,
        'maturity_std': maturity_std,
        'spot_mean': spot_mean,
        'spot_std': spot_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'y_scaler': y_scaler  # 添加y_scaler用于反归一化
    }
    
    return train_loader, test_loader, pinn_loader, input_size, scaler_params



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


def _normalize_target(target, target_mean, target_std):
    """归一化目标值"""
    return (target - target_mean) / target_std

def _indexed_mse(Y_hat, Y_target, indices):
    """计算指定索引的MSE"""
    if len(indices) == 0:
        return torch.tensor(0.0, device=Y_hat.device)
    return torch.mean(torch.square(Y_hat[indices] - Y_target[indices]))

def bs_pde_residual(Y_hat, X, params):
    """计算Black-Scholes PDE残差"""
    # params columns: [contractType, dayToExpire, iv, strikePrice, spotPrice, RFR]
    maturity = params[:, 1].reshape(-1, 1)
    vol = params[:, 2].reshape(-1, 1)
    strike = params[:, 3].reshape(-1, 1)
    spot = params[:, 4].reshape(-1, 1)
    r = params[:, 5].reshape(-1, 1)

    # 计算Y_hat对maturity和spot的导数
    Y_hat_grad = torch.autograd.grad(
        Y_hat, X, 
        grad_outputs=torch.ones_like(Y_hat),
        create_graph=True, retain_graph=True
    )[0]

    # 假设X的第0列是maturity，第1列是spot
    dVdT = Y_hat_grad[:, 0].reshape(-1, 1)
    dVdS = Y_hat_grad[:, 1].reshape(-1, 1)

    # 计算二阶导数
    d2VdS2 = torch.autograd.grad(
        dVdS, X,
        grad_outputs=torch.ones_like(dVdS),
        create_graph=True, retain_graph=True
    )[0][:, 1].reshape(-1, 1)

    # Black-Scholes PDE残差
    pde_res = dVdT + r * Y_hat - r * spot * dVdS - 0.5 * torch.square(vol * spot) * d2VdS2
    return pde_res

def loss_function(Y_hat, Y, X, params, beta, target_mean, target_std):
    # params columns: [contractType, dayToExpire, iv, strikePrice, spotPrice, RFR]
    maturity = params[:, 1]
    strike = params[:, 3]
    spot = params[:, 4]
    r = params[:, 5]

    # IVP target: V(t=T, S) = max(S-K, 0)
    ivp_target_raw = torch.clamp(spot - strike, min=0.0)
    ivp_target = _normalize_target(ivp_target_raw, target_mean, target_std).reshape_as(Y_hat)

    # BVP targets for call option
    bvp1_target_raw = torch.zeros_like(spot)
    bvp2_target_raw = spot - strike * torch.exp(-r * maturity)
    bvp1_target = _normalize_target(bvp1_target_raw, target_mean, target_std).reshape_as(Y_hat)
    bvp2_target = _normalize_target(bvp2_target_raw, target_mean, target_std).reshape_as(Y_hat)

    # Use smallest/ largest samples in batch as boundary subsets to form Phi/Gamma/Omega'
    batch_size = Y_hat.shape[0]
    k = max(1, batch_size // 8)
    ivp_idx = torch.argsort(maturity)[:k]
    bvp1_idx = torch.argsort(spot)[:k]
    bvp2_idx = torch.argsort(spot, descending=True)[:k]

    mse_ivp = _indexed_mse(Y_hat, ivp_target, ivp_idx)
    mse_bvp1 = _indexed_mse(Y_hat, bvp1_target, bvp1_idx)
    mse_bvp2 = _indexed_mse(Y_hat, bvp2_target, bvp2_idx)
    mse_bvp = mse_bvp1 + mse_bvp2

    pde_res = bs_pde_residual(Y_hat, X, params)
    mse_pde = torch.mean(torch.square(pde_res))

    return mse_ivp + mse_bvp + beta * mse_pde

if __name__ == "__main__":
    train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name='Heston', batch_size=32, num_workers=0)
