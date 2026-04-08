import sys
import os

# 获取当前文件所在目录（A目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（project目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statistics_model.Heston import Heston_Price_torch_c
from statistics_model.FVSJ import FVSJ_fun
from tqdm import trange

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_n_model(s_model_name):
    if s_model_name == "Heston":
        return Heston_Price_torch_c
    else:
        return FVSJ_fun


def get_s_model_params(s_model_name, n_model_name):
   
    if s_model_name == 'Heston':
        model_params_train = pd.read_csv('../data/calibration_params/Heston/Heston_params_b.csv').to_numpy()
        scaler = StandardScaler()
        scaler.fit(model_params_train)
        if n_model_name == 'PC_KAN':
            S_model_params = pd.read_csv('../data/calibration_params/Heston/PCKAN.csv').to_numpy()
            # S_model_params = scaler.inverse_transform(S_model_params)
        else:
            S_model_params = pd.read_csv('../data/calibration_params/Heston/NN_params.csv').to_numpy()
            S_model_params = scaler.inverse_transform(S_model_params)
    else:
        model_params_train = pd.read_csv('../data/calibration_params/FVSJ/FVSJ_params.csv').to_numpy()
        scaler = StandardScaler()
        scaler.fit(model_params_train)
        if n_model_name == 'PC_KAN':
            S_model_params = pd.read_csv('../data/calibration_params/FVSJ/PCKAN_params.csv').to_numpy()
            S_model_params = scaler.inverse_transform(S_model_params)
        else:
            S_model_params = pd.read_csv('../data/calibration_params/FVSJ/NN_params.csv').to_numpy()
            S_model_params = scaler.inverse_transform(S_model_params)

    return S_model_params

# def error(real_prices, model_prices, plot_len):
#     error_all = (real_prices - model_prices) ** 2
#     mse = np.mean(error_all)

#     min_error = float('inf')
#     min_error_index = 0
#     for i in trange(0, len(real_prices) - plot_len + 1):
#         tmp = np.mean(error_all[i:i+plot_len])
#         if min_error > tmp:
#             min_error = tmp
#             min_error_index = i

#     return min_error, min_error_index, mse

def log_rmse(y, yhat, eps=1e-8):
    return np.sqrt(np.mean((np.log(y + eps) - np.log(yhat + eps))**2))

def smape(y, yhat, eps=1e-8):
    return np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps))

def mse(y, yhat):
    return np.mean((y - yhat) ** 2)

def error(real_prices, model_prices, plot_len, metric="logrmse"):
    """
    real_prices, model_prices: (N,1) or (N,) 数组
    plot_len: 窗口长度
    metric: 'mse' | 'logrmse' | 'smape'
    """
    real_prices = real_prices.flatten()
    model_prices = model_prices.flatten()
    
    # 选择度量函数
    if metric == "mse":
        scorer = mse
    elif metric == "smape":
        scorer = smape
    else:
        scorer = log_rmse  # 默认 logrmse

    # 全局误差
    global_err = scorer(real_prices, model_prices)

    # 滑窗搜索最优片段
    min_err, min_idx = float("inf"), 0
    for i in trange(0, len(real_prices) - plot_len + 1):
        e = scorer(real_prices[i:i+plot_len], model_prices[i:i+plot_len])
        if e < min_err:
            min_err, min_idx = e, i

    return min_err, min_idx, global_err


def draw_real_model_prices(real_prices, model_prices, min_error_index, s_model_name, n_model_name, plot_len):
    real_prices = real_prices[min_error_index:min_error_index+plot_len]
    model_prices = model_prices[min_error_index:min_error_index+plot_len]
    print(real_prices)
    print(model_prices)

    real_prices = real_prices.flatten()
    model_prices = model_prices.flatten()

    x = np.arange(plot_len)

    plt.figure(figsize=(10, 5))
    plt.plot(x, real_prices, color='black', label='真实价格')
    plt.plot(x, model_prices, color='red', label='校准价格')
    plt.xlabel('时间')
    plt.ylabel('期权价格')
    plt.title('Real vs Model Option Prices')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig(f'{s_model_name}_{n_model_name}price.png', dpi=300)

def caculator_prices(option_params, real_prices_test, model_params, s_model_name, n_model_name):
    option_params = torch.tensor(option_params, dtype=torch.float32).to(device)
    model_params = torch.tensor(model_params, dtype=torch.float32).to(device)
    model_prices = torch.zeros((len(model_params), 1))
    
    # 加载模型
    # model = torch.load('../neural_network/train_res/Heston/PCKAN0.00017028213329533726.pt')
    # model = model.to(torch.float32).to(device)
    
    # # 标准化
    # scaler = StandardScaler()
    # option_params = scaler.fit_transform(option_params.cpu().numpy())  # 先转numpy再标准化
    # option_params = torch.tensor(option_params, dtype=torch.float32).to(device)

    # # 拼接输入
    # input_x = torch.cat([option_params, model_params], dim=1)  # 确认 shape 是否对齐

    # # 预测
    # with torch.no_grad():
    #     model_prices = model(input_x).cpu().numpy()

    model = get_n_model(s_model_name)
    for i in trange(len(model_params)):
        model_prices[i] = model(option_params[i], model_params[i], device)

    # 保存
    save_path = f'../data/calibration_prices/{s_model_name}/{n_model_name}.csv'
    df = pd.DataFrame({
        'model_price': model_prices.flatten(),
        'real_prices': real_prices_test.flatten()
    })
    df.to_csv(save_path, index=False)
    return model_prices.numpy()


if __name__ == '__main__':

    plot_len = 120
    option_params_test = pd.read_csv('../data/test_data_p.csv').to_numpy()
    option_params_test = option_params_test[:plot_len, :]
    
    real_prices_test = option_params_test[:, -1].reshape(-1, 1)
    option_params_test = option_params_test[:, :-1]

    s_model_name = 'Heston'
    n_model_name = 'PC_KAN'
    

    S_model_params_test = get_s_model_params(s_model_name, n_model_name)
    model_prices_inv = caculator_prices(option_params_test, real_prices_test, S_model_params_test, s_model_name, n_model_name)

    # model_prices_inv = pd.read_csv(f'../data/calibration_prices/{s_model_name}/{n_model_name}.csv').to_numpy()
    min_error, min_error_index, mse = error(real_prices_test, model_prices_inv, plot_len, metric="logrmse")
    print(f'最小误差为{min_error}\n')
    print(f'最小误差在{min_error_index}')
    print(f'全局误差为{mse}\n')
    draw_real_model_prices(real_prices_test, model_prices_inv, min_error_index, s_model_name, n_model_name, plot_len)
    

