# import sys
# import os
#
# # 获取当前文件所在目录（A目录）
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录（project目录）
# project_root = os.path.dirname(current_dir)
# # 将项目根目录添加到模块搜索路径
# sys.path.insert(0, project_root)
#
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from scipy.optimize import differential_evolution
# from tqdm import trange
#
# import pandas as pd
# import numpy as np
# import torch
#
#
# def calibration_nn(option_params, real_prices, bounds, n_len, save_dir, net_name):
#     model = torch.load(f'../neural_network/train_res/{save_dir}/PCKAN0.00017504163937849033.pt')
#     # 每条数据找一组参数
#     calibration_params = []
#     for i in trange(n_len):
#         def object_fun(x):
#             x = x.reshape(1, -1)
#             option_params_iter = option_params[i].reshape(1, -1)
#             input_x = np.concatenate([option_params_iter, x], axis=1)
#             input_x = torch.tensor(input_x)
#             with torch.no_grad():
#                 price = model(input_x)
#             price = torch.nan_to_num(price, nan=100, posinf=100, neginf=100)
#             error = (price - real_prices[i]) ** 2
#             return error.item()
#
#         result = differential_evolution(object_fun, bounds=bounds, popsize=50, maxiter=150, workers=1)
#         calibration_params.append(result.x)
#
#     if save_dir == 'Heston':
#         calibration_params_df = pd.DataFrame(calibration_params, columns=["sigma", "v0", "rho", "theta", "kappa"])
#         calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}.csv", index=False)
#     else:
#         calibration_params_df = pd.DataFrame(calibration_params)
#         calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}.csv", index=False)
#
#
# if __name__ == '__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     train_data = pd.read_csv('../data/train_data.csv').to_numpy()
#     option_params_train = train_data[:, :-1]
#     real_prices_train = train_data[:, -1].reshape(-1, 1)
#
#     test_data = pd.read_csv('../data/test_data.csv').to_numpy()
#     option_params_test = test_data[:, :-1]
#     real_prices_test = test_data[:, -1].reshape(-1, 1)
#
#     option_params_scaler = StandardScaler()
#     option_params_train_s = option_params_scaler.fit_transform(option_params_train)
#     option_params_test_s = option_params_scaler.transform(option_params_test)
#
#     real_prices_scaler = MinMaxScaler()
#     real_prices_train_s = real_prices_scaler.fit_transform(real_prices_train)
#     real_prices_test_s = real_prices_scaler.transform(real_prices_test)
#
#     # n_len = len(test_data)
#     n_len= 120
#
#     # Bounds for optimization  sigma v0 rho theta kappa
#     save_dir = 'Heston'
#     net_name = 'PCKAN'
#     Heston_bounds = ((0.01, 1), (0, 5), (-1, 0), (0, 3), (0.01, 15))
#     calibration_nn(option_params_test_s, real_prices_test_s, Heston_bounds, n_len, save_dir, net_name)



import sys
import os

# 获取当前文件所在目录（A目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（project目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, project_root)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import differential_evolution
from tqdm import trange

import pandas as pd
import numpy as np
import torch

torch.set_default_dtype(torch.float32)

def progress_callback(xk, convergence):
    print(f"当前解: {xk}, 收敛度: {convergence}")

def calibration_nn(option_params, real_prices, bounds, n_len, save_dir, net_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 把模型加载到同一设备，并设为eval
    model = torch.load(
        f'../neural_network/train_res/{save_dir}/PCKAN0.00032055925112217665.pt',
        map_location=device
    )
    model.eval()

    calibration_params = []

    # 2) 目标函数：把输入和real_price一起搬到device；确保返回float标量
    @torch.inference_mode()
    def object_fun(x, option_params_iter, real_price):
        # x: (D,) -> (1,D)
        x = x.reshape(1, -1).astype(np.float32)
        input_x = np.concatenate([option_params_iter, x], axis=1).astype(np.float32)

        input_x = torch.from_numpy(input_x).to(device)  # 保证和模型同设备
        price = model(input_x)
        price = torch.nan_to_num(price, nan=100.0, posinf=100.0, neginf=100.0).squeeze()  # 变成标量/1D

        # real_price 也在同设备、同dtype，并squeeze到1D
        rp = real_price.to(device).to(price.dtype).squeeze()

        error = (price - rp) ** 2
        # 返回python float
        return float(error.detach().cpu().item())

    for i in trange(n_len):
        option_params_iter = option_params[i].reshape(1, -1).astype(np.float32)
        real_price = torch.tensor(real_prices[i], dtype=torch.float32)  # 先建tensor，下面在目标函数里搬设备

        result = differential_evolution(
            lambda x: object_fun(x, option_params_iter, real_price),
            bounds=bounds,
            strategy="best1bin",  # 常用策略
            popsize=15,  # 每个维度种群数，越大越稳定，但计算量大
            mutation=(0.5, 1),  # 变异因子范围，(0.5,1)比固定0.8更灵活
            recombination=0.7,  # 重组率
            tol=1e-6,  # 收敛容忍度，调小能让结果更精确
            maxiter=100,  # 迭代次数，建议 >100
            polish=True,
            workers=1,     # 用CUDA时，建议保持1，避免多进程CUDA初始化问题
            callback=progress_callback
        )
        calibration_params.append(result.x)

    if save_dir == 'Heston':
        cols = ["sigma", "v0", "rho", "theta", "kappa"]
        calibration_params_df = pd.DataFrame(calibration_params, columns=cols)
        calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}_120_tr.csv", index=False)
    else:
        calibration_params_df = pd.DataFrame(calibration_params)
        calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}.csv", index=False)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = pd.read_csv('../data/test_data.csv').to_numpy()
    option_params_test = test_data[:, :-1]
    real_prices_test = test_data[:, -1].reshape(-1, 1)

    option_params_scaler = StandardScaler()
    option_params_test_s = option_params_scaler.fit_transform(option_params_test)

    real_prices_scaler = MinMaxScaler()
    real_prices_test_s = real_prices_scaler.fit_transform(real_prices_test)

    option_params_test_s = option_params_test_s[12704:12704 + 120, :]
    real_prices_test_s = real_prices_test_s[12704:12704 + 120, :]

    # n_len = len(test_data)
    n_len = 120

    # Bounds for optimization  sigma v0 rho theta kappa
    save_dir = 'FVSJ'
    # save_dir = "Heston"

    net_name = 'PCKAN'
    # net_name = 'NN'

    # bounds = ((-2, 2), (-2, 5), (-2, 2), (-1, 5), (-2, 2))
    bounds = ((-1.8, 1.9), (-1.8, 1.9), (-2, 1.8), (-2, 1.8), (-1.4, 2.3), (-1.4, 2.3),
              (-1.9, 1.9), (-2, 1.9), (-1.4, 2.1), (-1.4, 2), (-2, 1.3), (-2, 1.3),
              (-1.8, 1.8), (-1.9, 1.8), (-1.9, 1.9), (-1.9, 1.9), (-1.7, 1.9), (-1.7, 1.9),
              (-2.1, 1.8), (-2.2, 1.8), (-1.9, 1.9), (-1.9, 1.9), (-2.2, 1.7), (-2, 1.8), (-2.8, 1.1)
            )
    calibration_nn(option_params_test_s, real_prices_test_s, bounds, n_len, save_dir, net_name)
