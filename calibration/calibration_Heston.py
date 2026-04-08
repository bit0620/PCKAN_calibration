# import sys
# import os
# # 获取当前文件所在目录（A目录）
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 获取项目根目录（project目录）
# project_root = os.path.dirname(current_dir)
# # 将项目根目录添加到模块搜索路径
# sys.path.insert(0, project_root)

# from statistics_model.Heston import Heston_Price_torch_c
# from scipy.optimize import differential_evolution
# from tqdm import trange

# import pandas as pd
# import numpy as np
# import torch


# device = "cuda" if torch.cuda.is_available() else "cpu"

# train_data = pd.read_csv('../data/test_data.csv')
# train_data = np.array(train_data)
# train_data_tensor = torch.tensor(train_data[:2000, :]).to(device)

# option_params = train_data_tensor[:, :-1]
# real_prices = train_data_tensor[:, -1]

# # Bounds for optimization  sigma v0 rho theta kappa
# Heston_bounds = ((0.01, 1), (0, 5), (-1, 0), (0, 3), (0.01, 15))

# n_len = len(train_data_tensor)

# # 每条数据找一组参数
# calibration_params = []
# for i in trange(n_len):
#     def object_fun(x):
#         x = torch.tensor(x, dtype=torch.float32, device=device)
#         price = Heston_Price_torch_c(option_params[i], x, device)
#         price = torch.nan_to_num(price, nan=100, posinf=100, neginf=100)
#         error = (price - real_prices[i]) ** 2
#         return error.item()

#     result = differential_evolution(object_fun, bounds=Heston_bounds, popsize=20, maxiter=30)
#     calibration_params.append(result.x)

# calibration_params_df = pd.DataFrame(calibration_params, columns=["sigma", "v0", "rho", "theta", "kappa"])
# calibration_params_df.to_csv("../data/calibration_params/Heston/Heston_params_b.csv", index=False)

import sys
import os
# 获取当前文件所在目录（A目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（project目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, project_root)

from statistics_model.Heston import Heston_Price_torch_c
from scipy.optimize import differential_evolution
from tqdm import trange

import pandas as pd
import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = pd.read_csv('../data/test_data.csv')
train_data = np.array(train_data)
train_data_tensor = torch.tensor(train_data[:2000, :], dtype=torch.float32).to(device)

option_params = train_data_tensor[:, :-1]
real_prices = train_data_tensor[:, -1]

# Bounds for optimization  sigma v0 rho theta kappa
Heston_bounds = ((0.01, 1), (0, 5), (-1, 0), (0, 3), (0.01, 15))

n_len = len(train_data_tensor)

# 定义目标函数（循环外）
def object_fun(x, option_param, real_price):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    price = Heston_Price_torch_c(option_param, x, device)
    price = torch.nan_to_num(price, nan=100, posinf=100, neginf=100)
    error = (price - real_price) ** 2
    return error.item()

def progress_callback(xk, convergence):
    print(f"当前解: {xk}, 收敛度: {convergence}")

# 每条数据找一组参数
calibration_params = []
for i in trange(n_len):
    option_param = option_params[i]
    real_price = real_prices[i]

    # 传给 differential_evolution 的函数只接受 x
    result = differential_evolution(
        lambda x: object_fun(x, option_param, real_price),
        bounds=Heston_bounds,
        popsize=20,
        maxiter=30,
        callback=progress_callback
    )
    calibration_params.append(result.x)

calibration_params_df = pd.DataFrame(
    calibration_params, 
    columns=["sigma", "v0", "rho", "theta", "kappa"]
)
calibration_params_df.to_csv("../data/calibration_params/Heston/Heston_params_b.csv", index=False)

