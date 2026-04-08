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


def calibration_nn(option_params, real_prices, bounds, n_len, save_dir, net_name):
    model = torch.load(f'../neural_network/train_res/{save_dir}/PCKAN0.00017504163937849033.pt')

    calibration_params = []

    # 定义目标函数（循环外）
    def object_fun(x, option_params_iter, real_price):
        x = x.reshape(1, -1)
        input_x = np.concatenate([option_params_iter, x], axis=1)
        input_x = torch.tensor(input_x, dtype=torch.float32)
        with torch.no_grad():
            price = model(input_x)
        price = torch.nan_to_num(price, nan=100, posinf=100, neginf=100)
        error = (price - real_price) ** 2
        return error.item()

    for i in trange(n_len):
        option_params_iter = option_params[i].reshape(1, -1)
        real_price = real_prices[i]

        result = differential_evolution(
            lambda x: object_fun(x, option_params_iter, real_price),
            bounds=bounds,
            popsize=50,
            maxiter=150,
            workers=1
        )
        calibration_params.append(result.x)

    if save_dir == 'Heston':
        calibration_params_df = pd.DataFrame(calibration_params, columns=["sigma", "v0", "rho", "theta", "kappa"])
        calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}.csv", index=False)
    else:
        calibration_params_df = pd.DataFrame(calibration_params)
        calibration_params_df.to_csv(f"../data/calibration_params/{save_dir}/{net_name}.csv", index=False)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = pd.read_csv('../data/train_data.csv').to_numpy()
    option_params_train = train_data[:, :-1]
    real_prices_train = train_data[:, -1].reshape(-1, 1)

    test_data = pd.read_csv('../data/test_data.csv').to_numpy()
    option_params_test = test_data[:, :-1]
    real_prices_test = test_data[:, -1].reshape(-1, 1)

    option_params_scaler = StandardScaler()
    option_params_train_s = option_params_scaler.fit_transform(option_params_train)
    option_params_test_s = option_params_scaler.transform(option_params_test)

    real_prices_scaler = MinMaxScaler()
    real_prices_train_s = real_prices_scaler.fit_transform(real_prices_train)
    real_prices_test_s = real_prices_scaler.transform(real_prices_test)

    n_len = len(test_data)

    # Bounds for optimization  sigma v0 rho theta kappa
    save_dir = 'Heston'
    net_name = 'PCKAN'
    Heston_bounds = ((0.01, 1), (0, 5), (-1, 0), (0, 3), (0.01, 15))
    calibration_nn(option_params_test_s, real_prices_test_s, Heston_bounds, n_len, save_dir, net_name)

