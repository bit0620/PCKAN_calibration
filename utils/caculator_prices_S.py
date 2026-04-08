import sys
import os
# 获取当前文件所在目录（A目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（project目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
sys.path.insert(0, project_root)

from statistics_model.Heston import Heston_Price_torch_c
from statistics_model.FVSJ import FVSJ_fun
from tqdm import trange

import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name):
    if model_name == "Heston":
        return Heston_Price_torch_c
    else:
        return FVSJ_fun

def prices_error(s_model_params, model_name):
    model = get_model(model_name)
    train_data = pd.read_csv('../data/train_data.csv')
    train_data = np.array(train_data)
    train_data_tensor = torch.tensor(train_data).to(device)

    option_params = train_data_tensor[:1000, :-1]
    real_prices = train_data_tensor[:1000, -1]

    s_model_params = np.array(s_model_params)

    prices = torch.zeros((len(option_params))).to(device)

    for i in trange(len(option_params)):
        prices[i] = model(option_params[i], s_model_params[i], device)

    error_mape = ((real_prices - prices) / prices) ** 2
    print(f'最大MAPE误差为{torch.max(error_mape)}')
    error_mean_mape = torch.sum(error_mape) / len(error_mape)
    print(f'平均MAPE误差为{error_mean_mape}')

    error_mse = ((real_prices - prices)) ** 2
    print(f'最大MSE误差为{torch.max(error_mse)}')
    error_mean_mse = torch.sum(error_mse) / len(error_mse)
    print(f'平均MSE误差为{error_mean_mse}')

    # 保存预测结果到本地
    results_df = pd.DataFrame({
        "real_price": real_prices.cpu().numpy(),
        "predicted_price": prices.cpu().numpy(),
        "mape_error": error_mape.cpu().numpy(),
        "mse_error": error_mse.cpu().numpy()
    })
    results_df.to_csv(f"../data/calibration_prices/{model_name}/{model_name}_prices.csv", index=False, encoding="utf-8-sig")
    print("预测价格已保存")

    
if __name__ == '__main__':
    # s_model_params = pd.read_csv('../data/calibration_params/Heston/Heston_params.csv')
    # model_name = 'Heston'

    s_model_params = pd.read_csv('../data/calibration_params/FVSJ/FVSJ_params.csv')
    model_name = 'FVSJ'
    
    prices_error(s_model_params, model_name)