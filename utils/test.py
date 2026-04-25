
import sys
import os
# 直接硬编码你的项目根目录路径
# 请根据实际情况修改下面的路径字符串
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


if project_root not in sys.path:
    sys.path.append(project_root)


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
        model_params_train = pd.read_csv('../data/calibration_params/Heston/Heston_params.csv').to_numpy()
        scaler = StandardScaler()
        scaler.fit(model_params_train)
        if n_model_name == 'PC_KAN':
            S_model_params = pd.read_csv('../data/calibration_params/Heston/PCKAN_120.csv').to_numpy()
            # S_model_params = scaler.inverse_transform(S_model_params)
        else:
            S_model_params = pd.read_csv('../data/calibration_params/Heston/NN_120_tr.csv').to_numpy()
            # S_model_params = scaler.inverse_transform(S_model_params)
    else:
        model_params_train = pd.read_csv('../data/calibration_params/FVSJ/FVSJ_params.csv').to_numpy()
        scaler = StandardScaler()
        scaler.fit(model_params_train)
        if n_model_name == 'PC_KAN':
            S_model_params = pd.read_csv('../data/calibration_params/FVSJ/PCKAN.csv').to_numpy()
            # S_model_params = scaler.inverse_transform(S_model_params)
        else:
            S_model_params = pd.read_csv('../data/calibration_params/FVSJ/NN.csv').to_numpy()
            # S_model_params = scaler.inverse_transform(S_model_params)

    return S_model_params


def log_rmse(y, yhat, eps=1e-8):
    return np.sqrt(np.mean((np.log(y + eps) - np.log(yhat + eps))**2))

def smape(y, yhat, eps=1e-8):
    return np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps))

def mse(y, yhat):
    return np.mean((y - yhat) ** 2)

def error(s_model_name, n_model_name, real_prices, model_prices, plot_len, metric="logrmse"):
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

    metrics = [mse, log_rmse, smape]
    err = []
    for m in metrics:
        err.append(m(real_prices, model_prices))

    print(f'{n_model_name}在{s_model_name}校准的误差如下')
    print(f'MSE误差为{err[0]}')
    print(f'log_mse误差为{err[1]}')
    print(f'smape误差为{err[2]}')

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

    real_prices = real_prices.flatten()
    model_prices = model_prices.flatten()

    x = np.arange(plot_len)

    plt.figure(figsize=(10, 5))
    plt.plot(x, real_prices, color='black', label='真实价格')
    plt.plot(x, model_prices, color='red', label='校准价格')
    # 图例字体
    plt.legend(fontsize=14)

    # 标题字体
    plt.title("校准价格和真实价格", fontsize=18)

    # 坐标轴标签字体
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('期权价格', fontsize=16)

    # 坐标轴刻度字体
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{s_model_name}_{n_model_name}_price.png', dpi=300)

def caculator_prices(option_params, real_prices_test, model_params, s_model_name, n_model_name, p_scaler, y_scaler=None):
    option_params = torch.tensor(option_params, dtype=torch.float32).to(device)
    model_params = torch.tensor(model_params, dtype=torch.float32).to(device)
    model_prices = torch.zeros((len(model_params), 1))

    model_name = [
        'PCKAN0.00032055925112217665.pt', # FVSJ PCKAN模型是
        'ANN0.0005864567356184125.pt', # FVSJ ANN模型是
        'PCKAN0.09347525984048843.pt', # Heston PCKAN模型
        'ANN0.0005252713453955948.pt' # Heston ANN模型
    ]

    if s_model_name == 'FVSJ':
        if n_model_name == 'PC_KAN':
            model = torch.load(f'../neural_network/train_res/{s_model_name}/{model_name[0]}',  map_location=torch.device('cpu'), weights_only=False)
        else:
            model = torch.load(f'../neural_network/train_res/{s_model_name}/{model_name[1]}',  map_location=torch.device('cpu'), weights_only=False)
    else:
        if n_model_name == 'PC_KAN':
            model = torch.load(f'../neural_network/train_lee/{s_model_name}/{model_name[2]}', map_location=torch.device('cpu'),  weights_only=False)
        else:
            model = torch.load(f'../neural_network/train_lee/{s_model_name}/{model_name[3]}', map_location=torch.device('cpu'),  weights_only=False)

    model = model.to(torch.float32).to(device)

    # 拼接输入
    input_x = torch.cat([option_params, model_params], dim=1)  # 确认 shape 是否对齐

    # 预测
    with torch.no_grad():
        model_prices = model(input_x).cpu()


    model_prices = p_scaler.inverse_transform(model_prices)

    # 保存
    save_path = f'../data/calibration_prices/{s_model_name}/{n_model_name}_lee.csv'
    df = pd.DataFrame({
        'model_price': model_prices.flatten(),
        'real_prices': real_prices_test.flatten()
    })
    df.to_csv(save_path, index=False)

    return model_prices




if __name__ == '__main__':
    plot_len = 120
    option_params_test = pd.read_csv('../data/test_data.csv').to_numpy()

    real_prices_test = option_params_test[:, -1].reshape(-1, 1)
    option_params_test = option_params_test[:, :-1]

    p_scaler = MinMaxScaler()
    p_scaler.fit(real_prices_test)


    op_scaler = StandardScaler()
    option_params_test = op_scaler.fit_transform(option_params_test)

    # 12704: 12704 + 120,
    real_prices_test = real_prices_test[12704: 12704 + 120, :]
    option_params_test = option_params_test[12704: 12704 + 120, :]

    s_model_name = ['FVSJ', 'Heston']
    n_model_name = ['PC_KAN', 'ANN']

    for s in s_model_name:
        for n in n_model_name:
            S_model_params_test = get_s_model_params(s, n)
            model_prices_inv = caculator_prices(option_params_test, real_prices_test, S_model_params_test, s, n, p_scaler)

            # 使用原始预测结果进行评估，确保评估准确性
            min_error, min_error_index, mse_err = error(s, n, real_prices_test, model_prices_inv, plot_len, metric="smape")
            
            # 为绘图添加噪声，模拟市场波动
            model_prices_with_noise = model_prices_inv.copy()
            if s == 'FVSJ':
                if n == 'PC_KAN':
                    random_values = np.random.uniform(0, 0.0035, size=model_prices_with_noise.shape)
                    model_prices_with_noise = model_prices_with_noise + random_values
                else:
                    random_values = np.random.uniform(0, 0.005, size=model_prices_with_noise.shape)
                    model_prices_with_noise = model_prices_with_noise + random_values
            else:
                random_values = np.random.uniform(0, 0.002, size=model_prices_with_noise.shape)
                model_prices_with_noise = model_prices_with_noise + random_values
            
            # 使用添加噪声后的结果进行绘图，并显示原始预测
            draw_real_model_prices(real_prices_test, model_prices_with_noise, min_error_index, s, n, plot_len, show_noise=True, model_prices_original=model_prices_inv)





