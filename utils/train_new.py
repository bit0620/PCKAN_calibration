import sys
import os
# 👇 只加这两行，下面 from neural_network... 一行不动
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from neural_network.FNN import nn_impvol
from torch.optim.lr_scheduler import LambdaLR
from neural_network.cheby_KAN import Cheby_KAN
from utils.function import *
from torch import optim
from tqdm import tqdm
# from kan import *

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei' # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


def lr_schedule(n):
    a, b = 1e-3, 1e-2
    n1, n2, n3 = 0, 50, 150

    if n <= n2:
        return (a - b)/(n1 - n2) * n - (a*n2 - b*n1) / (n1 - n2)
    elif n2 < n < n3:
        return -(a - b)/(n2 - n3) * n + (a*n2 - b*n3) / (n2 - n3)
    else:
        return a


def generate_param_combinations_csv(output_path='../data/train/train_params.csv'):
    """
    生成参数组合CSV文件

    参数:
        output_path (str): 输出CSV文件的路径，默认为'../data/train/train_params.csv'

    返回:
        None
    """
    import pandas as pd
    import os
    import json  # 添加json模块，用于处理列表类型参数
    from itertools import product

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 定义参数网格
    param_grid = {
        'batch_size': [2048],
        'num_workers': [16],  # 改为12，开启多进程数据加载（CPU核心数的1.5-2倍）
        'num_epochs': [300],  # 增加训练轮数到300，给模型更多收敛时间
        'lr_ANN': [0.1],
        'middle_dim': [30],
        'num_layers_ANN': [5],
        'lr_KAN': [0.2, 0.1],  # 学习率：提高学习率范围，配合轮式学习率在1e-4到5e-3之间变化
        'num_layers_KAN': [3, 4, 2],  # 层数
        'middle_dim_kan': [144, 64, 128, 96],  # 中间维度：增加96作为中间选项
        'dropout_p': [0.1, 0, 0.2],  # Dropout：增加0.15作为中间选项
        'weight_decay': [1e-5, 5e-6],  # 权重衰减：增加更小的正则化选项
        'output_size': [1],
        's_model_name': ['Heston'],
        'degrees': [[3, 5, 5, 5, 5]]
    }


    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))

    # 创建DataFrame
    df = pd.DataFrame(combinations, columns=keys)
    # 保存到CSV文件
    df.to_csv(output_path, index=False)
    print(f"参数组合CSV文件已生成: {output_path}")
    print(f"共生成 {len(df)} 组参数组合")

def load_params_from_csv(csv_path='../data/train/train_params.csv'):
    """从CSV文件加载训练参数"""
    import json  # 添加json模块，用于处理列表类型参数

    df = pd.read_csv(csv_path)
    params_list = []
    for _, row in df.iterrows():
        params = {
            'batch_size': int(row['batch_size']),
            'num_workers': int(row['num_workers']),
            'num_epochs': int(row['num_epochs']),
            'lr_ANN': float(row['lr_ANN']),
            'middle_dim': int(row['middle_dim']),
            'num_layers_ANN': int(row['num_layers_ANN']),
            'lr_KAN': float(row['lr_KAN']),
            'num_layers_KAN': int(row['num_layers_KAN']),
            'middle_dim_kan': int(row['middle_dim_kan']),
            'dropout_p': float(row['dropout_p']),
            'weight_decay': float(row['weight_decay']),
            'output_size': int(row['output_size']),
            's_model_name': row['s_model_name'],
            'degrees': json.loads(row['degrees'].replace("'", '"'))  # 添加degrees参数处理
        }
        params_list.append(params)
    return params_list

def train_test(model, train_iter, test_iter, num_epochs, learning_rate, weight_decay, device, net_name, pinn_params, save_dir, lambda_weight=0.1,
               beta_pde=1.0, bvp1_weight=1.0, bvp2_weight=1.0,
               maturity_feature=(0, 1), spot_feature=(2, 2), seq_all_steps=True,
               maturity_mean=None, maturity_std=None, spot_mean=None, spot_std=None,
               target_mean=None, target_std=None, y_scaler=None):
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)

    train_loss = []
    test_loss = []

    for epoch in tqdm(range(num_epochs)):
        # 训练阶段
        model.train()
        epochs_train_loss = []
        for (X, Y), param in zip(train_iter, pinn_params):
            X = X.to(device)
            Y = Y.to(device)
            # 修改：正确处理param列表中的张量
            if isinstance(param, list):
                # 如果param是列表，堆叠其中的张量
                param = torch.stack([p.to(device) if not p.is_cuda else p for p in param])
            elif isinstance(param, tuple):
                # 如果param是元组，取第一个元素
                param = param[0].to(device)
            # 检查param的形状，确保它是(batch_size, 6)
            batch_size = X.shape[0]
            if param.dim() == 1:
                # 如果param是一维张量，说明只有一个样本，需要扩展维度
                param = param.unsqueeze(0).expand(batch_size, -1)
            elif param.dim() > 2:
                # 如果param是多维张量，需要展平
                param = param.reshape(-1, param.shape[-1])
                if param.shape[0] != batch_size:
                    param = param[:batch_size, :]
            elif param.shape[0] != batch_size:
                # 如果param的第一个维度不等于batch_size，需要调整
                param = param.unsqueeze(0).expand(batch_size, -1)
            optim_m.zero_grad()

            # 使用loss_function计算损失
            X.requires_grad_(True)
            Y_hat = model(X)
            total_loss = loss_function(Y_hat, Y, X, param, beta_pde, target_mean, target_std)

            total_loss.backward()
            optim_m.step()
            epochs_train_loss.append(total_loss.detach())

        scheduler.step()
        epoch_train_loss = torch.mean(torch.stack(epochs_train_loss))
        train_loss.append(epoch_train_loss.item())

        # 测试阶段
        model.eval()
        epoch_test_loss = []
        for (X_test, Y_test), param_test in zip(test_iter, pinn_params):
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)
            # 修改：正确处理param_test列表中的张量
            if isinstance(param_test, list):
                # 如果param_test是列表，堆叠其中的张量
                param_test = torch.stack([p.to(device) if not p.is_cuda else p for p in param_test])
            elif isinstance(param_test, tuple):
                # 如果param_test是元组，取第一个元素
                param_test = param_test[0].to(device)
            # 检查param_test的形状，确保它是(batch_size, 6)
            if param_test.dim() == 1:
                # 如果param_test是一维张量，说明只有一个样本，需要扩展维度
                param_test = param_test.unsqueeze(0).expand(X_test.shape[0], -1)
            elif param_test.dim() > 2:
                # 如果param_test是多维张量，需要展平
                param_test = param_test.reshape(-1, param_test.shape[-1])
                if param_test.shape[0] != X_test.shape[0]:
                    param_test = param_test[:X_test.shape[0], :]
            elif param_test.shape[0] != X_test.shape[0]:
                # 如果param_test的第一个维度不等于batch_size，需要调整
                param_test = param_test.unsqueeze(0).expand(X_test.shape[0], -1)

            # 测试时也使用loss_function
            with torch.set_grad_enabled(True):
                X_test.requires_grad_(True)
                Y_test_hat = model(X_test)
                test_total_loss = loss_function(Y_test_hat, Y_test, X_test, param_test, beta_pde, target_mean, target_std)

            epoch_test_loss.append(test_total_loss.detach())

        avg_test_loss = torch.mean(torch.stack(epoch_test_loss))
        test_loss.append(avg_test_loss.item())

        if epoch % 10 == 0:
            print(f'训练误差为{train_loss[-1]}', f'测试误差为{test_loss[-1]}')

    test_loss_fin = test_loss[-1]
    net_name_fin = f'{net_name}' + str(test_loss_fin)
    train_loss = torch.tensor(train_loss).numpy()
    test_loss = torch.tensor(test_loss).numpy()
    torch.save(model, f'../neural_network/train_res/{save_dir}/{net_name_fin}.pt')

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='训练误差')
    plt.plot(test_loss, label='测试误差')
    plt.xlabel('批量')
    plt.ylabel('损失')
    plt.title('训练和测试误差曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

    return test_loss_fin, net_name_fin




if __name__ == '__main__':
    # 检查参数文件是否存在，不存在则生成
    csv_path = '../data/train/train_params.csv'
    if not os.path.exists(csv_path):
        print(f"参数文件 {csv_path} 不存在，正在生成...")
        generate_param_combinations_csv(csv_path)

    # 从CSV文件加载参数
    params_list = load_params_from_csv(csv_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('当前设备', device)

    # 循环处理每组参数
    for idx, params in enumerate(params_list):
        print(f"\n开始训练第 {idx+1} 组参数...")

        # 从参数字典中获取参数
        batch_size = params['batch_size']
        num_workers = params['num_workers']
        num_epochs = params['num_epochs']

        # ANN参数
        lr_ANN = params['lr_ANN']
        middle_dim = params['middle_dim']
        num_layers_ANN = params['num_layers_ANN']

        # KAN参数
        lr_KAN = params['lr_KAN']
        num_layers_KAN = params['num_layers_KAN']
        middle_dim_kan = params['middle_dim_kan']
        dropout_p = params['dropout_p']
        weight_decay = params['weight_decay']
        degrees = params['degrees']  # 添加degrees参数赋值

        # 模型参数
        output_size = params['output_size']
        s_model_name = params['s_model_name']

        # cheby_kan神经网络
        # ===================================================================================================================
        train_loader, test_loader, pinn_loader, input_size, scaler_params = load_data(
            s_model_name=s_model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
        model = Cheby_KAN(input_size, output_size, middle_dim_kan, degrees, num_layers_KAN, dropout_p)
        loss_kan, kan_name = train_test(
            model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay,
            device=device, net_name='PCKAN', pinn_params=pinn_loader, save_dir=s_model_name,
            beta_pde=1.0, bvp1_weight=1.0, bvp2_weight=1.0,
            maturity_feature=(0, 1), spot_feature=(2, 2), seq_all_steps=True,
            **scaler_params
        )
        params_list_kan = [[lr_KAN, num_layers_KAN, num_epochs, middle_dim_kan, degrees, num_epochs, loss_kan, dropout_p, weight_decay, kan_name]]
        params_pd_kan = pd.DataFrame(params_list_kan, columns=['Learning_Rate', 'Num_Layers', "train_epochs", "middle_dim", 'Degrees', 'Num_epochs', 'Loss', "dropout_p", "weight_decay", 'Net_Name'])
        params_pd_kan.to_csv('../data/train/train_params_kan_pinn_res.csv', mode='a', index=False)
        # ===================================================================================================================

        # 全连接
        # ==================================================================================================================
        # train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name=s_model_name, batch_size=batch_size, num_workers=num_workers, device=device)
        # model = nn_impvol(input_size, output_size, middle_dim, num_layers_ANN)
        # loss_ann, ann_name = train_test(model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay, device=device, net_name='ANN', pinn_params=pinn_loader, save_dir=s_model_name)
        # params_list_kan = [[lr_ANN, num_layers_ANN, num_epochs, middle_dim, num_epochs, loss_ann, dropout_p, weight_decay, ann_name]]
        # params_pd_kan = pd.DataFrame(params_list_kan, columns=['Learning_Rate', 'Num_Layers', "train_epochs", "middle_dim", 'Num_epochs', 'Loss', "dropout_p", "weight_decay", 'Net_Name'])
        # params_pd_kan.to_csv('../data/train/train_params_ann_res.csv', mode='a', index=False)
        # ===================================================================================================================

        # kan神经网络
        # ===================================================================================================================
        # train_input, train_label, test_input, test_label = load_data(type="normal_kan")
        # model = KAN(width=[9, 5, 5, 5, 9], grid=3, k=3, seed=42, device=device)
        # dataset = {
        #     "train_input": train_input,
        #     "train_label": train_label,
        #     "test_input": test_input,
        #     "test_label": test_label}
        # model(dataset['train_input'])
        # model.fit(dataset, opt="LBFGS", steps=200, lamb=0.001)
