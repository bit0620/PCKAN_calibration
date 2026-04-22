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
    a, b = 1e-4, 1e-2  # 最小学习率1e-4，最大学习率1e-2（提高最大学习率）
    n1, n2, n3 = 0, 50, 200  # 0-50轮上升，50-200轮下降，200轮后保持最小值（大幅延长高学习率阶段）

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
        'batch_size': [131072],
        'num_workers': [16],  # 改为12，开启多进程数据加载（CPU核心数的1.5-2倍）
        'num_epochs': [300],  # 增加训练轮数到300，给模型更多收敛时间
        'lr_ANN': [0.1],
        'middle_dim': [30],
        'num_layers_ANN': [5],
        'lr_KAN': [0.005, 0.008, 0.01],  # 学习率：提高学习率范围，配合轮式学习率在1e-4到5e-3之间变化
        'num_layers_KAN': [2, 3, 4],  # 层数
        'middle_dim_kan': [64, 96, 128],  # 中间维度：增加96作为中间选项
        'dropout_p': [0.1, 0.15, 0.2],  # Dropout：增加0.15作为中间选项
        'weight_decay': [5e-6, 1e-5],  # 权重衰减：增加更小的正则化选项
        'output_size': [1],
        's_model_name': ['Heston'],
        'degrees': [[3, 5, 5, 5, 5], [3, 5, 5, 5, 6], [3, 4, 5, 5, 6]]  # 增加更多degrees组合
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



def train_test(model, train_iter, test_iter, num_epochs, learning_rate, weight_decay, device, net_name, pinn_params, save_dir, lambda_weight=0.1, patience=10, min_delta=1e-6):
    """
    训练模型并测试，包含早停机制和自适应权重
    
    注意：本实现采用三个自适应权重（数据损失 w_data、PDE 损失 w_pde 和边界损失 w_boundary），
          权重通过 softmax 形式学习，强制 w_data + w_pde + w_boundary = 1。
          所有损失项都使用相对形式，并进行适当的归一化处理。
    """
    import torch.optim as optim
    import torch.nn as nn
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os
    import numpy as np  # 添加numpy导入
    
    # 1. 初始化模型优化器
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)
    
    # 2. 初始化自适应权重参数 (logits 形式，用于softmax)
    # 使用三个权重，初始值设为0（softmax后为1/3）
    logit_w_data = torch.tensor([0.0], requires_grad=True, device=device)
    logit_w_pde = torch.tensor([0.0], requires_grad=True, device=device)
    logit_w_boundary = torch.tensor([0.0], requires_grad=True, device=device)
    
    # 3. 初始化权重优化器 (仅优化权重参数)
    optim_weights = optim.Adam([logit_w_data, logit_w_pde, logit_w_boundary], lr=0.01)

    train_loss = []
    test_loss = []
    
    # 早停相关变量
    best_test_loss = float('inf')
    counter = 0
    best_model_state = None

    print("开始训练 (三权重归一化自适应模式)...")

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epochs_train_loss = []
        epoch_loss_data_sum = 0.0
        epoch_loss_pde_sum = 0.0
        epoch_loss_boundary_sum = 0.0
        num_batches = 0

        for (X, Y), param in zip(train_iter, pinn_params):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            param = param[0].to(device, non_blocking=True)
            
            optim_m.zero_grad()
            Y_pred = model(X)
            
            # Data loss: 使用相对MSE
            loss_pred = torch.mean(((Y - Y_pred) / (torch.abs(Y) + 1e-8)) ** 2)
            
            # PDE loss: 确保是相对形式
            _, loss_pi = loss_function(Y, Y_pred, param, lambda_weight)
            
            # 如果loss_pi不是相对形式，需要转换
            if loss_pi.item() > 1.0:  # 经验阈值
                norm_factor = torch.mean(torch.abs(Y_pred)) + 1e-8
                loss_pi = loss_pi / norm_factor
            
            # Boundary loss: 计算边界条件损失（示例实现）
            # 这里假设边界条件是Y_pred在某些特定输入下的值应满足某种条件
            # 实际应用中需要根据具体问题定义边界条件
            loss_boundary = torch.tensor(0.0, device=device)
            if hasattr(model, 'compute_boundary_loss'):
                loss_boundary = model.compute_boundary_loss(X, Y_pred)
            else:
                # 默认实现：假设边界条件是输出值不应过大（正则化）
                loss_boundary = torch.mean(torch.abs(Y_pred)) / (torch.mean(torch.abs(Y)) + 1e-8)
            
            # 计算归一化的权重（softmax）
            logits = torch.cat([logit_w_data, logit_w_pde, logit_w_boundary])
            weights = torch.softmax(logits, dim=0)
            current_w_data = weights[0]
            current_w_pde = weights[1]
            current_w_boundary = weights[2]
            
            # 计算总损失
            loss = current_w_data * loss_pred + current_w_pde * loss_pi + current_w_boundary * loss_boundary

            loss.backward()
            optim_m.step()
            epochs_train_loss.append(loss.detach())

            epoch_loss_data_sum += loss_pred.detach().item()
            epoch_loss_pde_sum += loss_pi.detach().item()
            epoch_loss_boundary_sum += loss_boundary.detach().item()
            num_batches += 1

        scheduler.step()
        epoch_train_loss = torch.mean(torch.stack(epochs_train_loss))
        train_loss.append(epoch_train_loss.item())
        
        # --- 自适应权重更新逻辑 ---
        avg_loss_data = epoch_loss_data_sum / num_batches
        avg_loss_pde = epoch_loss_pde_sum / num_batches
        avg_loss_boundary = epoch_loss_boundary_sum / num_batches
        
        # 转换为tensor用于计算
        avg_loss_data_tensor = torch.tensor(avg_loss_data, device=device)
        avg_loss_pde_tensor = torch.tensor(avg_loss_pde, device=device)
        avg_loss_boundary_tensor = torch.tensor(avg_loss_boundary, device=device)
        
        # 计算权重更新梯度（基于论文建议）
        # 使用损失的倒数作为权重调整方向（损失越小，权重越大）
        with torch.no_grad():
            # 计算每个损失的相对重要性（归一化到[0,1]）
            total_loss = avg_loss_data + avg_loss_pde + avg_loss_boundary + 1e-8
            rel_data = avg_loss_data / total_loss
            rel_pde = avg_loss_pde / total_loss
            rel_boundary = avg_loss_boundary / total_loss
            
            # 更新logits：减少高损失项的权重，增加低损失项的权重
            # 使用负的相对损失作为梯度方向
            logit_w_data += 0.01 * (-rel_data)
            logit_w_pde += 0.01 * (-rel_pde)
            logit_w_boundary += 0.01 * (-rel_boundary)
            
            # 确保logits不会变得太大或太小
            logit_w_data = torch.clamp(logit_w_data, -5, 5)
            logit_w_pde = torch.clamp(logit_w_pde, -5, 5)
            logit_w_boundary = torch.clamp(logit_w_boundary, -5, 5)
        
        # ============================================================
        # 测试阶段
        model.eval()
        with torch.no_grad():
            epoch_test_loss = []
            for X_test, Y_test in test_iter:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                Y_pred_test = model(X_test)
                # 使用相对MSE作为测试损失
                test_loss_value = torch.mean(((Y_test - Y_pred_test) / (torch.abs(Y_test) + 1e-8)) ** 2)
                epoch_test_loss.append(test_loss_value.detach())

            avg_test_loss = torch.mean(torch.stack(epoch_test_loss))
            test_loss.append(avg_test_loss.item())

        # 早停检查
        if avg_test_loss < best_test_loss - min_delta:
            best_test_loss = avg_test_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f'早停触发，在第 {epoch+1} 轮停止训练')
                break
        
        # 计算当前权重（用于打印）
        logits = torch.cat([logit_w_data, logit_w_pde, logit_w_boundary])
        weights = torch.softmax(logits, dim=0)
        current_w_data = weights[0]
        current_w_pde = weights[1]
        current_w_boundary = weights[2]

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Total Loss: {epoch_train_loss.item():.5e} | '
                  f'Data Loss: {avg_loss_data:.5e} (w: {current_w_data.item():.2f}) | '
                  f'PDE Loss: {avg_loss_pde:.5e} (w: {current_w_pde.item():.2f}) | '
                  f'Boundary Loss: {avg_loss_boundary:.5e} (w: {current_w_boundary.item():.2f}) | '
                  f'Test Loss: {avg_test_loss.item():.5e}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss_fin = test_loss[-1]
    net_name_fin = f'{net_name}' + str(test_loss_fin)
    
    # 保存模型
    if not os.path.exists(f'../neural_network/train_lee/{save_dir}'):
        os.makedirs(f'../neural_network/train_lee/{save_dir}')
        
    torch.save(model, f'../neural_network/train_lee/{save_dir}/{net_name_fin}.pt')
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train&test_loss')
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
        train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name=s_model_name, batch_size=batch_size, num_workers=num_workers, device=device)
        model = Cheby_KAN(input_size, output_size, middle_dim_kan, degrees, num_layers_KAN, dropout_p)
        loss_kan, kan_name = train_test(model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay, device=device, net_name='PCKAN', pinn_params=pinn_loader, save_dir=s_model_name, patience=10, min_delta=1e-5)
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
        # ===================================================================================================================
