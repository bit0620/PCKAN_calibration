import sys
import os
# 获取当前文件所在目录（A目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（project目录）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到模块搜索路径
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
    a, b = 1e-4, 1e-2
    n1, n2, n3 = 0, 25, 150

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
        'batch_size': [32],
        'num_workers': [0],
        'num_epochs': [200],
        'lr_ANN': [0.1],
        'middle_dim': [30],
        'num_layers_ANN': [5],
        'lr_KAN': [0.001, 0.01],  # 学习率
        'num_layers_KAN': [3, 4],  # 层数
        'middle_dim_kan': [64, 128],  # 中间维度
        'dropout_p': [0.1, 0.2],  # Dropout
        'weight_decay': [1e-05],  # 权重衰减
        'output_size': [1],
        's_model_name': ['Heston'],
        'degrees': [[3, 5, 5, 5, 5], [3, 5, 5, 5, 6]]  # 添加degrees参数
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
    
    参数:
        model: 要训练的模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        num_epochs: 最大训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        device: 训练设备
        net_name: 网络名称
        pinn_params: PINN参数
        save_dir: 保存目录
        lambda_weight: 初始损失权重参考(自适应模式下主要用于初始化或忽略)
        patience: 早停耐心值
        min_delta: 被认为是改善的最小变化量
    """
    import torch.optim as optim
    import torch.nn as nn
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os
    
    # 1. 初始化模型优化器
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)
    
    # 2. 初始化自适应权重参数 (log 形式)
    # 初始权重设为 log(1) = 0，即初始权重为 1.0
    log_w_data = torch.zeros(1, requires_grad=True, device=device)
    log_w_pde = torch.zeros(1, requires_grad=True, device=device)
    
    # 3. 初始化权重优化器 (仅优化权重参数)
    # 权重的学习率通常可以稍大，以便快速响应损失变化
    optim_weights = optim.Adam([log_w_data, log_w_pde], lr=0.01)
    
    train_loss = []
    test_loss = []
    
    # 早停相关变量
    best_test_loss = float('inf')
    counter = 0
    best_model_state = None
    
    print("开始训练 (自适应权重模式)...")

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epochs_train_loss = []
        
        # 用于记录当前 epoch 的平均损失，用于权重更新
        epoch_loss_data_sum = 0.0
        epoch_loss_pde_sum = 0.0
        num_batches = 0

        for (X, Y), param in zip(train_iter, pinn_params):
            X = X.to(device)
            Y = Y.to(device)
            param = param[0].to(device)
            
            # --- A. 计算各项原始损失 ---
            optim_m.zero_grad()
            optim_weights.zero_grad()
            
            Y_pred = model(X)
            
            # 1. 数据损失
            loss_data = nn.MSELoss()(Y, Y_pred)
            
                        # 2. 物理损失
            # 调用 BS_PDE 计算方程残差
            # 注意：BS_PDE 内部已经处理了 requires_grad 和导数计算
            price_pde = BS_PDE(param)
            
            # 处理可能的 NaN 值，防止梯度爆炸或消失
            price_pde = torch.where(torch.isnan(price_pde), torch.zeros_like(price_pde), price_pde)
            
            # 物理损失目标是让 PDE 残差趋近于 0
            pi_loss_target = torch.zeros_like(price_pde)
            loss_pde = nn.MSELoss()(price_pde, pi_loss_target)

            
            # 累加损失用于后续计算平均值
            epoch_loss_data_sum += loss_data.item()
            epoch_loss_pde_sum += loss_pde.item()
            num_batches += 1
            
            # --- B. 计算加权总损失 ---
            w_data = torch.exp(log_w_data)
            w_pde = torch.exp(log_w_pde)
            
            total_loss = w_data * loss_data + w_pde * loss_pde
            
            # --- C. 反向传播更新网络参数 ---
            total_loss.backward(retain_graph=True) # 保留计算图用于权重更新
            optim_m.step()
            
            epochs_train_loss.append(total_loss.detach())

        scheduler.step()
        epoch_train_loss = torch.mean(torch.stack(epochs_train_loss))
        train_loss.append(epoch_train_loss.item())
        
        # --- D. 自适应权重更新逻辑 ---
        # 计算当前 epoch 的平均损失
        avg_loss_data = epoch_loss_data_sum / num_batches
        avg_loss_pde = epoch_loss_pde_sum / num_batches
        
        # 计算当前加权损失
        current_w_data = torch.exp(log_w_data)
        current_w_pde = torch.exp(log_w_pde)
        weighted_loss_data = current_w_data * avg_loss_data
        weighted_loss_pde = current_w_pde * avg_loss_pde
        
        # 目标：让两项加权损失趋于平衡
        # 计算目标值（两者的平均值）
        mean_weighted_loss = (weighted_loss_data + weighted_loss_pde) / 2.0
        
        # 定义权重优化的损失函数：使得各项加权损失与平均值的差距最小
        # 注意：这里我们要更新的是 log_w，所以直接对 log_w 求导
        loss_weights = (weighted_loss_data - mean_weighted_loss)**2 + \
                       (weighted_loss_pde - mean_weighted_loss)**2
        
        # 对权重进行反向传播
        loss_weights.backward()
        optim_weights.step()
        
        # ============================================================
        # 测试阶段
        model.eval()
        with torch.no_grad():
            epoch_test_loss = []
            for X_test, Y_test in test_iter:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                Y_pred_test = model(X_test)
                # 测试时通常只看数据损失，或者看总损失（使用训练好的权重）
                # 这里为了简单，只看 MSE
                test_loss_value = nn.MSELoss()(Y_test, Y_pred_test)
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
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Total Loss: {epoch_train_loss.item():.5e} | '
                  f'Data Loss: {avg_loss_data:.5e} (w: {current_w_data.item():.2f}) | '
                  f'PDE Loss: {avg_loss_pde:.5e} (w: {current_w_pde.item():.2f}) | '
                  f'Test Loss: {avg_test_loss.item():.5e}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss_fin = test_loss[-1]
    net_name_fin = f'{net_name}' + str(test_loss_fin)
    
    # 保存模型
    # 确保目录存在
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
        loss_kan, kan_name = train_test(model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay, device=device, net_name='PCKAN', pinn_params=pinn_loader, save_dir=s_model_name)
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
