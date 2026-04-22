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
    """
    适配300轮训练的学习率调度策略
    - 阶段1（0-50轮）：线性预热，从1e-4上升到5e-3
    - 阶段2（50-250轮）：余弦衰减，从5e-3下降到1e-4
    - 阶段3（250-300轮）：保持最小学习率1e-4
    """
    a, b = 1e-3, 1e-1  # 最小和最大学习率
    n1, n2, n3 = 0, 50, 250  # 预热、衰减和保持阶段的转折点

    if n <= n1:
        # 预热阶段：线性上升
        if n1 == 0:
            return b  # 避免除零错误
        return a + (b - a) * n / n1
    elif n1 < n <= n2:
        # 衰减阶段：余弦衰减
        return a + 0.5 * (b - a) * (1 + np.cos(np.pi * (n - n1) / (n2 - n1)))
    else:
        # 保持阶段：使用最小学习率
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
    
    # 定义参数网格（按论文要求调整）
    param_grid = {
        'batch_size': [8192],  # 论文推荐：从小批量开始验证，逐步增大
        'num_workers': [16],  # 数据加载进程数
        'num_epochs': [300],  # 论文推荐：固定大迭代次数
        'lr_KAN': [5e-2, 3e-2, 1e-2 ],  # 学习率范围：配合轮式学习率在1e-4到5e-3之间变化
        'num_layers_KAN': [2, 3, 4],  # 网络层数
        'middle_dim_kan': [32, 64, 96, 128],  # 中间维度
        'dropout_p': [0.0, 0.1, 0.2],  # Dropout概率
        'weight_decay': [1e-6, 5e-6, 1e-5],  # 权重衰减（L2正则化）
        'output_size': [1],
        's_model_name': ['Heston'],
        'degrees': [[3, 5, 5, 5, 5], [3, 5, 5, 5, 6], [3, 4, 5, 5, 6], [4, 5, 5, 5, 6]], # 切比雪夫多项式阶数
        # ANN参数（即使不使用，也需要定义以避免错误）
        'lr_ANN': [1e-4],
        'middle_dim': [32],
        'num_layers_ANN': [2]
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
    
    
    注意：本实现采用三个自适应权重（数据损失 w_data、PDE 损失 w_pde 和边界损失 w_boundary），
          权重通过梯度范数平衡机制更新（每1000步），符合论文Algorithm 1。
    """
    import torch.optim as optim
    import torch.nn as nn
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os
    import numpy as np  # 添加numpy导入




    # === 新增: 计算训练集全局统计量用于无量纲化 ===
    print("正在计算训练集统计量用于无量纲化...")
    X_train_all = []
    Y_train_all = []
    pinn_params_all = []
    
    # 重新创建 train_iter 和 pinn_params 的副本（因为它们是一次性迭代器）
    # 注意：这要求 DataLoader 可重复迭代（设置 drop_last=False, shuffle=False for this pass）
    # 实际中应确保 train_loader 可多次迭代
    train_iter_for_stats, pinn_params_for_stats = zip(*[(batch, param) for (batch, param) in zip(train_iter, pinn_params)])
    
    for (X_batch, Y_batch), param_batch in zip(train_iter_for_stats, pinn_params_for_stats):
        X_train_all.append(X_batch)
        Y_train_all.append(Y_batch)
        pinn_params_all.append(param_batch[0])  # param is list of one tensor
    
    X_train_tensor = torch.cat(X_train_all, dim=0).to(device)
    Y_train_tensor = torch.cat(Y_train_all, dim=0).to(device)
    pinn_params_tensor = torch.cat(pinn_params_all, dim=0).to(device)
    
    # 计算均值和标准差（沿 batch 维度）
    X_mean = X_train_tensor.mean(dim=0, keepdim=True)
    X_std = X_train_tensor.std(dim=0, keepdim=True) + 1e-8
    Y_mean = Y_train_tensor.mean(dim=0, keepdim=True)
    Y_std = Y_train_tensor.std(dim=0, keepdim=True) + 1e-8
    param_mean = pinn_params_tensor.mean(dim=0, keepdim=True)
    param_std = pinn_params_tensor.std(dim=0, keepdim=True) + 1e-8
    
    # 新增：记录spot的全局最小最大值用于边界条件
    spot_all = X_train_tensor[:, 0]
    spot_min = spot_all.min().item()
    spot_max = spot_all.max().item()
    
    # 释放内存
    del X_train_all, Y_train_all, pinn_params_all, X_train_tensor, Y_train_tensor, pinn_params_tensor
    torch.cuda.empty_cache() if device == 'cuda' else None
    # ===========================================
    
    # 1. 初始化模型优化器
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)
    
    # 2. 初始化自适应权重参数（梯度范数平衡机制）
    lambda_data = torch.tensor(1.0, device=device, requires_grad=False)
    lambda_pde = torch.tensor(1.0, device=device, requires_grad=False)
    lambda_boundary = torch.tensor(1.0, device=device, requires_grad=False)
    
    # 3. 初始化全局step计数器
    global_step = 0

    train_loss = []
    test_loss = []
    
    # best_test_loss = float('inf')
    # counter = 0
    # best_model_state = None

    print("开始训练 (梯度范数平衡自适应权重模式)...")

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epochs_train_loss = []
        epoch_loss_data_sum = 0.0
        epoch_loss_pde_sum = 0.0
        epoch_loss_boundary_sum = 0.0
        num_batches = 0

        # 重新创建迭代器（因为之前已消耗）
        # 注意：实际中应传递可重复迭代的 DataLoader，而非一次性迭代器
        train_iter_epoch, pinn_params_epoch = zip(*[(batch, param) for (batch, param) in zip(train_iter, pinn_params)])
        
        for (X, Y), param in zip(train_iter_epoch, pinn_params_epoch):
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            param = param[0].to(device, non_blocking=True)
            
            # === 新增: 无量纲化 ===
            X_norm = (X - X_mean) / X_std
            Y_norm = (Y - Y_mean) / Y_std
            param_norm = (param - param_mean) / param_std
            # =====================
            
            # 清零优化器梯度
            optim_m.zero_grad()
            
            Y_pred_norm = model(X_norm)
            
            # Data loss: 使用标准 MSE（因已在无量纲空间）
            loss_pred = torch.mean((Y_norm - Y_pred_norm) ** 2)
            
            # === 关键修正: PDE 损失在物理空间计算 ===
            # 反归一化到物理空间
            Y_pred_phys = Y_pred_norm * Y_std + Y_mean
            Y_phys = Y_norm * Y_std + Y_mean
            param_phys = param_norm * param_std + param_mean

            # 使用物理空间的值计算PDE残差
            _, loss_pi = loss_function(Y_pred_phys, Y_phys, param_phys, lambda_weight)
            # ===========================================
            
            # Boundary loss: 使用模型的compute_boundary_loss方法
            loss_boundary = model.compute_boundary_loss(X_norm, Y_pred_norm, X_mean, X_std, Y_mean, Y_std, spot_min, spot_max)
            
            # 旧的边界损失计算代码已删除，使用模型的compute_boundary_loss方法
            
            # 边界损失计算已整合到模型的compute_boundary_loss方法中
            # 旧的边界损失计算代码已删除
            
            # Boundary loss: 使用模型的compute_boundary_loss方法
            loss_boundary = model.compute_boundary_loss(X_norm, Y_pred_norm, X_mean, X_std, Y_mean, Y_std, spot_min, spot_max)
            
            # 每10步更新自适应权重（梯度范数平衡）
            if global_step % 10 == 0:
                # 计算各损失梯度范数
                grad_data = torch.autograd.grad(loss_pred, model.parameters(), retain_graph=True, create_graph=True)
                grad_boundary = torch.autograd.grad(loss_boundary, model.parameters(), create_graph=True)
                
                norm_data = torch.stack([g.norm() for g in grad_data]).mean()
                # 计算PDE损失的梯度范数
                grad_pde = torch.autograd.grad(loss_pi, model.parameters(), retain_graph=True, create_graph=True)
                norm_pde = torch.stack([g.norm() for g in grad_pde]).mean()
                norm_boundary = torch.stack([g.norm() for g in grad_boundary]).mean()
                
                # 更新权重（逆梯度范数）
                total_norm = norm_data + norm_pde + norm_boundary
                lambda_data.data = torch.tensor(total_norm / (norm_data + 1e-8), device=device)
                lambda_pde.data = torch.tensor(total_norm / (norm_pde + 1e-8), device=device)
                lambda_boundary.data = torch.tensor(total_norm / (norm_boundary + 1e-8), device=device)
            
            # 计算总损失
            loss = lambda_data * loss_pred + lambda_pde * loss_pi + lambda_boundary * loss_boundary

            # 对模型参数求导
            loss.backward()
            optim_m.step()
            
            epochs_train_loss.append(loss.detach())

            epoch_loss_data_sum += loss_pred.detach().item()
            epoch_loss_pde_sum += loss_pi.detach().item()
            epoch_loss_boundary_sum += loss_boundary.detach().item()
            num_batches += 1
            global_step += 1

        scheduler.step()
        epoch_train_loss = torch.mean(torch.stack(epochs_train_loss))
        train_loss.append(epoch_train_loss.item())
        
        # ============================================================
        # 测试阶段
        model.eval()
        with torch.no_grad():
            epoch_test_loss = []
            for X_test, Y_test in test_iter:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                # === 新增: 测试集无量纲化 ===
                X_test_norm = (X_test - X_mean) / X_std
                Y_test_norm = (Y_test - Y_mean) / Y_std
                # =========================
                Y_pred_test_norm = model(X_test_norm)
                # 使用标准MSE作为测试损失（无量纲空间）
                test_loss_value = torch.mean((Y_test_norm - Y_pred_test_norm) ** 2)
                epoch_test_loss.append(test_loss_value.detach())

            avg_test_loss = torch.mean(torch.stack(epoch_test_loss))
            test_loss.append(avg_test_loss.item())

        # if avg_test_loss < best_test_loss - min_delta:
        #     best_test_loss = avg_test_loss
        #     counter = 0
        #     best_model_state = model.state_dict().copy()
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f'早停触发，在第 {epoch+1} 轮停止训练')
        #         break
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Total Loss: {epoch_train_loss.item():.5e} | '
                  f'Data Loss: {epoch_loss_data_sum / num_batches:.5e} (w: {lambda_data.item():.2f}) | '
                  f'PDE Loss: {epoch_loss_pde_sum / num_batches:.5e} (w: {lambda_pde.item():.2f}) | '
                  f'Boundary Loss: {epoch_loss_boundary_sum / num_batches:.5e} (w: {lambda_boundary.item():.2f}) | '
                  f'Test Loss: {avg_test_loss.item():.5e}')

    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)
    
    test_loss_fin = test_loss[-1]
    net_name_fin = f'{net_name}' + str(test_loss_fin)
    
    # 保存模型（保存最终模型）
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