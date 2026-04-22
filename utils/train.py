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


# utils/train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

# ... (保留原有的 lr_schedule 函数定义，如果有的话) ...

def train_test(model, train_iter, test_iter, num_epochs, learning_rate, weight_decay, device, net_name, pinn_params, save_dir, lambda_weight=0.1):
    # 1. 初始化模型优化器
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)
    
    # 2. 初始化自适应权重参数 (log 形式)
    # 初始权重设为 log(1) = 0
    log_w_data = torch.zeros(1, requires_grad=True, device=device)
    log_w_pde = torch.zeros(1, requires_grad=True, device=device)
    
    # 3. 初始化权重优化器 (仅优化权重参数)
    optim_weights = optim.Adam([log_w_data, log_w_pde], lr=0.01) # 权重的学习率通常可以稍大
    
    train_loss = []
    test_loss = []
    
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
            
            # 调用修改后的 loss_function，获取分离的损失
            loss_data, loss_pde = loss_function(Y_pred, Y, param, lambda_weight)
            
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
        
        # 打印信息
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Total Loss: {epoch_train_loss.item():.5e} | '
                  f'Data Loss: {avg_loss_data:.5e} (w: {current_w_data.item():.2f}) | '
                  f'PDE Loss: {avg_loss_pde:.5e} (w: {current_w_pde.item():.2f})')

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

    test_loss_fin = test_loss[-1]
    net_name_fin = f'{net_name}' + str(test_loss_fin)
    
    # 保存模型
    torch.save(model, f'../neural_network/train_res/{save_dir}/{net_name_fin}.pt')
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return test_loss_fin, net_name_fin



if __name__ == '__main__':
    batch_size, num_workers, num_epochs, = 32, 0, 200
    lr_ANN, middle_dim, num_layers_ANN = 0.1, 30, 5
    lr_KAN, num_layers_KAN, middle_dim_kan, dropout_p, degrees, weight_decay = 0.2, 3, 144, 0, [3, 5, 5, 5, 5], 1e-5
    FFT_grid_size = 32
    output_size = 1
    s_model_name = 'Heston'
    # s_model_name = 'FVSJ'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('当前设备', device)

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
    train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name=s_model_name, batch_size=batch_size, num_workers=num_workers, device=device)
    model = nn_impvol(input_size, output_size, middle_dim, num_layers_ANN)
    loss_ann, ann_name = train_test(model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay, device=device, net_name='ANN', pinn_params=pinn_loader, save_dir=s_model_name)
    params_list_kan = [[lr_ANN, num_layers_ANN, num_epochs, middle_dim, num_epochs, loss_ann, dropout_p, weight_decay, ann_name]]
    params_pd_kan = pd.DataFrame(params_list_kan, columns=['Learning_Rate', 'Num_Layers', "train_epochs", "middle_dim", 'Num_epochs', 'Loss', "dropout_p", "weight_decay", 'Net_Name'])
    params_pd_kan.to_csv('../data/train/train_params_ann_res.csv', mode='a', index=False)
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