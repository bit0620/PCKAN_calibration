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


def train_test(model, train_iter, test_iter, num_epochs, learning_rate, weight_decay, device, net_name, pinn_params, save_dir, lambda_weight=0.1):
    optim_m = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    # scheduler = MultiStepLR(optimizer=optim_m, milestones=[20, 50], gamma=0.1)
    scheduler = LambdaLR(optimizer=optim_m, lr_lambda=lr_schedule)
    model = model.to(device)

    train_loss = []
    test_loss = []

    for epoch in tqdm(range(num_epochs)):
        # 训练阶段
        # ============================================================
        model.train()
        epochs_train_loss = []
        for (X, Y), param in zip(train_iter, pinn_params):
            X = X.to(device)
            Y = Y.to(device)
            param = param[0].to(device)
            optim_m.zero_grad()
            Y_pred = model(X)
            # loss = nn.MSELoss()(Y, Y_pred)
            loss = loss_function(Y, Y_pred, param, lambda_weight)
            loss.backward()
            optim_m.step()
            epochs_train_loss.append(loss.detach())

        scheduler.step()
        epoch_train_loss = torch.mean(torch.stack(epochs_train_loss))
        train_loss.append(epoch_train_loss.item())
        # ============================================================

        # ============================================================
        # 测试阶段
        model.eval()
        with torch.no_grad():
            epoch_test_loss = []
            for X_test, Y_test in test_iter:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                Y_pred_test = model(X_test)
                test_loss_value = loss_fn(Y_test, Y_pred_test)
                epoch_test_loss.append(test_loss_value.detach())

            avg_test_loss = torch.mean(torch.stack(epoch_test_loss))
            test_loss.append(avg_test_loss.item())

        # scheduler.step(avg_test_loss)
        # ============================================================
        if epoch % 10 == 0:
            print(f'训练误差为{train_loss[-1]}', f'测试误差为{test_loss[-1]}]')

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
    batch_size, num_workers, num_epochs, = 32, 0, 200
    lr_ANN, middle_dim, num_layers_ANN = 0.1, 30, 5
    lr_KAN, num_layers_KAN, middle_dim_kan, dropout_p, degrees, weight_decay = 0.2, 3, 72, 0, [3, 5, 5, 5, 5], 1e-5
    FFT_grid_size = 32
    output_size = 1
    s_model_name = 'Heston'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('当前设备', device)

    # cheby_kan神经网络
    # ===================================================================================================================
    train_loader, test_loader, pinn_loader, input_size = load_data(s_model_name=s_model_name, batch_size=batch_size, num_workers=num_workers, device=device)
    model = Cheby_KAN(input_size, output_size, middle_dim_kan, degrees, num_layers_KAN, dropout_p)
    loss_kan, kan_name = train_test(model, train_loader, test_loader, num_epochs, lr_KAN, weight_decay, device=device, net_name='PCKAN', pinn_params=pinn_loader, save_dir=s_model_name)
    params_list_kan = [[lr_KAN, num_layers_KAN, num_epochs, middle_dim_kan, degrees, num_epochs, loss_kan, dropout_p, weight_decay, kan_name]]
    params_pd_kan = pd.DataFrame(params_list_kan, columns=['Learning_Rate', 'Num_Layers', "train_epochs", "middle_dim", 'Degrees', 'Num_epochs', 'Loss', "dropout_p", "weight_decay", 'Net_Name'])
    params_pd_kan.to_csv('../data/train/train_params_kan_pinn.csv', mode='a', index=False)
    # ===================================================================================================================

    # 全连接
    # ==================================================================================================================
    # output_grid, train_iter, test_iter, input_size, output_size = load_data(type='neural_network', batch_size=batch_size, num_works=num_works, device=device)
    # model = nn_impvol(input_size=input_size, output_size=output_size, middle_size=middle_dim, num_layers=num_layers_ANN)
    # pinn_params = load_pinn_data(batch_size, output_size, num_works)
    # loss_ann, Ann_name = train_test(model, train_iter, test_iter, num_epochs, lr_ANN, weight_decay, device=device, net_name='NN', pinn_params=pinn_params)
    # params_list_ann = [[lr_ANN, num_epochs, middle_dim, num_layers_ANN, loss_ann, Ann_name]]
    # params_pd_ann = pd.DataFrame(params_list_ann, columns=['Learning_Rate', 'Num_epochs', 'middle_size', 'Num_Layers', 'Loss', 'Net_Name'])
    # params_pd_ann.to_csv('../data/train/train_params_ann.csv', mode='a', index=False)
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

