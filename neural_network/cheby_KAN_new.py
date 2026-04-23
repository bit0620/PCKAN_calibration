from torch import nn
import torch
import math

class ChebyshevGeLU(nn.Module):
    """
    Chebyshev多项式与GeLU的结合，用于解决谱偏置问题
    """
    def __init__(self, degree=5):
        super(ChebyshevGeLU, self).__init__()
        self.degree = degree

    def forward(self, x):
        # 首先应用tanh将输入映射到[-1,1]范围
        x_tanh = torch.tanh(x)

        # 计算切比雪夫多项式（第一类）
        cheby_list = []
        cheby_0 = torch.ones_like(x_tanh)  # T_0(x) = 1
        cheby_list.append(cheby_0)

        if self.degree > 0:
            cheby_1 = x_tanh  # T_1(x) = x
            cheby_list.append(cheby_1)

        for n in range(2, self.degree + 1):
            cheby_n = 2 * x_tanh * cheby_list[n - 1] - cheby_list[n - 2]
            cheby_list.append(cheby_n)

        # 将切比雪夫多项式作为输入，应用GeLU
        cheby_stack = torch.stack(cheby_list, dim=-1)  # (batch_size, ..., degree+1)

        # 对每个切比雪夫多项式项应用GeLU并求和
        output = torch.nn.functional.gelu(cheby_stack).sum(dim=-1)

        return output

class FourierFeatures(nn.Module):
    """
    Fourier特征层，用于解决谱偏置问题
    """
    def __init__(self, input_dim, num_features=64, scale=1.0):
        super(FourierFeatures, self).__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.register_buffer('B', torch.randn(input_dim, num_features) * scale)

    def forward(self, x):
        # x: (batch_size, input_dim)
        # 计算Fourier特征: cos(2πBx)
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Cheby2KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(Cheby2KANLayer, self).__init__()
        self.input_dim = input_dim  # n_{l-1}
        self.outdim = output_dim   # n_{l}
        self.degree = degree       # D_l

        # initialize W^{(l)}_{q,p}
        self.W = nn.Parameter(torch.empty(self.outdim, self.input_dim))
        std_W = torch.sqrt(torch.tensor(5.0 / (self.input_dim + self.outdim)))
        nn.init.normal_(self.W, mean=0.0, std=std_W)

        # initialize 	ilde{W}^{(l)}_{q,p,n}
        self.tilde_W = nn.Parameter(torch.empty(self.outdim, self.input_dim, self.degree + 1))
        std_tilde_W = 1.0 / (self.degree + 1)
        nn.init.normal_(self.tilde_W, mean=0.1, std=std_tilde_W)


    def forward(self, x):
        # Reshape input to (batch_size, n_{l-1})
        # x = x.view(-1, self.input_dim)

        # Apply tanh to get within [-1, 1]
        x_tanh = torch.tanh(x)

        batch_size = x.size(0)

        # compute Chebyshev polynomials (2nd kind) up to specified degree
        cheby_list = []
        cheby_0 = torch.ones(batch_size, self.input_dim, device=x.device)  # U_0(x) = 1
        cheby_list.append(cheby_0)

        if self.degree > 0:
            cheby_1 = 2 * x_tanh  # U_1(x) = 2x
            cheby_list.append(cheby_1)
        for n in range(2, self.degree + 1):
            cheby_n = 2 * x_tanh * cheby_list[n - 1] - cheby_list[n - 2]  # U_n(x) = 2xU_{n-1}(x) - U_{n-2}(x)
            cheby_list.append(cheby_n)

        cheby = torch.stack(cheby_list, dim=2)

        # s_{b,q,p} = sum_{n=0}^{D_l} 	ilde{W}_{q,p,n} * U_n(x_p)
        s = torch.einsum('bpn, qpn->bqp', cheby, self.tilde_W)
        y = torch.einsum('bqp, qp->bq', s, self.W)

        return y

class Cheby_KAN(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, degrees, num_layers, dropout_prob, use_fourier=True, use_cheby_gelu=True):
        super(Cheby_KAN, self).__init__()
        self.inpt_dim = input_dim
        self.output_dim = output_dim
        self.middle_dim = middle_dim
        self.degrees = degrees
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.use_fourier = use_fourier
        self.use_cheby_gelu = use_cheby_gelu

        self.layers = nn.ModuleList()

        # 可选：添加Fourier特征层
        if self.use_fourier:
            self.fourier = FourierFeatures(self.inpt_dim, num_features=32, scale=1.0)
            current_dim = self.fourier.num_features * 2  # sin和cos各一半
        else:
            self.fourier = None
            current_dim = self.inpt_dim

        # 可选：添加ChebyshevGeLU激活函数
        if self.use_cheby_gelu:
            self.cheby_gelu = ChebyshevGeLU(degree=5)

        self.layers.append(Cheby2KANLayer(current_dim, self.middle_dim, self.degrees[0]))
        self.layers.append(nn.Dropout(dropout_prob))

        for i in range(self.num_layers):
            self.layers.append(Cheby2KANLayer(self.middle_dim, self.middle_dim, self.degrees[i+1]))
            if i < num_layers - 1:  # 最后一个中间层后不加Dropout
                self.layers.append(nn.Dropout(dropout_prob))

        self.layers.append(Cheby2KANLayer(self.middle_dim, self.output_dim, self.degrees[-1]))

    def forward(self, x):
        # 可选：应用Fourier特征
        if self.use_fourier:
            x = self.fourier(x)

        # 遍历模型的所有层：i=层索引，layer=当前层
        for i, layer in enumerate(self.layers):
            # 判断：当前层是不是【切比雪夫二阶KAN层】（自定义核心层）
            if isinstance(layer, Cheby2KANLayer):
                # 核心：满足3个条件 → 给KAN层加【残差连接】
                # 残差连接条件：中间层 + KAN层 + 输入输出维度一致
                if i != 0 and i != len(self.layers) - 1 and x.size(1) == layer.outdim:
                    # 残差计算：层输出 + 层输入（Residual Connection）
                    x = layer(x) + x
                # 不满足条件 → KAN层正常计算，不加残差
                else:
                    x = layer(x)
            # 非KAN层（如归一化、激活、线性层）：直接正常计算
            else:
                x = layer(x)

        # 可选：应用ChebyshevGeLU激活函数
        if self.use_cheby_gelu:
            x = self.cheby_gelu(x)

        # 不再使用sigmoid限制输出，允许模型学习更广泛的范围
        # 返回最终前向传播结果
        return x

    def compute_boundary_loss(self, X_norm, Y_pred_norm, X_mean, X_std, Y_mean, Y_std, spot_min, spot_max):
        """
        计算边界条件损失

        参数:
            X_norm: 无量纲化的输入特征 (batch_size, feature_dim)
            Y_pred_norm: 无量纲化的预测输出 (batch_size, 1)
            X_mean, X_std: 输入特征的均值和标准差 (1, feature_dim)
            Y_mean, Y_std: 输出的均值和标准差 (1, 1)
            spot_min, spot_max: spot价格的最小和最大值（标量）

        返回:
            边界损失
        """
        # 反归一化到物理空间
        # X_mean和X_std的形状是(1, feature_dim)，我们需要正确地广播
        spot = X_norm[:, 0:1] * X_std[:, 0:1] + X_mean[:, 0:1]  # 形状: (batch_size, 1)
        Y_pred = Y_pred_norm * Y_std + Y_mean  # 形状: (batch_size, 1)

        # 边界1: S → 0 时期权价格 → 0 ⇒ 隐含波动率应有限
        mask_low = spot < (spot_min * 1.1)
        loss_low = torch.mean(Y_pred[mask_low]**2) if mask_low.any() else torch.tensor(0.0, device=X_norm.device)

        # 边界2: S → ∞ 时期权价格 → S - K*exp(-rT) ⇒ 波动率应平滑
        mask_high = spot > (spot_max * 0.9)
        if mask_high.any() and mask_high.sum() > 1:
            loss_high = torch.mean(torch.abs(torch.diff(Y_pred[mask_high])))
        else:
            loss_high = torch.tensor(0.0, device=X_norm.device)

        return loss_low + loss_high


if __name__ == '__main__':
    lr_KAN, num_layers_KAN, degrees = 0.0059, 4, [2, 5, 3, 4, 5]
    input_size, output_size, middle_dim = 7, 9, 16
    input = torch.normal(0, 1, size=(64, 7))
    model = Cheby_KAN(input_size, output_size, middle_dim, degrees, num_layers_KAN, 0.1)
    result = model(input)
    print(result.shape)
