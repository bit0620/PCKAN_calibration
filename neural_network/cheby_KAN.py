from torch import nn
import torch

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

        # initialize \tilde{W}^{(l)}_{q,p,n}
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

        # s_{b,q,p} = sum_{n=0}^{D_l} \tilde{W}_{q,p,n} * U_n(x_p)
        s = torch.einsum('bpn, qpn->bqp', cheby, self.tilde_W)
        y = torch.einsum('bqp, qp->bq', s, self.W)

        return y

class Cheby_KAN(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, degrees, num_layers, dropout_prob):
        super(Cheby_KAN, self).__init__()
        self.inpt_dim = input_dim
        self.output_dim = output_dim
        self.middle_dim = middle_dim
        self.degrees = degrees
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.layers = nn.ModuleList()

        self.layers.append(Cheby2KANLayer(self.inpt_dim, self.middle_dim, self.degrees[0]))
        self.layers.append(nn.Dropout(dropout_prob))

        for i in range(self.num_layers):
            self.layers.append(Cheby2KANLayer(self.middle_dim, self.middle_dim, self.degrees[i+1]))
            if i < num_layers - 1:  # 最后一个中间层后不加Dropout
                self.layers.append(nn.Dropout(dropout_prob))

        self.layers.append(Cheby2KANLayer(self.middle_dim, self.output_dim, self.degrees[-1]))

    def forward(self, x):
        # 2. 遍历模型的所有层：i=层索引，layer=当前层
        for i, layer in enumerate(self.layers):
            # 3. 判断：当前层是不是【切比雪夫二阶KAN层】（自定义核心层）
            if isinstance(layer, Cheby2KANLayer):
                # 4. 核心：满足3个条件 → 给KAN层加【残差连接】
                # 残差连接条件：中间层 + KAN层 + 输入输出维度一致
                if i != 0 and i != len(self.layers) - 1 and x.size(1) == layer.outdim:
                    # 残差计算：层输出 + 层输入（Residual Connection）
                    x = layer(x) + x
                # 不满足条件 → KAN层正常计算，不加残差
                else:
                    x = layer(x)
            # 5. 非KAN层（如归一化、激活、线性层）：直接正常计算
            else:
                x = layer(x)
        
        # 6. 返回最终前向传播结果
        return x





if __name__ == '__main__':
    lr_KAN, num_layers_KAN, degrees = 0.0059, 4, [2, 5, 3, 4, 5]
    input_size, output_size, middle_dim = 7, 9, 16
    input = torch.normal(0, 1, size=(64, 7))
    model = Cheby_KAN(input_size, output_size, middle_dim, degrees, num_layers_KAN, 0.1)
    result = model(input)
    print(result.shape)