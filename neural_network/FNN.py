from torch import nn

class nn_impvol(nn.Module):
    def __init__(self, input_size, output_size, middle_size, num_layers):
        super(nn_impvol, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size, middle_size))
        layers.append(nn.Softplus())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(middle_size, middle_size))
            layers.append(nn.Softplus())

        layers.append(nn.Linear(middle_size, output_size))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        result = self.net(x)

        return result