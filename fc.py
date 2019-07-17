from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class FCNet_org(nn.Module):
    """Simple class for non-linear fully connect network with gated tangent as in paper
    """
    def __init__(self, dims):
        super(FCNet_org, self).__init__()

        in_dim = dims[0]
        out_dim = dims[1]
        self.first_lin = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        self.tanh = nn.Tanh()
        self.second_lin = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        y_hat = self.tanh(self.first_lin(x))
        g = self.sigmoid(self.second_lin(x))

        y = y_hat * g

        return y


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
