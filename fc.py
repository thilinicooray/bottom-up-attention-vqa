from __future__ import print_function
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

import torch

from collections import OrderedDict
from torch import nn

import math


class MLP(nn.Module):
    """A simple MLP.
    """

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, dropout_p=0.1):
        """Constructor for MLP.

        Args:
            input_size: The number of input dimensions.
            hidden_size: The number of hidden dimensions for each layer.
            num_classes: The size of the output.
            num_layers: The number of hidden layers.
            dropout_p: Dropout probability.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        layers = []
        for i in range(num_layers):
            idim = hidden_size
            odim = hidden_size
            if i == 0:
                idim = input_size
            if i == num_layers-1:
                odim = self.num_classes
            fc = nn.Linear(idim, odim)
            fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
            fc.bias.data.fill_(0)
            layers.append(('fc'+str(i), fc))
            if i != num_layers-1:
                layers.append(('relu'+str(i), nn.ReLU()))
                layers.append(('dropout'+str(i), nn.Dropout(p=dropout_p)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def params_to_train(self):
        return self.layers.parameters()

    def forward(self, x):
        """Propagate through all the hidden layers.

        Args:
            x: Input of self.input_size dimensions.
        """
        out = self.layers(x)
        return out

class FCNet_relu(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet_relu, self).__init__()

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

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network with gated tangent as in paper
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

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

class BCNet(nn.Module):
    """
    Simple class for non-linear bilinear connect network
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act="ReLU", dropout=[0.2, 0.5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])

        if k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out is None:
            pass

        elif h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * self.k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if self.h_out is None:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = d_.transpose(1, 2).transpose(2, 3)
            return logits

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat
            logits = torch.matmul(h_, q_.unsqueeze(1).transpose(2, 3))
            logits = logits + self.h_bias
            return logits

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            return logits.transpose(2, 3).transpose(1, 2)

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)
        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)
        logits = logits.squeeze(3).squeeze(2)

        if self.k > 1:
            logits = logits.unsqueeze(1)
            logits = self.p_net(logits).squeeze(1) * self.k

        return logits


if __name__ == '__main__':
    fc1 = FCNet([10, 20, 10])
    print(fc1)

    print('============')
    fc2 = FCNet([10, 20])
    print(fc2)
