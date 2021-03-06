import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet, BCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[0.2, 0.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name="h_mat",
            dim=None,
        )

    def forward(self, v, q, v_mask=True):
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True):

        #print('input ', v[0, :, :5], q[0,:,:5])

        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)

        if v_mask:
            v_abs_sum = v.abs().sum(2)
            mask = (v_abs_sum == 0).unsqueeze(1).unsqueeze(3)
            mask = mask.expand(logits.size())
            logits.masked_fill_(mask, -float("inf"))

        expanded_logits = logits.view(-1, self.glimpse, v_num * q_num)
        p = nn.functional.softmax(expanded_logits, 2)

        print(logits[0], p[0])



        return p.view(-1, self.glimpse, v_num, q_num), logits
