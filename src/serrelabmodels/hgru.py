import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
import torchvision.models as models

from serrelabmodels.layers.hgru_cell import hGRUCell

class hConvGRU(nn.Module):
    '''
    Create the networl using a hGRU Cell
    '''

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(hConvGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hidden_size = 48
        
        self.conv0 = nn.Conv2d(1, self.hidden_size, kernel_size=5, padding=5 // 2)
        
        init.xavier_normal_(self.conv0.weight)

        self.unit1 = hGRUCell(self.hidden_size, self.hidden_size, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(self.hidden_size, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv2d(self.hidden_size, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-03, track_running_stats=False)
        self.avgpool1 = nn.AvgPool2d(10, stride=5)
        self.maxpool = nn.MaxPool2d(150, stride=1)
        self.dense1 = nn.Linear(2, 2)
        init.xavier_normal_(self.dense1.weight)

    def forward(self, x, testmode=False):
        x = self.conv0(x)
        x = F.softplus(x)
        internal_state = torch.zeros_like(x, requires_grad=False)
        states = []
        if self.grad_method == 'rbp':
            with torch.no_grad():
                for i in range(self.timesteps - 1):
                    if testmode: states.append(internal_state)
                    internal_state, g2t = self.unit1(x, internal_state, timestep=i)
            if testmode: states.append(internal_state)
            state_2nd_last = internal_state.detach().requires_grad_()
            i += 1
            last_state, g2t = self.unit1(x, state_2nd_last, timestep=i)
            internal_state = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
            if testmode: states.append(internal_state)

        elif self.grad_method == 'bptt':
            for i in range(self.timesteps):
                internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                if i == self.timesteps - 2:
                    state_2nd_last = internal_state
                elif i == self.timesteps - 1:
                    last_state = internal_state
        out = self.bn(internal_state)
        out = F.leaky_relu(self.conv6(out))
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.dense1(out)

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('NaN encountered in penalty')
        if testmode: return output, states, loss
        return out