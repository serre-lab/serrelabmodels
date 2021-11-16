import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class hGRUCell(nn.Module):
    """
    Generate a convolutional hGRU cell
    """

    def __init__(self, 
                input_size, 
                hidden_size, 
                kernel_size, 
                batchnorm = True, 
                timesteps = 8, 
                alpha = 0.1, 
                gamma = 1.0, 
                kappa = 0.5, 
                w = 0.5, 
                mu = 1, 
                bn_weight = 0.1
                ):

        # alpha     : Control linear inhibition by C_1
        # gamma     : Scales excitation by C_2
        # kappa     : Control linear contributions of horizontal connections to H
        # w         : Control quadratic contributions of horizontal connections to H
        # mu        : Control linear inhibition by C_1

        super().__init__()

        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        
        # Set to random values initially
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.batchnorm:
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size ) for i in range(4)])
            for bn in self.bn:
                init.constant_(bn.weight, bn_weight)
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        init.constant_(self.alpha, alpha)
        init.constant_(self.gamma, gamma)
        init.constant_(self.kappa, kappa)
        init.constant_(self.w, w)
        init.constant_(self.mu, mu)
        
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data


    def forward(self, input_, previous_state, timestep = 0):
        # H_2[t-1] -> Previous state [h2_tminus1]
        # H_1[t] -> Next State [h1_t]
        # G_1 -> Reset gate in hidden states
        # G_2 -> Update gate in hidden states

        h2_tminus1 = previous_state
        if timestep == 0 and h2_tminus1 is not None:
            h2_tminus1 = torch.empty_like(input_)
            init.xavier_normal_(h2_tminus1)
        
        i = timestep
        if self.batchnorm:
            g1_t = torch.sigmoid(self.bn[0](self.u1_gate(h2_tminus1)))
            c1_t = self.bn[1](F.conv2d(h2_tminus1 * g1_t, self.w_gate_inh, padding=self.padding))
            h1_t = F.relu(input_ - F.relu(c1_t*(self.alpha*h2_tminus1 + self.mu)))
            g2_t = torch.sigmoid(self.bn[2](self.u2_gate(h1_t)))
            c2_t = self.bn[3](F.conv2d(h1_t, self.w_gate_exc, padding=self.padding))
            h2_t = F.relu(self.kappa*h1_t + self.gamma*c2_t + self.w*h1_t*c2_t)
            h_n = (1 - g2_t)*h2_tminus1 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(h2_tminus1))
            c1_t = F.conv2d(h2_tminus1 * g1_t, self.w_gate_inh, padding=self.padding)
            h1_t = F.tanh(input_ - c1_t*(self.alpha*h2_tminus1 + self.mu))
            g2_t = F.sigmoid(self.bn[2](self.u2_gate(h1_t)))
            c2_t = F.conv2d(h1_t, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(h1_t + self.gamma*c2_t) + (self.w*(h1_t*(self.gamma*c2_t))))
            h_n = self.n[timestep]*((1 - g2_t)*h2_tminus1 + g2_t*h2_t)

        return h_n, h1_t