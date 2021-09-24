import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.batchnorm:
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(4)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data


    def forward(self, input_, prev_state2, timestep=0):

        if timestep == 0 and prev_state2 is not None:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)
        
        i = timestep
        if self.batchnorm:
            g1_t = torch.sigmoid(self.bn[0](self.u1_gate(prev_state2)))
            c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            
            next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            #next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            
            g2_t = torch.sigmoid(self.bn[2](self.u2_gate(next_state1)))
            c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            
            h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            #h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)
            
            new_state = (1 - g2_t)*prev_state2 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = F.sigmoid(self.bn[2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            new_state = self.n[timestep]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        return new_state, next_state1






class hConvGRUCell_2(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        # super().__init__()
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList([nn.BatchNorm2d(25, affine=True, track_running_stats=False) for i in range(4)])

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
#        self.w_gate_inh = nn.Parameter(self.w_gate_inh.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
#        self.w_gate_exc = nn.Parameter(self.w_gate_exc.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        # self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)
        # self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad, 1, 0)) * 0.5)
        #self.w_gate_inh.register_hook(lambda grad: print("inh"))
        #self.w_gate_exc.register_hook(lambda grad: print("exc"))
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        #init.constant_(self.scale, 0.1)
        #init.constant_(self.intercept, 0.) 
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        # self.outscale = nn.Parameter(torch.tensor([8.0]))
        # self.outintercpt = nn.Parameter(torch.tensor([-4.0]))
        # self.softpl = nn.Softplus()
        #self.softpl.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * 0.9, ))

    def forward(self, input_, prev_state2, timestep=0):

        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))  #self.bn[0]
        c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))  #self.bn[1]
        
        next_state1 = F.softplus(input_ - F.softplus(c1_t * (self.alpha * prev_state2 + self.mu)))
        
        g2_t = torch.sigmoid((self.u2_gate(next_state1)))  #self.bn[2]
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))  #self.bn[3]
        
        h2_t = F.softplus(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)
        
        #h2_t = h2_t / torch.norm(h2_t.view(h2_t.shape[0] * h2_t.shape[1], -1), dim=1).view(h2_t.shape[0], h2_t.shape[1], 1, 1)
        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        #prev_state2 = torch.sigmoid(prev_state2 * self.outscale + self.outintercpt)
        #prev_state2 = F.normalize(prev_state2, dim=0, p=2)
        #prev_state2 = prev_state2 / torch.norm(prev_state2.view(prev_state2.shape[0], -1), dim=1).view(prev_state2.shape[0], 1, 1, 1)
        prev_state2 = F.softplus(prev_state2)
        #prev_state2 = cielsoftplus.apply(prev_state2, 15.0)
        #prev_state2.requires_grad_()
        #_ = prev_state2.register_hook(lambda grad: grad * 0.9)
        #prev_state2 = self.softpl(prev_state2)
        return prev_state2, g2_t



class hGRUCellAct(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8):
        super().__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.batchnorm:
            #self.bn = nn.ModuleList([nn.GroupNorm(25, 25, eps=1e-03) for i in range(32)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03) for i in range(4)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
#        self.w_gate_inh = nn.Parameter(self.w_gate_inh.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
#        self.w_gate_exc = nn.Parameter(self.w_gate_exc.reshape(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
#        self.w_gate_inh.register_hook(lambda grad: print("inh"))
#        self.w_gate_exc.register_hook(lambda grad: print("exc"))
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data


    def forward(self, input_, prev_state2, timestep=0):

        if timestep == 0:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)

        #import pdb; pdb.set_trace()
        i = timestep
        if self.batchnorm:
            g1_t = torch.sigmoid(self.bn[0](self.u1_gate(prev_state2)))
            c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            
            next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            #next_state1 = F.relu(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            
            g2_t = torch.sigmoid(self.bn[2](self.u2_gate(next_state1)))
            c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            
            h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            #h2_t = F.relu(self.kappa*next_state1 + self.kappa*self.gamma*c2_t + self.w*next_state1*self.gamma*c2_t)
            
            new_state = (1 - g2_t)*prev_state2 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            g2_t = F.sigmoid(self.bn[2](self.u2_gate(next_state1)))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            new_state = self.n[timestep]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        dist = (new_state - prev_state2) ** 2
        dist = dist.view(dist.size(0), -1).mean(1)
        ponder = 1 - torch.exp(-(0.01 / dist))
        
        return new_state, next_state1, ponder