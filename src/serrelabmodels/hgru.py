import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
import torchvision.models as models

from serrelabmodels.layers.hgru_cell import hGRUCell

from serrelabmodels.utils.pt_utils import Conv2dSamePadding
from serrelabmodels.utils import pt_utils

from serrelabmodels.ops import model_tools

# vgg_example={
#     'name': 'vgg_hgru',
#     'import_prepath': 'serrelabmodels.models.vgg_16',
#     'import_class': 'VGG_16',
#     'args': {
#         'weight_path': '/media/data_cifs/model_weights/vgg16.pth',
#         'load_weights': True,
#         'freeze_layers': False,
#         'n_layers': 17
#     }
# }

# hgru_params_example = [['conv2_2', 9],
#                     ['conv3_3', 9],
#                     ['conv4_3', 9],
#                     ['conv5_3', 9],]

# class BasehGRU(nn.Module):
#     def __init__(self, 
#                 base_ff=vgg_example,
#                 hgru_params=hgru_params_example,
#                 timesteps=6,
#                 ):

#         super().__init__()

#         self.hgru_params = hgru_params
#         self.timesteps = timesteps
#         self.base_ff = model_tools.get_model(base_ff)
#         # self.base_ff = models.vgg16(pretrained=True)
#         self.build_fb_layers()
    
#     def build_fb_layers(self):
#         self.h_units = []
#         self.ds_blocks = []
        
#         prev_pos = 0
#         base_layers = self.base_ff.layers
#         base_filters = self.base_ff.filters
        
#         # downsampling and horizontal units
#         for pos,k_size in self.hgru_params:

#             layer_pos = base_layers.index(pos)+1
#             feats = base_filters[base_layers.index(pos)]

#             h_unit = hConvGRUCell(input_size = feats, 
#                             hidden_size = feats, 
#                             kernel_size = k_size)
#             h_unit.train()
#             self.h_units.append(h_unit)
#             self.ds_blocks.append(self.create_ds_block(base_layers[prev_pos:layer_pos]))
#             prev_pos = layer_pos

#         # last downsampling output block
#         if layer_pos+1 < len(base_layers):
#             self.output_block = self.create_ds_block(base_layers[prev_pos:len(base_layers)])
        
#         self.input_block = self.ds_blocks.pop(0)
#         self.output_feats = feats
#         self.h_units = nn.ModuleList(self.h_units)
#         self.ds_blocks = nn.ModuleList(self.ds_blocks)

#     def create_ds_block(self,base_layers):
#         # ds block: depends on base_ff non_linearity and bn
#         module_list = []
#         for l in base_layers:
#             module_list.append(getattr(self.base_ff,l))
#             if 'conv' in l:
#                 module_list.append(nn.ReLU())
        
#         return nn.Sequential(*module_list) if len(module_list)>1 else module_list[0]

#     def forward(self, x, return_hidden=False):
        
#         h_hidden = [None] * len(self.h_units)
#         conv_input = self.input_block(x)
#         last_hidden = []
        
#         selector = torch.ones(x.shape[0]).byte().to(x.device)
#         accum_h = torch.zeros(x.shape[0]).to(x.device)
#         step_count = torch.zeros(x.shape[0]).to(x.device)
#         step_ponder_cost = torch.zeros(x.shape[0]).to(x.device)

#         for i in range(self.timesteps):
#             x = conv_input

#             idxs = selector.nonzero().squeeze(1)
#             step_ponder_cost[idxs] =  -accum_h[idxs]
#             accum_dist = 0

#             for l in range(len(self.h_units)):
#                 if i==0:
#                     hidden = torch.empty_like(x)
#                     init.xavier_normal_(hidden)
#                     h_hidden[l] = hidden
#                 hidden, _ = self.h_units[l](x, h_hidden[l], timestep=i)
#                 dist = (hidden - h_hidden[l]) ** 2
#                 accum_dist += dist.view(dist.size(0), -1).mean(1)

#                 h_hidden[l] = hidden
#                 if l<len(self.ds_blocks):
#                     x = self.ds_blocks[l](hidden)
#                 else:
#                     x = hidden
            
#             accum_dist /= len(self.h_units)
#             ponder = 1 - torch.exp(-(0.01 / accum_dist))
                        
#             accum_h[idxs] = accum_h[idxs] + ponder
#             step_count[idxs] += 1
#             selector = (accum_h < 1 - self.eps).data
#             if not selector.any():
#                 break

#         if return_hidden:
#             last_hidden = torch.stack(last_hidden,dim=1)
#             return x, last_hidden, step_ponder_cost, step_count
#         else:
#             return x, step_ponder_cost, step_count

class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell

    This class could be replaced simply by:

    import serrelabmodels.layers.hgru_cell
    hgru_cell =  serrelabmodels.layers.hgru_cell.hRGUCell(5, 5, 3)
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8, grad_method='bptt'):
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

        self.bn = nn.ModuleList([nn.BatchNorm2d(self.hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(4)])

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
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
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        
        self.softpl = nn.Softplus()
        self.softpl.register_backward_hook(lambda module, grad_i, grad_o: print(len(grad_i)))

    def forward(self, input_, prev_state2, timestep=0):
        activ = F.softplus  # relu
        #activ = torch.sigmoid
        #activ = torch.tanh
        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))
        c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
        
        next_state1 = activ(input_ - activ(c1_t * (self.alpha * prev_state2 + self.mu)))
        
        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
        
        h2_t = activ(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)
        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        #prev_state2 = F.softplus(prev_state2)

        return prev_state2, g2_t


class hConvGRU(nn.Module):
    '''
    Create the networl using a hGRU Cell

    This could also be done by simply using the serrelabmodels module:
    
    import serrelabmodels.hgru
    hgru_model = serrelabmodels.hgru.BasehGRU()
    '''

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(hConvGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hidden_size = 48
        
        # Change number of channels here
        self.conv0 = nn.Conv2d(1, self.hidden_size, kernel_size=5, padding=5 // 2)
        
        init.xavier_normal_(self.conv0.weight)

        self.unit1 = hConvGRUCell(self.hidden_size, self.hidden_size, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(self.hidden_size, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv2d(self.hidden_size, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))
        self.bn2 = nn.BatchNorm2d(2, eps=1e-03, track_running_stats=False)
        self.avgpool1 = nn.AvgPool2d(10, stride=5)
        self.maxpool = nn.MaxPool2d(150, stride=1)
        self.dense1 = nn.Linear(2, 2)
        init.xavier_normal_(self.dense1.weight)

    def forward(self, x, testmode=False):
        x = self.conv0(x)
        # x = self.bn0(x)
        x = F.softplus(x)  # torch.pow(x, 2)
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
        #internal_state = torch.tanh(internal_state)
        out = self.bn(internal_state)
        out = F.leaky_relu(self.conv6(out))
        # out = self.maxpool(out)
        out = F.avg_pool2d(out, kernel_size=out.size()[2:])
        # out = self.bn2(out)
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