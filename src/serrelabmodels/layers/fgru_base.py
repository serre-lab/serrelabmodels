import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

from serrelabmodels.utils.pt_utils import conv2d_same_padding, Conv2dSamePadding, tied_conv2d_same_padding, space_tied_conv2d_same_padding
from serrelabmodels.utils import pt_utils

import numpy as np


class fGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, 
                input_size, 
                hidden_size, 
                kernel_size,
                hidden_init='identity',
                attention='gala',
                attention_layers=2,
                saliency_filter_size=5,
                norm_attention=False,
                tied_kernels='depth',
                normalization_fgru='InstanceNorm2d',
                normalization_fgru_params={'affine': True},
                normalization_gate='InstanceNorm2d',
                normalization_gate_params={'affine': True},
                ff_non_linearity='ReLU',
                force_alpha_divisive=True,
                force_non_negativity=True,
                multiplicative_excitation=True,
                gate_bias_init='chronos', #'ones'
                timesteps=8
                ):
        super().__init__()
        
        self.padding = 'same' # kernel_size // 2

        self.kernel_size = kernel_size
        self.tied_kernels = tied_kernels

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_init = hidden_init

        self.ff_nl = pt_utils.get_nl(ff_non_linearity)

        self.normalization_fgru = normalization_fgru
        self.normalization_gate = normalization_gate
        self.normalization_fgru_params = normalization_fgru_params if normalization_fgru_params is not None else {}
        self.normalization_gate_params = normalization_gate_params if normalization_gate_params is not None else {}

        self.normalization_gate = normalization_gate

        if self.normalization_fgru:
            normalization_fgru = pt_utils.get_norm(normalization_fgru)
        if self.normalization_gate:
            normalization_gate = pt_utils.get_norm(normalization_gate)

        self.force_alpha_divisive = force_alpha_divisive
        self.force_non_negativity = force_non_negativity
        self.multiplicative_excitation = multiplicative_excitation

        # Adding attention
        if attention is not None and attention_layers>0:
            if attention == 'se':
                self.attention = SE_Attention(  hidden_size, hidden_size, 1,
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate if norm_attention else None, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            elif attention == 'gala':
                self.attention = GALA_Attention(hidden_size, hidden_size, saliency_filter_size, 
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate if norm_attention else None, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            else:
                raise 'attention type unknown'
        else:
            self.conv_g1_w = nn.Parameter(torch.empty(hidden_size , hidden_size, 1, 1))
            init.orthogonal_(self.conv_g1_w)

        self.conv_g1_b = nn.Parameter(torch.empty(hidden_size,1,1))

        if self.normalization_gate:
            self.bn_g1 = normalization_gate(hidden_size, **self.normalization_gate_params)
            init.constant_(self.bn_g1.weight, 0.1)
            init.constant_(self.bn_g1.bias, 0)

        if self.normalization_fgru:
            self.bn_c1 = normalization_fgru(hidden_size, **self.normalization_fgru_params)
            init.constant_(self.bn_c1.weight, 0.1)
            init.constant_(self.bn_c1.bias, 0)

        if tied_kernels=='channel':
            self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , hidden_size , 1, 1))
        elif tied_kernels=='spatial':
            self.conv_c1_w = nn.Parameter(torch.empty(1 , 1 , kernel_size, kernel_size))
        elif tied_kernels=='depth':
            self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , 1 , kernel_size, kernel_size))
        else:
            self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.conv_g2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , 1, 1))
        init.orthogonal_(self.conv_g2_w)

        self.conv_g2_b = nn.Parameter(torch.empty(hidden_size,1,1))
        
        if self.normalization_gate:
            self.bn_g2 = normalization_gate(hidden_size, **self.normalization_gate_params) 
            init.constant_(self.bn_g2.weight, 0.1)
            init.constant_(self.bn_g2.bias, 0)

        if self.normalization_fgru:
            self.bn_c2 = normalization_fgru(hidden_size, **self.normalization_fgru_params) 
            init.constant_(self.bn_c2.weight, 0.1)
            init.constant_(self.bn_c2.bias, 0)
        
        if tied_kernels=='channel':
            self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , 1, 1))
        elif tied_kernels=='spatial':
            self.conv_c2_w = nn.Parameter(torch.empty(1 , 1 , kernel_size, kernel_size))
        elif tied_kernels=='depth':
            self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , 1 , kernel_size, kernel_size))
        else:
            self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        init.orthogonal_(self.conv_c1_w)
        init.orthogonal_(self.conv_c2_w)

        if tied_kernels!='depth':
            self.conv_c1_w.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
            self.conv_c2_w.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)

        if gate_bias_init == 'chronos':
            init_chronos = -np.log(np.random.uniform(1.0, max(float(timesteps- 1), 1.0), [hidden_size,1,1]))
            self.conv_g1_b.data = torch.FloatTensor(init_chronos)
            self.conv_g2_b.data = torch.FloatTensor(- init_chronos)

        else:
            init.constant_(self.conv_g1_b, 1)
            init.constant_(self.conv_g2_b, 1)

        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu = nn.Parameter(torch.empty((hidden_size,1,1)))

        if self.multiplicative_excitation:
            self.omega = nn.Parameter(torch.empty((hidden_size,1,1)))
            self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        
        if self.force_alpha_divisive:
            init.kaiming_normal_(self.alpha) # variance scaling
        else:
            init.constant_(self.alpha, 0.1)

        init.constant_(self.mu, 0)
        if self.multiplicative_excitation:
            init.constant_(self.omega, 0.5)
            init.constant_(self.kappa, 0.5)

    def forward(self, input_, prev_state2, timestep=0):
        if timestep == 0 and prev_state2 is None:
            if self.hidden_init =='identity':
                prev_state2 = input_
            elif self.hidden_init =='zero':
                prev_state2 = torch.empty_like(input_)
                init.zeros_(prev_state2)
            else:
                prev_state2 = torch.empty_like(input_)
                init.xavier_normal_(prev_state2)

        i = timestep
        
        h2_int = prev_state2
        
        ############## Circuit Input
        
        if hasattr(self,'attention'):
            g1 = self.attention(h2_int)
        else:
            g1 = F.conv2d(h2_int, self.conv_g1_w)
        
        # g1_intermediate
        g1_n = self.bn_g1(g1 + self.conv_g1_b) if self.normalization_gate is not None else g1
        
        h2_int_2 = h2_int * torch.sigmoid(g1_n)

        # c1 -> conv2d symmetric_weights, dilations
        if self.tied_kernels=='channel':
            c1 = tied_conv2d_same_padding(h2_int_2,self.conv_c1_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c1 = space_tied_conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')

        c1_n = self.bn_c1(c1) if self.normalization_fgru is not None else c1

        ############## input integration
        # alpha, mu
        if self.force_alpha_divisive:
            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = self.alpha
        inh = (alpha * prev_state2 + self.mu) * c1_n

        if self.force_non_negativity:
            h1 = self.ff_nl(self.ff_nl(input_) - self.ff_nl(inh))
        else:
            h1 = self.ff_nl(input_ - self.ff_nl(inh))
        # h1 = F.relu(input_ - inh)

        ############## circuit output
        g2 = conv2d_same_padding(h1, self.conv_g2_w)

        g2_n = self.bn_g2(g2 + self.conv_g2_b) if self.normalization_gate is not None else g2

        g2_s = torch.sigmoid(g2_n)

        if self.tied_kernels=='channel':
            c2 = tied_conv2d_same_padding(h1,self.conv_c2_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c2 = space_tied_conv2d_same_padding(h1,self.conv_c2_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c2 = conv2d_same_padding(h1,self.conv_c2_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c2 = conv2d_same_padding(h1, self.conv_c2_w, padding_mode='reflect')

        c2_n = self.bn_c2(c2) if self.normalization_fgru is not None else c2
        
        ############## output integration
        if self.multiplicative_excitation:
            h2_hat = self.ff_nl( self.kappa*(h1 + c2_n) + self.omega*(h1 * c2_n) )
        else:
            h2_hat = self.ff_nl(h1 + c2_n)

        h2 = g2_s * prev_state2 + (1 - g2_s) * h2_hat
        
        return h2, h1


class fGRUCell2(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, 
                input_size, 
                hidden_size, 
                kernel_size,
                hidden_init='identity',
                attention='gala', # 'se', None
                attention_layers=2,
                saliency_filter_size=5,
                tied_kernels=None,
                norm_attention=False,
                normalization_fgru='InstanceNorm2d',
                normalization_fgru_params={'affine': True},
                normalization_gate='InstanceNorm2d',
                normalization_gate_params={'affine': True},
                ff_non_linearity='ReLU',
                force_alpha_divisive=True,
                force_non_negativity=True,
                multiplicative_excitation=True,
                gate_bias_init='chronos', #'ones'
                timesteps=8
                ):
        super().__init__()
        
        self.padding = 'same' # kernel_size // 2

        self.kernel_size = kernel_size
        self.tied_kernels = tied_kernels

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_init = hidden_init

        self.ff_nl = pt_utils.get_nl(ff_non_linearity)

        self.normalization_fgru = normalization_fgru
        self.normalization_gate = normalization_gate
        self.normalization_fgru_params = normalization_fgru_params if normalization_fgru_params is not None else {}
        self.normalization_gate_params = normalization_gate_params if normalization_gate_params is not None else {}

        self.normalization_gate = normalization_gate

        if self.normalization_fgru:
            normalization_fgru = pt_utils.get_norm(normalization_fgru)
        if self.normalization_gate:
            normalization_gate = pt_utils.get_norm(normalization_gate)

        self.force_alpha_divisive = force_alpha_divisive
        self.force_non_negativity = force_non_negativity
        self.multiplicative_excitation = multiplicative_excitation

        # Adding attention
        if attention is not None and attention_layers>0:
            if attention == 'se':
                self.attention = SE_Attention(  hidden_size, hidden_size, 1,
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate if norm_attention else None, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            elif attention == 'gala':
                self.attention = GALA_Attention(hidden_size, hidden_size, saliency_filter_size, 
                                                layers=attention_layers, 
                                                normalization=self.normalization_gate if norm_attention else None, # 'BatchNorm2D'
                                                normalization_params=self.normalization_gate_params,
                                                non_linearity=ff_non_linearity,
                                                norm_pre_nl=False)
            else:
                raise 'attention type unknown'
        else:
            self.conv_g1_w = nn.Parameter(torch.empty(hidden_size , hidden_size, 1, 1))
            init.orthogonal_(self.conv_g1_w)

        self.conv_g1_b = nn.Parameter(torch.empty(hidden_size,1,1))

        if self.normalization_gate:
            self.bn_g1 = normalization_gate(hidden_size, track_running_stats=False, **self.normalization_gate_params)
            init.constant_(self.bn_g1.weight, 0.1)
            init.constant_(self.bn_g1.bias, 0)

        if self.normalization_fgru:
            self.bn_c1 = normalization_fgru(hidden_size, track_running_stats=False, **self.normalization_fgru_params)
            init.constant_(self.bn_c1.weight, 0.1)
            init.constant_(self.bn_c1.bias, 0)

        if tied_kernels=='depth':
            self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , 1 , kernel_size, kernel_size))
        else:
            self.conv_c1_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        self.conv_g2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , 1, 1))
        init.orthogonal_(self.conv_g2_w)

        self.conv_g2_b = nn.Parameter(torch.empty(hidden_size,1,1))
        
        if self.normalization_gate:
            self.bn_g2 = normalization_gate(hidden_size, track_running_stats=False, **self.normalization_gate_params) 
            init.constant_(self.bn_g2.weight, 0.1)
            init.constant_(self.bn_g2.bias, 0)

        if self.normalization_fgru:
            self.bn_c2 = normalization_fgru(hidden_size, track_running_stats=False, **self.normalization_fgru_params) 
            init.constant_(self.bn_c2.weight, 0.1)
            init.constant_(self.bn_c2.bias, 0)
        
        if tied_kernels=='depth':
            self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , 1 , kernel_size, kernel_size))
        else:
            self.conv_c2_w = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        
        init.orthogonal_(self.conv_c1_w)
        init.orthogonal_(self.conv_c2_w)

        if gate_bias_init == 'chronos':
            init_chronos = np.log(np.random.uniform(1.0, max(float(timesteps- 1), 1.0), [hidden_size,1,1]))

            self.conv_g1_b.data = torch.FloatTensor(init_chronos)
            self.conv_g2_b.data = torch.FloatTensor(- init_chronos)
        else:
            init.constant_(self.conv_g1_b, 1)
            init.constant_(self.conv_g2_b, 1)

        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu = nn.Parameter(torch.empty((hidden_size,1,1)))
        
        self.omega = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.mu, 1.0)

        init.constant_(self.omega, 0.5)
        init.constant_(self.kappa, 0.5)

    def forward(self, input_, prev_state2, timestep=0):        
        if timestep == 0 and prev_state2 is None:
            
            if self.hidden_init =='identity':
                prev_state2 = input_
            elif self.hidden_init =='zero':
                prev_state2 = torch.empty_like(input_)
                init.zeros_(prev_state2)
            else:
                prev_state2 = torch.empty_like(input_)
                init.xavier_normal_(prev_state2)

        i = timestep
        
        h2_int = prev_state2
        
        ############## circuit input
        
        if hasattr(self,'attention'):
            g1 = self.attention(h2_int)
        else:
            g1 = F.conv2d(h2_int, self.conv_g1_w)
        
        # g1_intermediate
        g1_n = self.bn_g1(g1 + self.conv_g1_b) if self.normalization_gate is not None else g1
        

        # this changed from conv -> bn(in) -> bias -> sigmoid
        #                to conv -> sigmoid -> bn (beta is chronos)
        h2_int_2 = h2_int * torch.sigmoid(g1_n)

        # c1 -> conv2d symmetric_weights, dilations
        if self.tied_kernels=='channel':
            c1 = tied_conv2d_same_padding(h2_int_2,self.conv_c1_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c1 = space_tied_conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')

        c1_n = self.bn_c1(c1) if self.normalization_fgru is not None else c1

        h1 = self.ff_nl(input_ - self.ff_nl((self.alpha * prev_state2 + self.mu) * c1_n))
        
        ############## circuit output

        g2 = conv2d_same_padding(h1, self.conv_g2_w)

        g2_n = self.bn_g2(g2 + self.conv_g2_b) if self.normalization_gate is not None else g2

        g2_s = torch.sigmoid(g2_n)

        if self.tied_kernels=='channel':
            c2 = tied_conv2d_same_padding(h1,self.conv_c2_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c2 = space_tied_conv2d_same_padding(h1,self.conv_c2_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c2 = conv2d_same_padding(h1,self.conv_c2_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c2 = conv2d_same_padding(h1, self.conv_c2_w, padding_mode='reflect')

        c2_n = self.bn_c2(c2) if self.normalization_fgru is not None else c2
        
        ############## output integration
        h2_hat = self.ff_nl( self.kappa*(h1 + c2_n) + self.omega*(h1 * c2_n) )

        h2 = (1 - g2_s) * prev_state2 + g2_s * h2_hat
        
        return h2, h1

class fGRUCell2_td(fGRUCell2):
    def forward(self, input_, prev_state2, timestep=0):
        if timestep == 0 and prev_state2 is None:
            if self.hidden_init =='identity':
                prev_state2 = input_
            elif self.hidden_init =='zero':
                prev_state2 = torch.empty_like(input_)
                init.zeros_(prev_state2)
            else:
                prev_state2 = torch.empty_like(input_)
                init.xavier_normal_(prev_state2)

        i = timestep
        
        h2_int = prev_state2
        
        ############## circuit input
        if hasattr(self,'attention'):
            g1 = self.attention(h2_int)
        else:
            g1 = F.conv2d(h2_int, self.conv_g1_w)
        
        # g1_intermediate
        g1_n = self.bn_g1(g1 + self.conv_g1_b) if self.normalization_gate is not None else g1
        

        # this changed from conv -> bn(in) -> bias -> sigmoid
        #                to conv -> sigmoid -> bn (beta is chronos)
        h2_int_2 = h2_int * torch.sigmoid(g1_n)

        # c1 -> conv2d symmetric_weights, dilations
        if self.tied_kernels=='channel':
            c1 = tied_conv2d_same_padding(h2_int_2,self.conv_c1_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c1 = space_tied_conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c1 = conv2d_same_padding(h2_int_2,self.conv_c1_w, padding_mode='reflect')

        c1_n = self.bn_c1(c1) if self.normalization_fgru is not None else c1

        h1 = self.ff_nl(input_ - self.ff_nl((self.alpha * prev_state2 + self.mu) * c1_n))
        
        ############## circuit output

        g2 = conv2d_same_padding(h1, self.conv_g2_w)

        g2_n = self.bn_g2(g2 + self.conv_g2_b) if self.normalization_gate is not None else g2

        g2_s = torch.sigmoid(g2_n)

        if self.tied_kernels=='channel':
            c2 = tied_conv2d_same_padding(h1,self.conv_c2_w,pool_size=(self.kernel_size,self.kernel_size), padding_mode='reflect')
        elif self.tied_kernels=='spatial':
            c2 = space_tied_conv2d_same_padding(h1,self.conv_c2_w, padding_mode='reflect')
        elif self.tied_kernels=='depth':
            c2 = conv2d_same_padding(h1,self.conv_c2_w, groups=self.hidden_size, padding_mode='reflect')
        else:
            c2 = conv2d_same_padding(h1, self.conv_c2_w, padding_mode='reflect')

        c2_n = self.bn_c2(c2) if self.normalization_fgru is not None else c2
        
        ############## output integration
        h2_hat = self.ff_nl( self.kappa*(h1 + c2_n) + self.omega*(h1 * c2_n) )
        h2 = (1 - g2_s) * input_ + g2_s * h2_hat
        
        return h2, h1



class SE_Attention(nn.Module):
    """ if layers > 1  downsample -> upsample """
    
    def __init__(self, 
                input_size, 
                output_size, 
                filter_size, 
                layers, 
                normalization=True, 
                normalization_type='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):
        super().__init__()
        
        if normalization_params is None:
            normalization_params={}
        
        curr_feat = input_size
        self.module_list = []
            
        for i in range(layers):
            if i == layers-1:
                next_feat = output_size
            elif i < layers//2:
                next_feat = curr_feat // 2
            else:
                next_feat = curr_feat * 2
            
            conv = Conv2dSamePadding(curr_feat, next_feat, filter_size)
            init.orthogonal_(conv.weight)
            init.constant_(conv.bias, 0)
            self.module_list.append(conv)
            
            if non_linearity is not None:
                nl = pt_utils.get_nl(non_linearity)
                
            if normalization is not None:
                norm = pt_utils.get_norm(normalization)(next_feat, **normalization_params)
                init.constant_(norm.weight, 0.1)
                init.constant_(norm.bias, 0)
            
            if norm_pre_nl :
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)
            
            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)
    
    def forward(self, input_):
        return self.attention(input_)

class SA_Attention(nn.Module):
    """ if layers > 1  downsample til 1 """
    
    def __init__(self, 
                input_size, 
                output_size, 
                filter_size, 
                layers, 
                normalization='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):
        super().__init__()
        
        if normalization_params is None:
            normalization_params={}
        
        curr_feat = input_size
        self.module_list = []
        for i in range(layers):
            if i == layers-1:
                next_feat = output_size
            else:
                next_feat = curr_feat // 2
            
            conv = Conv2dSamePadding(curr_feat, next_feat, filter_size)
            init.orthogonal_(conv.weight)
            init.constant_(conv.bias, 0)
            self.module_list.append(conv)
            
            if non_linearity is not None:
                nl = pt_utils.get_nl(non_linearity)
                
            if normalization is not None:
                norm = pt_utils.get_norm(normalization)(next_feat, **normalization_params)
                init.constant_(norm.weight, 0.1)
                init.constant_(norm.bias, 0)
            
            if norm_pre_nl :
                if normalization is not None:
                    self.module_list.append(norm)
                if non_linearity is not None:
                    self.module_list.append(nl)
            else:
                if non_linearity is not None:
                    self.module_list.append(nl)
                if normalization is not None:
                    self.module_list.append(norm)

            curr_feat = next_feat
        self.attention = nn.Sequential(*self.module_list)
    
    def forward(self, input_):
        return self.attention(input_)

class GALA_Attention(nn.Module):
    """ if layers > 1  downsample til spatial saliency is 1 """
    def __init__(self, 
                input_size, 
                output_size, 
                saliency_filter_size, 
                layers, 
                normalization='InstanceNorm2d', # 'BatchNorm2D'
                normalization_params=None,
                non_linearity='ReLU',
                norm_pre_nl=False):

        super().__init__()

        self.se = SE_Attention(input_size, output_size, 1,                    layers, 
                                normalization=normalization, # 'BatchNorm2D'
                                normalization_params=normalization_params,
                                non_linearity=non_linearity,
                                norm_pre_nl=norm_pre_nl)
        self.sa = SA_Attention(input_size, 1,           saliency_filter_size, layers,
                                normalization=normalization, # 'BatchNorm2D'
                                normalization_params=normalization_params,
                                non_linearity=non_linearity,
                                norm_pre_nl=norm_pre_nl)
    
    def forward(self, input_):
        return self.sa(input_) * self.se(input_)