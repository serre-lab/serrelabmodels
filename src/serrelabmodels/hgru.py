import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
import torchvision.models as models

from serrelabmodels.layers.hgru_base import hConvGRUCell

from serrelabmodels.utils.pt_utils import Conv2dSamePadding
from serrelabmodels.utils import pt_utils

from serrelabmodels.ops import model_tools

vgg_example={
    'name': 'vgg_hgru',
    'import_prepath': 'serrelabmodels.models.vgg_16',
    'import_class': 'VGG_16',
    'args': {
        'weight_path': '/media/data_cifs/model_weights/vgg16.pth',
        'load_weights': True,
        'freeze_layers': False,
        'n_layers': 17
    }
}

hgru_params_example = [['conv2_2', 9],
                    ['conv3_3', 9],
                    ['conv4_3', 9],
                    ['conv5_3', 9],]

class BasehGRU(nn.Module):
    def __init__(self, 
                base_ff=vgg_example,
                hgru_params=hgru_params_example,
                timesteps=6,
                ):

        super().__init__()

        self.hgru_params = hgru_params
        self.timesteps = timesteps
        self.base_ff = model_tools.get_model(base_ff)
        # self.base_ff = models.vgg16(pretrained=True)
        self.build_fb_layers()
    
    def build_fb_layers(self):
        self.h_units = []
        self.ds_blocks = []
        
        prev_pos = 0
        base_layers = self.base_ff.layers
        base_filters = self.base_ff.filters
        
        # downsampling and horizontal units
        for pos,k_size in self.hgru_params:

            layer_pos = base_layers.index(pos)+1
            feats = base_filters[base_layers.index(pos)]

            h_unit = hConvGRUCell(input_size = feats, 
                            hidden_size = feats, 
                            kernel_size = k_size)
            h_unit.train()
            self.h_units.append(h_unit)
            self.ds_blocks.append(self.create_ds_block(base_layers[prev_pos:layer_pos]))
            prev_pos = layer_pos

        # last downsampling output block
        if layer_pos+1 < len(base_layers):
            self.output_block = self.create_ds_block(base_layers[prev_pos:len(base_layers)])
        
        self.input_block = self.ds_blocks.pop(0)
        self.output_feats = feats
        self.h_units = nn.ModuleList(self.h_units)
        self.ds_blocks = nn.ModuleList(self.ds_blocks)

    def create_ds_block(self,base_layers):
        # ds block: depends on base_ff non_linearity and bn
        module_list = []
        for l in base_layers:
            module_list.append(getattr(self.base_ff,l))
            if 'conv' in l:
                module_list.append(nn.ReLU())
        
        return nn.Sequential(*module_list) if len(module_list)>1 else module_list[0]

    def forward(self, x, return_hidden=False):
        
        h_hidden = [None] * len(self.h_units)
        conv_input = self.input_block(x)
        last_hidden = []
        
        selector = torch.ones(x.shape[0]).byte().to(x.device)
        accum_h = torch.zeros(x.shape[0]).to(x.device)
        step_count = torch.zeros(x.shape[0]).to(x.device)
        step_ponder_cost = torch.zeros(x.shape[0]).to(x.device)

        for i in range(self.timesteps):
            x = conv_input

            idxs = selector.nonzero().squeeze(1)
            step_ponder_cost[idxs] =  -accum_h[idxs]
            accum_dist = 0

            for l in range(len(self.h_units)):
                if i==0:
                    hidden = torch.empty_like(x)
                    init.xavier_normal_(hidden)
                    h_hidden[l] = hidden
                hidden, _ = self.h_units[l](x, h_hidden[l], timestep=i)
                dist = (hidden - h_hidden[l]) ** 2
                accum_dist += dist.view(dist.size(0), -1).mean(1)

                h_hidden[l] = hidden
                if l<len(self.ds_blocks):
                    x = self.ds_blocks[l](hidden)
                else:
                    x = hidden
            
            accum_dist /= len(self.h_units)
            ponder = 1 - torch.exp(-(0.01 / accum_dist))
                        
            accum_h[idxs] = accum_h[idxs] + ponder
            step_count[idxs] += 1
            selector = (accum_h < 1 - self.eps).data
            if not selector.any():
                break

        if return_hidden:
            last_hidden = torch.stack(last_hidden,dim=1)
            return x, last_hidden, step_ponder_cost, step_count
        else:
            return x, step_ponder_cost, step_count