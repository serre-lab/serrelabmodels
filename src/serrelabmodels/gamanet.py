#c_input-pool          readout
#  h------------------>td ------->
#  c-p                 us
#    h-------------->td --------->
#    c-p             us
#      h---------->td ----------->
#      c-p         us
#        h------>td ------------->
#        c-p     us
#            h ------------------>

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from serrelabmodels.layers.fgru_base import fGRUCell
from serrelabmodels.layers.fgru_base import fGRUCell_topdown as fGRUCell_td

from serrelabmodels.utils.pt_utils import Conv2dSamePadding
from serrelabmodels.utils import pt_utils

from serrelabmodels.ops import model_tools

vgg_example={
    'name': 'vgg_16',
    'import_prepath': 'serrelabmodels.models.vgg_16',
    'import_class': 'VGG_16',
    'args': {
        'weight_path': '/media/data_cifs/model_weights/vgg16.pth',
        'load_weights': True,
        'freeze_layers': True,
        'n_layers': 17
    }
}

gn_params_example = [['conv2_2', 3],
                    ['conv3_3', 3],
                    ['conv4_3', 3],
                    ['conv5_3', 1],
                    ['conv4_3', 1],
                    ['conv3_3', 1],
                    ['conv2_2', 1]]

class BaseGN(nn.Module):
    def __init__(self, 
                base_ff=vgg_example, 
                gn_params=gn_params_example, 
                timesteps=6, 
                hidden_init='identity',
                attention='gala', # 'se', None
                attention_layers=2,
                saliency_filter_size=5,
                norm_attention=False,
                normalization_fgru='InstanceNorm2d',
                normalization_fgru_params={'affine': True},
                normalization_gate='InstanceNorm2d',
                normalization_gate_params={'affine': True},
                force_alpha_divisive=True,
                force_non_negativity=True,
                multiplicative_excitation=True,
                ff_non_linearity='ReLU',
                us_resize_before_block=True,
                skip_horizontal=True,
                readout=True,
                readout_feats=1):
        super().__init__()
        self.timesteps = timesteps
        self.gn_params = gn_params

        assert len(gn_params)%2==1, 'the number of fgrus is not odd'
        self.gn_layers = len(gn_params)//2 +1
        
        self.ff_non_linearity = ff_non_linearity
        self.normalization_fgru = normalization_fgru

        self.us_resize_before_block = us_resize_before_block

        self.normalization_fgru_params = normalization_fgru_params
        self.fgru_params = {
            'hidden_init'               : hidden_init,
            'attention'                 : attention,
            'attention_layers'          : attention_layers,
            'saliency_filter_size'      : saliency_filter_size,
            'norm_attention'            : norm_attention,
            'normalization_fgru'        : normalization_fgru,
            'normalization_fgru_params' : normalization_fgru_params,
            'normalization_gate'        : normalization_gate,
            'normalization_gate_params' : normalization_gate_params,
            'ff_non_linearity'          : ff_non_linearity,
            'force_alpha_divisive'      : force_alpha_divisive,
            'force_non_negativity'      : force_non_negativity,
            'multiplicative_excitation' : multiplicative_excitation,
            'timesteps'                 : timesteps
        }
        
        self.base_ff = model_tools.get_model(base_ff)

        self.build_fb_layers()
        self.skip_horizontal = skip_horizontal
        self.use_readout=False
        if readout:
            self.use_readout=True
            self.build_readout(readout_feats)
    
    def build_fb_layers(self):

        self.h_units = []
        self.ds_blocks = []

        self.td_units = []
        self.us_blocks = []
        
        prev_pos = 0
        base_layers = self.base_ff.layers
        base_filters = self.base_ff.filters
        
        # downsampling and horizontal units
        for i in range(self.gn_layers):
            pos,k_size = self.gn_params[i]
            layer_pos = base_layers.index(pos)+1
            
            feats = base_filters[base_layers.index(pos)]

            h_unit = fGRUCell(input_size = feats, 
                            hidden_size = feats, 
                            kernel_size = k_size,
                            **self.fgru_params)
            h_unit.train()
            self.h_units.append(h_unit)

            self.ds_blocks.append(self.create_ds_block(base_layers[prev_pos:layer_pos]))
            prev_pos = layer_pos

        # last downsampling output block
        if layer_pos+1 < len(base_layers):
            self.output_block = self.create_ds_block(base_layers[prev_pos:len(base_layers)])
            
        td_feats = feats

        # upsampling and topdown units
        for i in range(self.gn_layers,len(self.gn_params)):
            pos,k_size = self.gn_params[i]
            layer_pos = base_layers.index(pos)+1
            
            feats = base_filters[base_layers.index(pos)]

            td_unit = fGRUCell_td(input_size = feats, 
                            hidden_size = feats, 
                            kernel_size = k_size,
                            **self.fgru_params)
            td_unit.train()
            self.td_units.append(td_unit)

            us_block = self.create_us_block(td_feats,feats)
            self.us_blocks.append(us_block)

            td_feats = feats

        self.input_block = self.ds_blocks.pop(0)
        
        self.output_feats = td_feats
        
        self.h_units = nn.ModuleList(self.h_units)
        self.ds_blocks = nn.ModuleList(self.ds_blocks)

        self.td_units = nn.ModuleList(self.td_units)
        self.us_blocks = nn.ModuleList(self.us_blocks)

    def create_ds_block(self,base_layers):
        # ds block: depends on base_ff non_linearity and bn
        module_list = []
        for l in base_layers:
            module_list.append(getattr(self.base_ff,l))
            if 'conv' in l:
                module_list.append(nn.ReLU())
            
            
        
        return nn.Sequential(*module_list) if len(module_list)>1 else module_list[0]

    def create_us_block(self,input_feat, output_feat):
        # us options: norm top_h, resize before or after block, ...
        normalization_fgru = pt_utils.get_norm(self.normalization_fgru)
        
        
        norm = normalization_fgru(input_feat,**self.normalization_fgru_params)
        init.constant_(norm.weight, 0.1)
        init.constant_(norm.bias, 0)
        conv1 = Conv2dSamePadding(input_feat,output_feat,1)
        init.kaiming_normal_(conv1.weight)
        init.constant_(conv1.bias, 0)
        nl1 = nn.ReLU()
        conv2 = Conv2dSamePadding(output_feat,output_feat,1)
        init.kaiming_normal_(conv2.weight)
        init.constant_(conv2.bias, 0)
        nl2 = nn.ReLU()
        
        module_list = [norm,conv1,nl1,conv2,nl2]
        # bilinear resize -> dependent on the other size
        # other version : norm -> conv 1*1 -> norm -> (extra conv 1*1 ->) resize
        # other version : transpose_conv 4*4/2 -> conv 3*3 -> norm
        return nn.Sequential(*module_list)
        
    def us_block(self, block, input_, out_size, resize_before_block=None):
        if resize_before_block is None:
            resize_before_block = self.us_resize_before_block
        if resize_before_block:
            input_ = F.interpolate(input_,out_size, mode='bilinear',align_corners=True)
            output = block(input_) 
        else:
            input_ = block(input_)
            output = F.interpolate(input_,out_size, mode='bilinear',align_corners=True) 
        
        return output

    def build_readout(self, readout_feats):
        normalization_fgru = pt_utils.get_norm(self.normalization_fgru)
        self.readout_norm = normalization_fgru(self.output_feats,**self.normalization_fgru_params)
        init.constant_(self.readout_norm.weight, 0.1)
        init.constant_(self.readout_norm.bias, 0)
        self.readout_conv = Conv2dSamePadding(self.output_feats, readout_feats, 1)
        init.kaiming_normal_(self.readout_conv.weight)
        init.constant_(self.readout_conv.bias, 0)


    def readout(self, input_, output_size):
        x = self.readout_norm(input_)
        x = F.interpolate(x,output_size, mode='bilinear',align_corners=True) 
        x = self.readout_conv(x)

        return x

    def forward(self, inputs, return_hidden=False):
        x = inputs
        h_hidden = [None] * len(self.h_units)
        conv_input = self.input_block(x)

        last_hidden = []
        for i in range(self.timesteps):
            x = conv_input

            # down_sample
            for l in range(len(self.h_units)):
                hidden, _ = self.h_units[l](x, h_hidden[l], timestep=i)
                h_hidden[l] = hidden
                if l<len(self.ds_blocks):
                    x = self.ds_blocks[l](hidden)
                else:
                    x = hidden
            
            # up_sample
            for l,h in enumerate(reversed(h_hidden[:-1])):
                x = self.us_block(self.us_blocks[l], x, h.shape[2:])
                x, _ = self.td_units[l](h, x, timestep=i)
                if self.skip_horizontal:
                    x += h
                h_hidden[len(self.td_units)-l-1] = x

            if return_hidden:
                last_hidden.append(x)

        if self.use_readout:
            x = self.readout(x, inputs.shape[2:])

        if return_hidden:
            last_hidden = torch.stack(last_hidden,dim=1)
            if self.use_readout:
                in_shape = last_hidden.shape
                last_hidden = self.readout(last_hidden.view((-1,)+in_shape[2:]), inputs.shape[2:])
                out_shape = last_hidden.shape
                last_hidden = last_hidden.view(in_shape[:2] + out_shape[1:])

            return x, last_hidden
        else:
            return x
        

class BaseResNetGN(BaseGN):

    def create_ds_block(self,base_layers):
        # ds block: depends on base_ff non_linearity and bn
        module_list = []
        for l in base_layers:
            module_list.append(getattr(self.base_ff,l))
            if 'bn1' in l:
                module_list.append(nn.ReLU())
            
        return nn.Sequential(*module_list) if len(module_list)>1 else module_list[0]
 