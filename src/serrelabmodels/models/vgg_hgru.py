import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init

from serrelabmodels.models.vgg_16 import VGG_16
from serrelabmodels.ops import model_tools

from serrelabmodels.layers.hgru_cell import hGRUCell


class VGG_16_hGRU(nn.Module):
    def __init__(self, weights_path, load_weights=True, add_hgru=[], filter_size=9, timesteps=6):
        super().__init__()
        
        self.timesteps = timesteps
        self.weight_path = weight_path
        self.add_hgru = add_hgru

        self.n_hgru_layers = len(add_hgru)

        self.base_ff = VGG_16(weights_path=weights_path, load_weights=load_weights)

        self.build_fb_layers()
    
    def build_fb_layers(self):

        self.hgru_units = []
        self.hgru_positions = []
        self.base_blocks = []
        prev_pos = 0
        for pos,f_size in self.add_hgru:
            layer_pos = self.base_ff.layers.index(pos)+1
            
            f = self.base_ff.filters[self.base_ff.layers.index(pos)]
            unit = hConvGRUCell(f, f, f_size)
            unit.train()
            self.hgru_units.append(unit)

            block_layers = [getattr(self.base_ff,k) for k in self.base_ff.layers[prev_pos:layer_pos]]
            
            self.base_blocks.append(nn.Sequential(*block_layers) if len(block_layers)==1 else block_layers[0])    
            prev_pos = layer_pos

        if layer_pos < len(self.base_ff.layers):

            block_layers = [getattr(self.base_ff,k) for k in self.base_ff.layers[prev_pos:len(self.base_ff.layers)]]
            self.base_blocks.append(nn.Sequential(*block_layers) if len(block_layers)==1 else block_layers[0])

        self.input_block = self.base_blocks.pop(0)
        if len(self.base_blocks) > len(self.hgru_units):
            self.output_block = self.base_blocks.pop(-1)

        self.hgru_units = nn.ModuleList(self.hgru_units)
        self.base_blocks = nn.ModuleList(self.base_blocks)
        
    def build_readout(self):
        pass

    def forward(self, inputs):
        x = inputs
        hgru_hidden = [None] * self.n_hgru_layers
        conv_input = self.input_block(x)

        last_hidden = []

        for i in range(self.timesteps):
            x = conv_input
            for l in range(self.n_hgru_layers):
                hidden, _ = self.hgru_units[l](x,hgru_hidden[l], timestep=i)
                hgru_hidden[l] = hidden
                if l<len(self.base_blocks):
                    x = self.base_blocks[l](hidden)
                else:
                    x = hidden
            if return_hidden:
                last_hidden.append(x)

        if return_hidden:
            last_hidden = torch.stack(last_hidden,dim=1)
            if hasattr(self, 'output_block'):
                in_shape = last_hidden.shape.tolist()
                last_hidden = self.output_block(last_hidden.view([-1]+in_shape[2:]))
                out_shape = last_hidden.shape.tolist()
                last_hidden = last_hidden.view(in_shape[:2] + out_shape[1:])
            if hasattr(self,'readout'):
                in_shape = last_hidden.shape.tolist()
                last_hidden = self.readout(last_hidden.view([-1]+in_shape[2:]))
                out_shape = last_hidden.shape.tolist()
                last_hidden = last_hidden.view(in_shape[:2] + out_shape[1:])
            
        if hasattr(self, 'output_block'):
            x = self.output_block(x)

        if hasattr(self,'readout'):
            x = self.readout(x)
        
        return x, last_hidden
