import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init


from serrelabmodels.layers.hgru_cell import hGRUCell


class VGG_16(nn.Module):
    def __init__(self, weight_path, load_weights=True, freeze_layers=True, n_layers=17):
        super().__init__()
        self.layers = [ 'conv1_1','conv1_2','maxpool1',
                        'conv2_1','conv2_2','maxpool2',
                        'conv3_1','conv3_2','conv3_3','maxpool3',
                        'conv4_1','conv4_2','conv4_3','maxpool4',
                        'conv5_1','conv5_2','conv5_3']
        self.layers = self.layers[:n_layers]
        
        self.filters = [64,      64,      64,      
                        128,     128,     128,     
                        256,     256,     256,    256,
                        512,     512,     512,    512,
                        512,     512,     512]
                        
        self.filters = self.filters[:n_layers]

        self.build_layers()
        
        self.weight_path = weight_path

        if load_weights:
            self.load_state_dict(torch.load(weight_path),strict=False)
        if freeze_layers:
            self.freeze_layers()  
        
    
    def build_layers(self):
        #vgg model
        prev_fan = 3

        for l, f in zip(self.layers, self.filters):
            if 'conv' in l:
                setattr(self, l, nn.Conv2d(prev_fan, f, kernel_size=3, padding=1))
            else:
                setattr(self, l, nn.MaxPool2d(kernel_size=2, stride=2))
            prev_fan = f

    def freeze_layers(self,layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = False
                getattr(self,l).bias.requires_grad = False

    def unfreeze_layers(self, layers=None):
        if layers is None:
            layers = self.layers

        for l in layers:
            if 'conv' in l:
                getattr(self,l).weight.requires_grad = True
                getattr(self,l).bias.requires_grad = True

    def load_layers(self):
        
        weights = np.load(self.weight_path).item()
        for l in self.layers:
            if 'conv' in l:
                getattr(self,l).weight.data = torch.FloatTensor(weights[l]['weight'])
                getattr(self,l).bias.data = torch.FloatTensor(weights[l]['bias'])

    def forward(self, inputs):
        x = inputs

        for l in self.layers:
            if 'conv' in l:
                x = F.relu(getattr(self,l)(x))
            else:
                x = getattr(self,l)(x)
        
        return x