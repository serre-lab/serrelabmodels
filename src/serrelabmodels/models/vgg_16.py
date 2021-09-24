import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init


from serrelabmodels.layers.hgru_base import hConvGRUCell


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
            #self.load_layers()
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

        # self.conv1_1 = nn.Conv2d(  3,  64, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1)
        
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)


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
        
        # x = F.relu(self.conv1_1(x))
        # x = F.relu(self.conv1_2(x))

        # x = self.maxpool1(x)
        
        # x = F.relu(self.conv2_1(x))
        # x = F.relu(self.conv2_2(x))
        
        # x = self.maxpool2(x)

        # x = F.relu(self.conv3_1(x))
        # x = F.relu(self.conv3_2(x))
        # x = F.relu(self.conv3_3(x))

        # x = self.maxpool3(x)
        
        # x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = F.relu(self.conv4_3(x))

        # x = self.maxpool4(x)

        # x = F.relu(self.conv5_1(x))
        # x = F.relu(self.conv5_2(x))
        # x = F.relu(self.conv5_3(x))

        return x



# import torchvision
# import torch
# from collections import OrderedDict

# keys_mappings = {'features.0.weight': 'conv1_1.weight', 
#                 'features.0.bias': 'conv1_1.bias', 
#                 'features.2.weight': 'conv1_2.weight', 
#                 'features.2.bias': 'conv1_2.bias', 
#                 'features.5.weight': 'conv2_1.weight', 
#                 'features.5.bias': 'conv2_1.bias', 
#                 'features.7.weight': 'conv2_2.weight', 
#                 'features.7.bias': 'conv2_2.bias', 
#                 'features.10.weight': 'conv3_1.weight', 
#                 'features.10.bias': 'conv3_1.bias', 
#                 'features.12.weight': 'conv3_2.weight', 
#                 'features.12.bias': 'conv3_2.bias', 
#                 'features.14.weight': 'conv3_3.weight', 
#                 'features.14.bias': 'conv3_3.bias', 
#                 'features.17.weight': 'conv4_1.weight', 
#                 'features.17.bias': 'conv4_1.bias', 
#                 'features.19.weight': 'conv4_2.weight', 
#                 'features.19.bias': 'conv4_2.bias', 
#                 'features.21.weight': 'conv4_3.weight', 
#                 'features.21.bias': 'conv4_3.bias', 
#                 'features.24.weight': 'conv5_1.weight', 
#                 'features.24.bias': 'conv5_1.bias', 
#                 'features.26.weight': 'conv5_2.weight', 
#                 'features.26.bias': 'conv5_2.bias', 
#                 'features.28.weight': 'conv5_3.weight', 
#                 'features.28.bias': 'conv5_3.bias', 
#                 }


# model = torchvision.models.vgg16(pretrained=True)
# state_dict = model.state_dict()

# new_state_dict = OrderedDict()

# for k,v in keys_mappings.items():
#     new_state_dict[v] = state_dict[k]

# torch.save(new_state_dict, 'vgg_16.pth.tar')




# sd1 = torch.load('model_weights/vgg_16.pth.tar')

#'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'