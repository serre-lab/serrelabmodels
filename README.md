# Serre Lab In-house Models
Repository to create a public python library to import in-house models.

## Models:
The repository currently includes the following models:
1. hGRU
2. fGRU
3. KuraNet
4. Gamanet

## Usage:
### Installing the module from pip

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps serrelabmodels
```

*_NOTE:_* This repository is uploaded to TestPyPi, and will be moved to PyPi when complete.

---

### Importing the required model

#### hGRU

```python
import serrelabmodels.base_hgru
hgru_model = serrelabmodels.base_hgru.BasehGRU()
```

#### fGRU

```
# In-progress
```

#### KuraNet

```python
import serrelabmodels.kuranet
kuranet_model = serrelabmodels.kuranet.KuraNet(<feature_dimensions>)
```

#### GamaNet

```python
import serrelabmodels.base_gamanet
gamanet_model = serrelabmodels.base_gamanet.BaseGN()
```

## Examples:

#### hGRUCell

```python
>>> import serrelabmodels.layers.hgru_base as hgru_base
>>> h = hgru_base.hConvGRUCell(5, 5, 3)
>>> h2 = hgru_base.hConvGRUCell(5, 5, 3, batchnorm = False)
>>> h2
hConvGRUCell(
  (u1_gate): Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1))
  (u2_gate): Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1))
)
>>> h
hConvGRUCell(
  (u1_gate): Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1))
  (u2_gate): Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1))
  (bn): ModuleList(
    (0): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
```


#### hGRU

```python
>>> import serrelabmodels.base_hgru
agg
>>> b = serrelabmodels.base_hgru.BasehGRU()
importing  serrelabmodels.models.vgg_16 . VGG_16
>>> b
BasehGRU(
  (base_ff): VGG_16(
    (conv1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (input_block): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU()
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU()
  )

  ...
  
  (ds_blocks): ModuleList(
    (0): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (2): ReLU()
      (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU()
    )
    (1): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (2): ReLU()
      (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU()
    )
    (2): Sequential(
      (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (2): ReLU()
      (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): ReLU()
      (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU()
    )
  )
)

```


#### KuraNet

```python
>>> import serrelabmodels.kuranet
>>> k = serrelabmodels.kuranet.KuraNet(5)
>>> k
KuraNet(
  (layers): Sequential(
    (0): Linear(in_features=10, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.01)
    (3): Linear(in_features=128, out_features=128, bias=True)
    (4): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): LeakyReLU(negative_slope=0.01)
    (6): Linear(in_features=128, out_features=1, bias=False)
  )
)

```

### GamaNet

```python
>>> import serrelabmodels.base_gamanet
>>> g = serrelabmodels.base_gamanet.BaseGN()
importing  serrelabmodels.models.vgg_16 . VGG_16
>>> g
BaseGN(
  (base_ff): VGG_16(
    (conv1_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv1_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2_1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv3_1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3_2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3_3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv4_1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (maxpool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv5_1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5_2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5_3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (input_block): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU()
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU()
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU()
  )
    
    ...
    
  (us_blocks): ModuleList(
    (0): Sequential(
      (0): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (1): Conv2dSamePadding(512, 512, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (2): ReLU()
      (3): Conv2dSamePadding(512, 512, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (4): ReLU()
    )
    (1): Sequential(
      (0): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (1): Conv2dSamePadding(512, 256, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (2): ReLU()
      (3): Conv2dSamePadding(256, 256, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (4): ReLU()
    )
    (2): Sequential(
      (0): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (1): Conv2dSamePadding(256, 128, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (2): ReLU()
      (3): Conv2dSamePadding(128, 128, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
      (4): ReLU()
    )
  )
  (readout_norm): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
  (readout_conv): Conv2dSamePadding(128, 1, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
)

```
