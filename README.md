# Serre Lab In-house Models
- Repository to create a public python library to import in-house models of the Serre Lab at Brown University.
- The models can be imported exactly like any other PyTorch module (eg: GRU).
- The models are plug-and-play and are designed to work out of the box, with no additional tinkering required. All the paramters are the same as the PyTorch GRU module.

## Models:
The repository currently includes the following models:
1. fGRU
2. hGRU
3. hGRU Example
5. Gamanet

## Usage:
### Installing the module from pip

```
pip install serrelabmodels
```

---

### Importing the required model


#### fGRU

```python
import serrelabmodels.layers.fgru_cell as sl_fgru
fgru_cell = sl_fgru.fRGUCell(5, 5, 3)
```


#### hGRU

```python
import serrelabmodels.layers.hgru_cell as sl_hgru
hgru_cell =  sl_hgru.hRGUCell(5, 5, 3)
```


#### hGRU Example

```python
import serrelabmodels.hgru as sl_hgru_ex
hgru_model = sl_hgru_ex.BasehGRU()
```

#### GamaNet

```python
import serrelabmodels.base_gamanet as sl_gnet
gamanet_model = sl_gnet.BaseGN()
```

## Examples:

### fGRUCell

```python
>>> import serrelabmodels.layers.fgru_cell as sl_fgru
>>> fgru_cell = sl_fgru.fGRUCell(5, 5, 3)
>>> fgru_cell
fGRUCell(
  (ff_nl): ReLU()
  (attention): GALA_Attention(
    (se): SE_Attention(
      (attention): Sequential(
        (0): Conv2dSamePadding(5, 2, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
        (1): ReLU()
        (2): Conv2dSamePadding(2, 5, kernel_size=(1, 1), stride=(1, 1), padding_mode=reflect)
        (3): ReLU()
      )
    )
    (sa): SA_Attention(
      (attention): Sequential(
        (0): Conv2dSamePadding(5, 2, kernel_size=(5, 5), stride=(1, 1), padding_mode=reflect)
        (1): ReLU()
        (2): Conv2dSamePadding(2, 1, kernel_size=(5, 5), stride=(1, 1), padding_mode=reflect)
        (3): ReLU()
      )
    )
  )
  (bn_g1): InstanceNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
  (bn_c1): InstanceNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
  (bn_g2): InstanceNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
  (bn_c2): InstanceNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
)

```

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


#### hGRU Example (VGG 16)

```python
>>> import serrelabmodels.base_hgru as sl_hgru_ex
agg
>>> b = sl_hgru_ex.BasehGRU()
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
  ...
  
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


### GamaNet

```python
>>> import serrelabmodels.base_gamanet as sl_gnet
>>> g = sl_gnet.BaseGN()
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
    ...
    
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
