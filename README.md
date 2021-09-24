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

```
import serrelabmodels.base_hgru
hgru_model = serrelabmodels.base_hgru.BasehGRU()
```

#### fGRU

```
# In-progress
```

#### KuraNet

```
import serrelabmodels.kuranet
kuranet_model = serrelabmodels.kuranet.KuraNet(<feature_dimensions>)
```

#### GamaNet

```
import serrelabmodels.base_gamanet
gamanet_model = serrelabmodels.base_gamanet.BaseGN()
```

## Examples:

#### hGRU



#### KuraNet

```
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
