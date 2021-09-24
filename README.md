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
kuranet_model = serrelabmodels.kuranet.KuraNet(<parameters>)
```

#### GamaNet

```
import serrelabmodels.base_gamanet
gamanet_model = serrelabmodels.base_gamanet.BaseGN()
```
