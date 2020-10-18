# &#x1f54a; FeatherMap

## What is FeatherMap?
FeatherMap is a tool that compresses deep neural networks. Centered around computer vision models, it implements the Google Research paper [Structured Multi-Hashing for Model Compression (CVPR 2020)](references/Structured_Multi-Hashing_for_Model_Compression_CVPR_2020.pdf). Taking the form of a Python package, the tool takes a user-defined PyTorch model and compresses it to a desired factor without modification to the underlying architecture. Using its simple API, FeatherMap can easily be applied across a broad array of models. 

## Table of Contents
  * [Installation](#installation)
  * [Usage](#usage)
    + [General Usage](#general-usage)
    + [Training](#training)
    + [Deployment](#deployment)
  * [Results](#results)
  * [What is Structured Multi-Hashing?](#what-is-structured-multi-hashing)

## Installation
* Clone into directory
```
git clone https://github.com/phelps-matthew/FeatherMap.git
cd FeatherMap
```
* Pip Install
``` 
pip install -e .
```
* Conda Install
```
conda develop .
```
## Usage
### General Usage
To compress a model such as Resnet-34, import the model from `feathermap/models/` and simply wrap the model with the `FeatherNet` module, initializing with the desired compression. One can then proceed with forward and backward passes as normal, as well as `state_dict` loading and saving.
```python
from feathermap.models.resnet import ResNet34
from feathermap.feathernet import FeatherNet
import torch.nn as nn

base_model = ResNet34()
model = FeatherNet(base_model, compress=0.10)

# Forward pass ...
y = model(x)
loss = criterion(y, target)
...

# Backward and optimize ...
loss.backward()
optimizer.step()
...
```
See `feathermap/models/` for a zoo of available CV models to compress.
### Training
Models are trained on CIFAR-10 using `feathermap/train.py` (defaults to training ResNet-34). See the argument options by using the help flag `--help`.
```bash
python train.py --compress 0.1
```

### Deployment
Upon defining your `FeatherNet` model, switch to deploy mode to calculate weights on the fly (see [What is Structured Multi-Hashing?](#what-is-structured-multi-hashing)).
```python
base_model = ResNet34()
model = FeatherNet(base_model, compress=0.10)
model.deploy()
```

## Results
Below are results as applied to a ResNet-34 architecture, trained and tested on CIFAR-10. Latency benchmarked on CPU (AWS c5a.8xlarge) iterating over 30k images with batch size of 100. To add some context, one can compress ResNet-34 to 2% of its original size while still achieving over 90% accuracy (a 5% accuracy drop compared to the base model), while incurring only a 4% increase in latency time.
<p align="center"> <img src="/references/resnet34_acc_latency.png"  width="5000"> </p>



## What is Structured Multi-Hashing?
There are two main concepts behind structured multi-hashing. The first concept is to take the weights of each *layer*, flatten them, and tile them into a single square  matrix. This *global weight matrix* represents the weights of the entire network.
<p align="center"> <img src="/references/smh_1.png"  width="550"> </p>
The second concept is purely linear algebra and it is the understanding that if we take a pair of columns and matrix-multiply them by a pair of rows, we obtain a square matrix.
<p align="center"> <img src="/references/smh_2.png"  width="550"> </p>
Putting these two ideas together, we can implement structured multi-hashing! Here's how it works:

1. Let the total number of tunable parameters describing the entire network be the set of two rows (2 x n) and two columns (n x 2)
2. Matrix multiply the columns and rows to obtain a square matrix of size (n x n)
3. Map each element of the matrix above to each element in the *global weight matrix*

Putting it all together, we have this process.
<p align="center"> <img src="/references/smh_3.png"  width="900"> </p>

What we have effectively done with this mapping is a reduction of the number of *tunable parameters* from n^2 to 4n, thus achieving the desired compression! 

Additional Remarks:
- To obtain a target compression factor, generalize the respective dimension of the rows and columns from 2 to m, to thus begin with a total of 2mn tunable parameters. The compression factor will then be 2mn/n^2 = 2m/n. By varying m, one can achieve varying levels of compression.
- For practical deployment, in order to constrain RAM consumption, each weight must be calculated 'on the fly' during the foward pass. Such additional calculations will induce latency overhead; however, the 'structured' nature of this multi-hashing approach embraces memory locality and I have found that for small compression factors the overhead is minimal (see [Results](#results)).


