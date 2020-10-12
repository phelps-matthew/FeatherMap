# &#x1f54a; FeatherMap

## What is FeatherMap?
FeatherMap is a tool that compresses deep neural networks. Centered around computer vision models, it implements the Google Research paper [Structured Multi-Hashing for Model Compression (CVPR 2020)](references/Structured_Multi-Hashing_for_Model_Compression_CVPR_2020.pdf). Taking the form of a Python package, the tool takes a user-defined PyTorch model and compresses it to a desired factor without modification to the underlying architecture. Using it's simple API, FeatherMap can easily be applied across a broad array of models. 

## Table of Contents
  * [Installation](#installation)
  * [Usage](#usage)
  * [Results](#results)
  * [What is Structured Multi-Hashing?](#what-is-structured-multi-hashing)

## Installation
* Clone into directory `<my_dir>`
```
git clone https://github.com/phelps-matthew/FeatherMap.git
```
* Enable virtual environment (optional)
```
python -m venv myvenv
<my_dir> myvenv/bin/activate
```
* Install package
``` 
pip install <my_dir>

# Or, to make package editable
pip install -e <my_dir>
```
## Usage
### General Usage
To apply to a ResNet-34, simply import the model and wrap with `FeatherNet` module, selecting desired compression.
```python
from feathermap.models.resnet import ResNet34
from feathermap.feathernet import FeatherNet

base_model = ResNet34()
model = FeatherNet(base_model, compress=0.10)

# Forward pass ...
y = model(x)
loss = criterion(y, target)

# Backward and optimize ...
loss.backward()
optimizer.step()
```
### Training
Models may be trained using `main.py`. See the argument options by using the help flag `--h`.
```bash
python main.py --compress 0.1
```
### Deployment
Upon defining your `FeatherNet` model, switch to deploy mode to calculate weights on the fly
```python
base_model = ResNet34()
model = FeatherNet(base_model, compress=0.10)
model.deploy()
```
## Results
As applied to a ResNet-34 architecture, trained and tested on CIFAR-10. Latency benchmarked on CPU (AWS c5a.8xlarge) iterating over 30k images with batch size of 100.
<p align="center"> <img src="/references/resnet34_acc_latency.png"  width="2500"> </p>



## What is Structured Multi-Hashing?
<p align="center"> <img src="/references/smh1.png"  width="2500"> </p>


