# &#x1f54a; FeatherMap

## What is FeatherMap?
FeatherMap is a tool that compresses deep neural networks. Centered around computer vision models, it implements the Google Research paper [Structured Multi-Hashing for Model Compression (CVPR 2020)](references/Structured_Multi-Hashing_for_Model_Compression_CVPR_2020.pdf). Taking the form of a Python package, the tool takes a user-defined PyTorch model and compresses it to a desired factor without modification to the underlying architecture. Using it's simple API, FeatherMap can easily be applied across a broad array of models. 

## Table of Contents
  * [Installation](#installation)
  * [Usage](#usage)
  * [Results](#results)
  * [What is Structured Multi-Hashing?](#what-is-structured-multi-hashing-)

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

## Results
As applied to a ResNet-34 architecture, trained and tested on CIFAR-10. Latency benchmarked on AWS c5a.8xlarge iterating over 30k images with batch size of 100.
<p align="center"> <img src="/references/resnet34_acc_latency.png"  width="2500"> </p>



## What is Structured Multi Hashing?
<p align="center"> <img src="/references/smh1.png"  width="2500"> </p>


