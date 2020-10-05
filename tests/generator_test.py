import torch

def gen():
    for i in range(10):
        yield torch.randn(1)

v = [gen()]
print(v)
print(next(v[0]))
print(v[0])
print(next(v[0]))
