import torch
import torch.nn as nn


model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())
x = torch.randn(1, 1, 32, 32)


def prehook(module, inputs):
    print("Prehook: ", module)


def posthook(module, inputs, outputs):
    print("Posthook: ", module)


for module in model.modules():
    module.register_forward_pre_hook(prehook)
    module.register_forward_hook(posthook)

print(x, model(x))
