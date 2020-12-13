from rectpack import newPacker
from feathermap.models.resnet import ResNet, ResidualBlock
from feathermap.models.feathernet import FeatherNet
import torch.nn as nn


base_model = ResNet(ResidualBlock, [2, 2, 2])
model = FeatherNet(base_model, exclude=(nn.BatchNorm2d))


packer = newPacker()

for name, tensor in model.get_WandB():
    if tensor.dim() > 2:
        tensor = tensor.flatten(end_dim=-2)
    if tensor.dim() == 1:
        tensor = tensor.view(-1, 1)
    rect_obj = (tensor.size(1), tensor.size(0), name)
    packer.add_rect(*rect_obj)
    print(rect_obj)

bin_factor = 10
packer.add_bin(bin_factor * model.size_n, bin_factor * model.size_n)
packer.pack()

for r in packer.rect_list():
    print(r)
