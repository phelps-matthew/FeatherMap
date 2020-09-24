import torch
import torch.nn as nn
from torch.nn import Parameter
from feathermap.resnet import ResNet, ResidualBlock
from torch import Tensor, device, dtype
from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
)
from math import ceil, sqrt


class FeatherNet(nn.Module):
    """Overrides parameters() method."""

    def __init__(
        self, module: nn.Module, compress: float = 1, exclude: tuple = ()
    ) -> None:
        super().__init__()
        self.module = module
        self.compress = compress
        self.exclude = exclude
        self.unregister_params()
        self.size_n = ceil(sqrt(self.num_WandB()))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.randn(self.size_n, self.size_m))
        self.V2 = Parameter(torch.randn(self.size_n, self.size_m))
        self.V = torch.matmul(self.V1, self.V2.transpose(0, 1))

    def WandBtoV(self):
        i, j = 0, 0
        V = self.V.view(-1, 1)
        for name, v in self.get_WandB():
            v = v.view(-1, 1)
            for j in range(len(v)):
                v[j] = V[i]
                i += 1

    def num_WandB(self):
        """Return total number of weights and biases"""
        return sum(v.numel() for name, v in self.get_WandB())

    def get_WandB(self):
        for name, module, kind in self.get_WandB_modules():
            yield name + "." + kind, getattr(module, kind)

    def get_WandB_modules(self):
        """Helper function to return weight and bias modules in order"""
        for name, module in self.named_modules():
            try:
                if isinstance(module, self.exclude):
                    continue
                if getattr(module, "weight") is not None:
                    yield name, module, "weight"
                if getattr(module, "bias") is not None:
                    yield name, module, "bias"
            except nn.modules.module.ModuleAttributeError:
                pass

    def unregister_params(self):
        """Delete params, set attributes as Tensors of prior data"""
        for name, module, kind in self.get_WandB_modules():
            try:
                data = module._parameters[kind].data
                del module._parameters[kind]
                print(
                    "Parameter unregistered, assigned to type Tensor: {}".format(
                        name + "." + kind
                    )
                )
                setattr(module, kind, data)
            except KeyError:
                print(
                    "{} is already registered as {}".format(
                        name + "." + kind, type(getattr(module, kind))
                    )
                )

    def forward(self, x):
        self.WandBtoV()
        return self.module(x)



def main():
    # parameters(named_parameters(_named_members(named_modules(named_modules))))

    # Device configuration

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(ResidualBlock, [2, 2, 2]).to(my_device)
    f_model = FeatherNet(model)
    lmodel = nn.Linear(2, 4)
    flmodel = FeatherNet(lmodel, compress=0.5)
    flmodel.unregister_params()
    print(flmodel.num_WandB(), flmodel.size_n, flmodel.size_m)
    a = lmodel.weight
    print(a)
    print(flmodel.V)
    flmodel.WandBtoV()
    f_model.unregister_params()
    f_model.WandBtoV()
    print("V1: {}".format(f_model.V1))
    print("V2: {}".format(f_model.V2))
    print("V: {}".format(f_model.V))
    [print(name, v) for name, v in f_model.named_parameters()]
    # print(f_model.parameter_pop())
    # [print(name) for name, v in f_model.named_parameters()]
    # print(*list(filter(lambda x: not(isinstance(x, nn.BatchNorm2d)), f_model.modules())),sep='\n')
    # [print(name) for name, v in f_model.named_parameters()]
    # print("-" * 20)
    # f_model.del_params()
    # [print(name + '.weight') for name, m, in f_model.get_weights(exclude=(nn.BatchNorm2d))]
    # [print(name + '.bias') for name, m, in f_model.get_bias()]
    # for name, weight in f_model.get_modules(exclude=(nn.BatchNorm2d)):
    #     print(name)
    # f_model.unregister_params()
    # print("-" * 20)
    # [print(name) for name, v in f_model.named_parameters()]
    # print("-" * 20)
    # [print(name) for name, v in f_model.get_WandB(exclude=(nn.BatchNorm2d))]
    # print(model.conv.weight.numel())
    # print(model.conv.weight.size())
    # print(f_model._num_WorB())
    # print(f_model._num_WorB(kind="bias"))
    # a = f_model._num_WorB() + f_model._num_WorB(kind="bias")
    # print(a)
    # print(ceil(sqrt(a)))
    # print(f_model.size_n)
    # print(443 ** 2)
    exit()

    # print(*dict(model.named_parameters()).keys(), sep="\n")
    # print(*dict(f_model.named_parameters()).keys(), sep="\n")

    del model.fc.weight


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
