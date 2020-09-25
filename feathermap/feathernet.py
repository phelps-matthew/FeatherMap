import torch
import torch.nn as nn
from torch.nn import Parameter
from feathermap.resnet import ResNet, ResidualBlock
from torch import Tensor
from typing import Iterator, Tuple
from math import ceil, sqrt


class FeatherNet(nn.Module):
    """Implementation of structured multihashing for model compression"""

    def __init__(
        self,
        module: nn.Module,
        compress: float = 1,
        exclude: tuple = (),
    ) -> None:
        super().__init__()
        self.module = module
        self.compress = compress
        self.exclude = exclude
        self.unregister_params()
        self.size_n = ceil(sqrt(self.num_WandB()))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.Tensor(self.size_n, self.size_m))
        self.V2 = Parameter(torch.Tensor(self.size_m, self.size_n))

        self.norm_V()
        #self.WandBtoV()

    def norm_V(self):
        k = sqrt(12) / 2 * self.size_m ** (-1 / 4)
        torch.nn.init.uniform_(self.V1, -k, k)
        torch.nn.init.uniform_(self.V2, -k, k)
        #torch.nn.init.uniform_(self.V, -k, k)

    def WandBtoV(self):
        self.V = torch.matmul(self.V1, self.V2)
        i = 0
        V = self.V.view(-1, 1)
        for name, v in self.get_WandB():
            v = v.view(-1, 1)
            for j in range(len(v)):
                v[j] = V[i]
                i += 1

    def num_WandB(self) -> int:
        """Return total number of weights and biases"""
        return sum(v.numel() for name, v in self.get_WandB())

    def get_WandB(self) -> Iterator[Tuple[str, Tensor]]:
        for name, module, kind in self.get_WandB_modules():
            yield name + "." + kind, getattr(module, kind)

    def get_WandB_modules(self) -> Iterator[Tuple[str, nn.Module, str]]:
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

    def unregister_params(self) -> None:
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

    def forward(self, *args):
        self.WandBtoV()
        return self.module(*args)


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_test():
        lmodel = nn.Linear(2, 4).to(device)
        flmodel = FeatherNet(lmodel, compress=0.5)
        print(flmodel.num_WandB(), flmodel.size_n, flmodel.size_m)
        print("V1: {}".format(flmodel.V1))
        print("V2: {}".format(flmodel.V2))
        print("V: {}".format(flmodel.V))
        flmodel.WandBtoV()
        [print(name, v) for name, v in flmodel.named_parameters()]

    def res_test():
        rmodel = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        frmodel = FeatherNet(rmodel).to(device)
        print(frmodel.num_WandB(), frmodel.size_n, frmodel.size_m)
        print("V1: {}".format(frmodel.V1))
        print("V2: {}".format(frmodel.V2))
        print("V: {}".format(frmodel.V))
        frmodel.WandBtoV()
        [print(name, v) for name, v in frmodel.named_parameters()]

    linear_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
