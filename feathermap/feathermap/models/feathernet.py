import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator, Tuple
from math import ceil, sqrt
import copy


class FeatherNet(nn.Module):
    """Implementation of "Structured Multi-Hashing for Model Compression"""

    def __init__(
        self,
        module: nn.Module,
        compress: float = 1,
        exclude: tuple = (),
        clone: bool = True,
    ) -> None:
        super().__init__()
        self.module = copy.deepcopy(module) if clone else module
        self.compress = compress
        self.exclude = exclude

        # Unregister module Parameters, create scaler attributes
        self.unregister_params()

        self.size_n = ceil(sqrt(self.num_WandB()))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.Tensor(self.size_n, self.size_m))
        self.V2 = Parameter(torch.Tensor(self.size_m, self.size_n))

        # Noramlize V1 and V2
        self.norm_V()

    def norm_V(self):
        """Currently implemented only for uniform intializations"""
        # sigma = M**(-1/4); bound follows from uniform dist.
        bound = sqrt(12) / 2 * (self.size_m ** (-1 / 4))
        torch.nn.init.uniform_(self.V1, -bound, bound)
        torch.nn.init.uniform_(self.V2, -bound, bound)

    def WandBtoV(self):
        """Needs to be efficient"""
        self.V = torch.matmul(self.V1, self.V2)
        V = self.V.view(-1, 1)  # V.is_contiguous() = True
        i = 0
        for name, module, kind in self.get_WandB_modules():
            v = getattr(module, kind)
            j = v.numel()  # elements in weight or bias
            v_new = V[i : i + j].reshape(v.size())  # confirmed contiguous

            # Scaler Parameter, e.g. nn.Linear.weight_p
            scaler = getattr(module, kind + "_p")
            # Update weights and biases, point to elems in V
            setattr(module, kind, scaler * v_new)
            i += j

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
        """Delete params, set attributes as Tensors of prior data,
        register scaler params"""
        # fan_in will fail on BatchNorm2d.weight
        for name, module, kind in self.get_WandB_modules():
            try:
                data = module._parameters[kind].data
                if kind == "weight":
                    fan_in = torch.nn.init._calculate_correct_fan(data, "fan_in")
                # get bias fan_in from corresponding weight
                else:
                    fan_in = torch.nn.init._calculate_correct_fan(
                        getattr(module, "weight"), "fan_in"
                    )
                del module._parameters[kind]
                scaler = 1 / sqrt(3 * fan_in)
                setattr(module, kind, data)
                # Add scale parameter to each weight or bias
                module.register_parameter(
                    kind + "_p", Parameter(torch.Tensor([scaler]))
                )
                print(
                    "Parameter unregistered, assigned to type Tensor: {}".format(
                        name + "." + kind
                    )
                )
            except KeyError:
                print(
                    "{} is already registered as {}".format(
                        name + "." + kind, type(getattr(module, kind))
                    )
                )
            except ValueError:
                print(
                    "Check module exclusion list. Note, cannot calculate fan_in\
                    for BatchNorm2d layers."
                )
                raise TypeError

    def load_state_dict(self, *args, **kwargs):
        """Update weights and biases from stored V1, V2 values"""
        out = nn.Module.load_state_dict(self, *args, *kwargs)
        self.WandBtoV()
        return out

    def eval(self, *args, **kwargs):
        """Update weights and biases from final-most batch after training"""
        self.WandBtoV()
        return nn.Module.eval(self, *args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.training:
            self.WandBtoV()
        return self.module(*args, **kwargs)


def main():
    from feathermap.resnet import ResNet, ResidualBlock

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
        frmodel = FeatherNet(rmodel, exclude=(nn.BatchNorm2d), compress=0.5).to(device)
        # print(frmodel.num_WandB(), frmodel.size_n, frmodel.size_m)
        # print("V1: {}".format(frmodel.V1))
        # print("V2: {}".format(frmodel.V2))
        # print("V: {}".format(frmodel.V))
        # frmodel.WandBtoV()
        for name, mod, kind in frmodel.get_WandB_modules():
            # print(name + "." + kind)
            pass
        for name, param in rmodel.named_parameters():
            print(name)
        for name, param in frmodel.named_parameters():
            print(name)
        # Compare compressed sizes; confirmed.
        torch.save(frmodel.state_dict(), "feather.ckpt")
        torch.save(rmodel.state_dict(), "resnet.ckpt")

    res_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
