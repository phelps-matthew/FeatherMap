import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator, Tuple
from math import ceil, sqrt
import copy
from timeit import default_timer as timer


class LoadLayer:
    """Forward prehook for inner layers"""

    def __init__(self, name, module, V):
        self.module = module
        self.name = name
        self.V = V
        self.w_size = module.weight.size()
        self.w_num = module.weight.numel()
        self.bias = module.bias is not None
        if self.bias:
            self.b_size = module.bias.size()
            self.b_num = module.bias.numel()

    def __call__(self, module, inputs):
        print("prehook activated: {} {}".format(self.name, self.module))
        w = torch.empty(self.w_num)
        V_iterator = self.V[0]
        for i in range(self.w_num):
            w[i] = next(V_iterator)
            module.weight = w.reshape(self.w_size)
        if self.bias:
            b = torch.empty(self.b_num)
            for i in range(self.b_num):
                b[i] = next(V_iterator)
                module.bias = b.reshape(self.b_size)
        print(self.V)


class UnloadLayer:
    """Forward hook for inner layers. Note, could turn into function"""

    def __init__(self, name, module):
        self.module = module
        self.name = name

    def __call__(self, module, inputs, outputs):
        print("posthook activated: {} {}".format(self.name, self.module))
        module.weight = None
        module.bias = None


def construct_V(module, inputs):
    """Forward prehook for outermost FeatherNet layer"""
    print("Outer prehook activated: {}".format(module.V[0]))
    module.V[0] = module.V_iter()


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

        self.prehooks = None
        self.posthooks = None
        self.prehook_outer = None

        # Noramlize V1 and V2
        self.norm_V()

        # Use list as pointer
        self.V = [self.V_iter()]
        self.register_inter_hooks()
        self.register_outer_hooks()

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

    def get_WorB_modules(self) -> Iterator[Tuple[str, nn.Module, str]]:
        """Helper function to return weight and bias modules in order"""
        for name, module in self.named_modules():
            try:
                if isinstance(module, (self.exclude, FeatherNet)):
                    continue
                if getattr(module, "weight") is not None:
                    yield name, module
            except nn.modules.module.ModuleAttributeError:
                pass

    def V_iter(self):
        for i in range(self.size_n):
            for j in range(self.size_n):
                yield torch.dot(self.V1[i, :], self.V2[:, j])

    def register_inter_hooks(self):
        prehooks, posthooks = [], []
        for name, module in self.get_WorB_modules():
            load_layer = LoadLayer(name, module, self.V)
            unload_layer = UnloadLayer(name, module)
            module.register_forward_pre_hook(load_layer)
            module.register_forward_hook(unload_layer)
            prehooks.append(load_layer)
            posthooks.append(load_layer)
        self.prehooks = prehooks
        self.posthooks = posthooks

    def register_outer_hooks(self):
        prehook = self.register_forward_pre_hook(construct_V)
        self.prehook_outer = prehook

    def get_modules(self):
        for module in self.modules():
            if isinstance(module, self.exclude):
                continue

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
                # print( "Parameter unregistered, assigned to type Tensor: {}".format( name + "." + kind))
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
            #self.WandBtoV()
            pass
        return self.module(*args, **kwargs)


def main():
    from feathermap.models.resnet import ResNet, ResidualBlock

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_test():
        lmodel = nn.Linear(2, 4).to(device)
        flmodel = FeatherNet(lmodel, compress=0.5)
        flmodel.WandBtoV()
        print(flmodel.num_WandB(), flmodel.size_n, flmodel.size_m)
        print("V1: {}".format(flmodel.V1))
        print("V2: {}".format(flmodel.V2))
        print("V: {}".format(flmodel.V))
        flmodel.WandBtoV()
        [print(name, v) for name, v in flmodel.named_parameters()]

    def res_test():
        x = torch.randn([1, 3, 32, 32])
        rmodel = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        frmodel = FeatherNet(rmodel, exclude=(nn.BatchNorm2d), compress=0.8).to(device)
        start = timer()
        # 692 s over 100 images
        for _ in range(100):
            frmodel(x)
        end = timer()
        print(end-start)
        exit()

    res_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
