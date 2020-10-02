import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator, Tuple
from math import ceil, sqrt
import copy
from timeit import default_timer as timer


class LoadLayer:
    """Forward prehook for inner layers. Load weights and biases from V, calculating on
    the fly. Must be as fast as possible"""

    def __init__(self, name, module, V_iter):
        self.module = module
        self.name = name
        self.V_iter = V_iter
        self.w_size = module.weight.size()
        self.w_num = module.weight.numel()
        self.w_p = module.weight_p  # weight scaler
        self.bias = module.bias is not None
        if self.bias:
            self.b_size = module.bias.size()
            self.b_num = module.bias.numel()
            self.b_p = module.bias_p  # bias scaler

    def __call__(self, module, inputs):
        # print("prehook activated: {} {}".format(self.name, self.module))
        w = torch.empty(self.w_num)
        V = self.V_iter[0]
        for i in range(self.w_num):
            w[i] = self.w_p * next(V)
        module.weight = w.reshape(self.w_size)
        if self.bias:
            b = torch.empty(self.b_num)
            for i in range(self.b_num):
                b[i] = self.b_p * next(V)
            module.bias = b.reshape(self.b_size)


def unload_layer(module, inputs, outputs):
    """Forward hook for inner layers. Unloads weights and biases for given layer"""
    module.weight = None
    module.bias = None


def reset_V_generator(module, inputs):
    """Forward prehook for outermost FeatherNet layer"""
    module.V_iter[0] = module.V_generator()


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
        self.exclude = exclude
        self.prehooks = None
        self.posthooks = None
        self.prehook_outer = None
        self.prehook_callables = None

        # Check compression range
        self.max_compress = self.get_max_compression()
        if compress < self.max_compress:
            print(
                (
                    "Due to streaming layer weight allocation, cannot compress beyond {:.4f}."
                    + "Setting compression to {:.4f}"
                ).format(self.max_compress, self.max_compress)
            )
            self.compress = self.max_compress
        else:
            self.compress = compress

        # Unregister module Parameters, create scaler attributes
        self.unregister_params()

        self.size_n = ceil(sqrt(self.get_num_WandB()))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.Tensor(self.size_n, self.size_m))
        self.V2 = Parameter(torch.Tensor(self.size_m, self.size_n))
        self.V = None

        # Normalize V1 and V2
        self.norm_V()

        # Use list as pointer
        self.V_iter = [self.V_generator()]

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

    def get_max_compression(self):
        """Calculate maximum compression rate based on largest layer size"""
        max_layer_size, _ = self.get_max_num_WandB()
        return max_layer_size / self.get_num_WandB()

    def get_max_num_WandB(self):
        """Return size of largest layer's weights + biases"""
        w, b, size = 0, 0, 0
        layer = None
        for name, module in self.get_WorB_modules():
            if module.bias is not None:
                b = module.bias.numel()
            else:
                b = 0
            if module.weight is not None:
                w = module.weight.numel()
            else:
                w = 0
            if w + b > size:
                size = w + b
                layer = module
        return size, layer

    def get_num_WandB(self) -> int:
        """Return total number of weights and biases"""
        return sum(v.numel() for name, v in self.get_WandB())

    def get_WandB(self) -> Iterator[Tuple[str, Tensor]]:
        for name, module, kind in self.get_WandB_modules():
            yield name + "." + kind, getattr(module, kind)

    def get_WandB_modules(self) -> Iterator[Tuple[str, nn.Module, str]]:
        """Helper function to return weight and bias modules in order.
        Adheres to `self.exclusion` list"""
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

    def get_WorB_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Helper function to return weight or bias modules in order
        Adheres to `self.exclusion` list"""
        for name, module in self.named_modules():
            try:
                if isinstance(module, (self.exclude, FeatherNet)):
                    continue
                if getattr(module, "weight") is not None:
                    yield name, module
            except nn.modules.module.ModuleAttributeError:
                pass

    def V_generator(self):
        for i in range(self.size_n):
            for j in range(self.size_n):
                yield torch.dot(self.V1[i, :], self.V2[:, j])

    def register_inter_hooks(self):
        prehooks, posthooks, prehook_callables = [], [], []
        for name, module in self.get_WorB_modules():
            prehook_callable = LoadLayer(name, module, self.V_iter)
            prehook_handle = module.register_forward_pre_hook(prehook_callable)
            posthook_handle = module.register_forward_hook(unload_layer)
            prehooks.append(prehook_handle)
            posthooks.append(posthook_handle)
            prehook_callables.append(prehook_callable)
        self.prehooks = prehooks
        self.posthooks = posthooks
        self.prehook_callables = prehook_callables

    def register_outer_hooks(self):
        prehook_handle = self.register_forward_pre_hook(reset_V_generator)
        self.prehook_outer = [prehook_handle]

    def unregister_hooks(self, hooks):
        # Remove hooks
        if hooks is not None:
            for hook in hooks:
                hook.remove()
        # Set weights and biases to empty (non None) tensors; necessary for training mode
        if hooks is self.prehooks:
            for layer_obj in self.prehook_callables:
                layer_obj.module.weight = torch.empty(layer_obj.w_size)
                if layer_obj.bias:
                    layer_obj.module.bias = torch.empty(layer_obj.b_size)

    def unregister_params(self) -> None:
        """Delete params, set attributes as Tensors of prior data,
        register new params to scale weights and biases"""
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

    def train(self, mode: bool = True):
        """Remove forward hooks, load weights and biases.
        `self.eval()` calls self.train(False)"""
        # Remove forward hooks
        if mode:
            self.unregister_hooks(self.prehooks)
            self.unregister_hooks(self.posthooks)
            self.unregister_hooks(self.prehook_outer)
            self.WandBtoV()
        # eval mode
        else:
            # Clear V weight matrix
            self.V = None
            # Add forward hooks
            self.register_inter_hooks()
            self.register_outer_hooks()
        return nn.Module.train(self, mode)

    def forward(self, *args, **kwargs):
        if self.training:
            self.WandBtoV()
        return self.module(*args, **kwargs)


def main():
    from feathermap.models.resnet import ResNet, ResidualBlock

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_test():
        lmodel = nn.Linear(2, 4).to(device)
        flmodel = FeatherNet(lmodel, compress=0.5)
        flmodel.WandBtoV()
        print(flmodel.get_num_WandB(), flmodel.size_n, flmodel.size_m)
        print("V1: {}".format(flmodel.V1))
        print("V2: {}".format(flmodel.V2))
        print("V: {}".format(flmodel.V))
        flmodel.WandBtoV()
        [print(name, v) for name, v in flmodel.named_parameters()]

    def res_test():
        x = torch.randn([1, 3, 32, 32])
        x_20 = torch.randn([100, 3, 32, 32])
        rmodel = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        frmodel = FeatherNet(rmodel, exclude=(nn.BatchNorm2d), compress=0.1).to(device)
        start = timer()
        frmodel.eval()
        with torch.no_grad():
            for i in range(10):
                frmodel(x)
        end = timer()
        print(end - start)
        start = timer()
        for i in range(10):
            frmodel(x)
        end = timer()
        print(end - start)
        start = timer()
        frmodel.train()
        #print(*list(frmodel.get_WandB_modules()))

        with torch.no_grad():
            for i in range(10):
                frmodel(x)
        end = timer()
        print(end - start)

    res_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
