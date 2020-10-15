import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator, Tuple
from math import ceil, sqrt
import copy


class LoadLayer:
    """Forward prehook for inner layers. Load weights and biases from V1 and V2,
    calculating on the fly. Must be as optimized as possible."""

    def __init__(self, name, module, V1, V2, size_n, offset, verbose=False):
        self.module = module
        self.name = name
        self.verbose = verbose
        self.V1 = V1
        self.V2 = V2
        self.offset = offset
        self.size_n = size_n
        self.w_size = module.weight.size()
        self.w_num = module.weight.numel()
        self.w_p = module.weight_p  # weight scaler
        self.bias = module.bias is not None
        if self.bias:
            self.b_size = module.bias.size()
            self.b_num = module.bias.numel()
            self.b_p = module.bias_p  # bias scaler

        self.w_ops = None
        self.b_ops = None
        self.set_operands()

    def get_weight_index_range(self):
        """Return global weight index range associated with given layer"""
        i1, j1 = divmod(self.offset + 1, self.size_n)
        i2, j2 = divmod(self.offset + self.w_num, self.size_n)
        #self.offset += self.w_num
        print(self, [i1, j1], [i2, j2])
        return (i1, j1, i2, j2)

    def get_bias_index_range(self):
        """Return global weight index range associated with given layer"""
        i1, j1 = divmod(self.offset + 1, self.size_n)
        i2, j2 = divmod(self.offset + self.b_num, self.size_n)
        self.offset += self.b_num
        #print(self, [i1, j1], [i2, j2])
        return (i1, j1, i2, j2)

    @staticmethod
    def get_block_rows(i: int, j: int, numels: int, n: int) -> list:
        """Get list of full rows in an (n x n) matrix starting at index [i,j]
        and spanning numels elements. Returns row numbers"""
        print("get start block rows i,j n numels {} {} {} {}".format(i,j,n,numels))
        j_start = (j == 0)
        rows = []
        for _ in range(numels):
            if j > n:
                if j_start:
                    rows.append(i)
                i += 1
                j = 0
                j_start = True
            j += 1
        print("get block rows i,j n {} {} {}".format(i,j,n))
        return rows

    @staticmethod
    def get_row_set(V1, V2, i1, j1, i2, j2, numels, n):
        """Generate the set of operands corresponding to partial or full rows. E.g.

        | _ x x x |            | x x x |
        | x x x x |  ------>             + | x x x x |
        | x x x x |                        | x x x x | +
        | x x _ _ |                                      | x x |

        Necessary to make the most use of vectorized matrix multiplication. Sequentially
        calculating V[i, j] = V1[i, :] @ V2[:, j] leads to large latency.
        """
        ops = dict()
        block_rows = LoadLayer.get_block_rows(i1, j1, numels, n)
        # Only one row, return whether complete or incomplete
        if i2 - i1 == 0:
            ops["top"] = (V1[i1, :], V2[:, j1 : j2 + 1])
            return ops
        # Has block rows
        if len(block_rows) != 0:
            ops["block"] = (V1[block_rows, :], V2)
            for row in range(i1, i2 + 1):
                if row not in block_rows:
                    if row < min(block_rows):
                        ops["top"] = (V1[row, :], V2[:, j1:])
                    else:
                        ops["bottom"] = (V1[row, :], V2[:, : j2 + 1])
            return ops
        # Two rows, no blocks
        else:
            ops["top"] = (V1[i1, :], V2[:, j1:])
            ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
            return ops

    def set_operands(self):
        """Set weight and bias attributes corresponding to operands of V1 and V2"""
        w_ops_dict = self.get_row_set(
            self.V1, self.V2, *self.get_weight_index_range(), self.w_num, self.size_n
        )
        self.w_ops = [w_ops_dict[k] for k in w_ops_dict]
        if self.bias:
            b_ops_dict = self.get_row_set(
                self.V1, self.V2, *self.get_bias_index_range(), self.b_num, self.size_n
            )
            self.b_ops = [b_ops_dict[k] for k in b_ops_dict]

    def mm_map(self, a):
        """Helper function for matrix multiplication, including scale parameter"""
        return self.w_p * torch.matmul(*a).view(-1, 1)

    def __call__(self, module, inputs):
        if self.verbose:
            print("prehook activated: {} {}".format(self.name, self.module))

        # Load weights
        if len(self.w_ops) == 1:
            a = self.mm_map(*self.w_ops)
            print(self.w_num)
            print(a.numel())
            w_ops_dict = self.get_row_set(
                self.V1, self.V2, *self.get_weight_index_range(), self.w_num, self.size_n
            )
            print(w_ops_dict.keys())
            module.weight = self.mm_map(*self.w_ops).reshape(self.w_size)
        else:
            a = tuple(map(self.mm_map, self.w_ops))
            print(self.w_num)
            print(torch.cat(a).numel())
            w_ops_dict = self.get_row_set(
                self.V1, self.V2, *self.get_weight_index_range(), self.w_num, self.size_n
            )
            print(w_ops_dict.keys())
            module.weight = torch.cat(a).reshape(self.w_size)

        # Load biases
        if self.bias:
            if len(self.b_ops) == 1:
                module.bias = self.mm_map(*self.b_ops).reshape(self.b_size)
            else:
                a = tuple(map(self.mm_map, self.b_ops))
                module.bias = torch.cat(a).reshape(self.b_size)


class UnloadLayer:
    """Forward posthook callable class with verbose switch"""

    verbose = False

    @classmethod
    def __call__(cls, module, inputs, outputs):
        if UnloadLayer.verbose:
            print("posthook activated: {}".format(module))
        module.weight = None
        module.bias = None


def unload_layer(module, inputs, outputs):
    """Forward hook for inner layers. Unloads weights and biases for given layer"""
    # print("posthook activated: {}".format(module))
    module.weight = None
    module.bias = None


class FeatherNet(nn.Module):
    """Implementation of "Structured Multi-Hashing for Model Compression"""

    def __init__(
        self,
        module: nn.Module,
        compress: float = 1,
        exclude: tuple = (),
        clone: bool = True,
        verbose: bool = False,
        constrain: bool = False,
    ) -> None:
        super().__init__()
        self.module = copy.deepcopy(module) if clone else module
        self.verbose = verbose
        self.constrain = constrain
        self.exclude = exclude
        self.prehooks = None
        self.posthooks = None
        self.prehook_callables = None
        self.posthook_callable = None

        # Check compression range
        self.max_compress = self.get_max_compression()
        if compress < self.max_compress and self.constrain:
            print(
                (
                    "Due to streaming layer weight allocation, cannot compress beyond {:.4f}."
                    + "Setting compression to {:.4f}"
                ).format(self.max_compress, self.max_compress)
            )
            self.compress = self.max_compress
        else:
            self.compress = compress

        # Unregister module Parameters, create scaler attributes, set weights
        # as tensors of prior data
        self.unregister_params()

        self.size_n = ceil(sqrt(self.get_num_WandB()))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.Tensor(self.size_n, self.size_m))
        self.V2 = Parameter(torch.Tensor(self.size_m, self.size_n))
        self.V = None

        # Normalize V1 and V2
        self.norm_V()

    def norm_V(self):
        """Currently implemented only for uniform intializations"""
        # sigma = M**(-1/4); bound follows from uniform dist.
        bound = sqrt(12) / 2 * (self.size_m ** (-1 / 4))
        torch.nn.init.uniform_(self.V1, -bound, bound)
        torch.nn.init.uniform_(self.V2, -bound, bound)

    def WandBtoV(self):
        """Calculate V = V1*V2 and allocate to all weights and biases"""
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

    def clear_WandB(self):
        """Set weights and biases as empty tensors"""
        for name, module, kind in self.get_WandB_modules():
            tensor_size = getattr(module, kind).size()
            setattr(module, kind, torch.empty(tensor_size))

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
            except AttributeError:
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
            except AttributeError:
                pass

    def register_hooks(self):
        prehooks, posthooks, prehook_callables = [], [], []
        offset = -1
        for name, module in self.get_WorB_modules():
            # Create callable prehook object; see LoadLayer; update running V.view(-1, 1) index
            prehook_callable = LoadLayer(
                name, module, self.V1, self.V2, self.size_n, offset, self.verbose
            )
            offset += prehook_callable.w_num
            if getattr(module, "bias", None) is not None:
                offset += prehook_callable.b_num

            # Create callable posthook object
            posthook_callable = UnloadLayer()
            posthook_callable.verbose = self.verbose

            # Register hooks
            prehook_handle = module.register_forward_pre_hook(prehook_callable)
            posthook_handle = module.register_forward_hook(posthook_callable)

            # Collect removable handles
            prehooks.append(prehook_handle)
            posthooks.append(posthook_handle)
            prehook_callables.append(prehook_callable)

        # Pass handles into attributes
        self.prehooks = prehooks
        self.posthooks = posthooks
        self.prehook_callables = prehook_callables
        self.posthook_callable = posthook_callable

    def unregister_hooks(self, hooks):
        # Remove hooks
        if hooks is not None:
            for hook in hooks:
                hook.remove()

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
                if self.verbose:
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

    def load_state_dict(self, *args, **kwargs):
        """Update weights and biases from stored V1, V2 values"""
        out = nn.Module.load_state_dict(self, *args, *kwargs)
        self.WandBtoV()
        return out

    def train(self, mode: bool = True):
        """Remove forward hooks, load weights and biases.
        `self.eval()` calls `self.train(False)`"""
        self.training = mode
        self.WandBtoV()
        return nn.Module.train(self.module, mode)

    def deploy(self, mode: bool = True):
        """Whether in train or eval mode, activate deploy mode
        `self.eval()` calls `self.train(False)`"""
        if mode:
            nn.Module.train(self.module, mode=False)
            self.training = False
            # Clear V weight matrix
            self.V = None
            # Add forward hooks
            self.register_hooks()
            # Clear weights and biases
            self.clear_WandB()

        # Remove forward hooks
        else:
            self.unregister_hooks(self.prehooks)
            self.unregister_hooks(self.posthooks)
            self.unregister_hooks(self.prehook_outer)
            self.WandBtoV()

    def forward(self, x):
        if self.training:
            self.WandBtoV()
        output = self.module(x)
        if self.verbose:
            print("\tIn Model: input size", x.size(), "output size", output.size())
        return output


def tests():
    from feathermap.models.resnet import ResNet34, ResNet18

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_test():
        lmodel = nn.Linear(2, 4).to(device)
        flmodel = FeatherNet(lmodel, compress=0.5)
        flmodel.WandBtoV()
        print(flmodel.get_num_WandB(), flmodel.size_n, flmodel.size_m)
        print("V: {}".format(flmodel.V))
        flmodel.WandBtoV()
        [print(name, v) for name, v in flmodel.named_parameters()]

    def res_test():
        def pic_gen():
            for i in range(100):
                yield torch.randn([1, 3, 32, 32])

        base_model = ResNet18().to(device)
        model = FeatherNet(base_model, exclude=(nn.BatchNorm2d), compress=0.5, verbose=True).to(device)
        for name, module, kind in model.get_WandB_modules():
            p = getattr(module, kind)
            #print(name, kind, p.size(), p.numel())
        model.deploy()
        for prehook in model.prehook_callables:
            #print(prehook.w_num)
            ...
        print("Size n: {}".format(model.size_n))
        with torch.no_grad():
            for x in pic_gen():
                model(x)

    res_test()


if __name__ == "__main__":
    try:
        tests()
    except KeyboardInterrupt:
        exit()
