"""Provides compression wrapper for user-defined PyTorch model.
    - Computes weights based on structured multi-hashing
    - Activates forward pre and post hooks for weight layer caching
    - Overloads appropriate nn.Module methods
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Iterator, Tuple, List, Dict, Callable
from math import ceil, sqrt
from feathermap.utils import get_block_rows
import copy
from torch.utils.hooks import RemovableHandle


class LoadLayer:
    """Forward prehook callable for inner layers. Load weights and biases from V1 and V2,
    calculating on the fly. Must be as optimized as possible for low latency."""

    def __init__(
        self,
        name: str,
        module: nn,
        V1: Tensor,
        V2: Tensor,
        size_n: int,
        offset: int,
        verbose: bool = False,
    ):
        self._module = module
        self._name = name
        self._verbose = verbose
        self._V1 = V1
        self._V2 = V2
        self._offset = offset
        self._size_n = size_n
        self._w_size = module.weight.size()
        self._w_num = module.weight.numel()
        self._w_p = module.weight_p  # weight scaler
        self._w_ops = list(
            self._get_operands(
                self._V1, self._V2, *self.__get_index_range(), self._size_n
            ).values()
        )
        self._bias = module.bias is not None
        self._b_ops = None
        if self._bias:
            self._b_size = module.bias.size()
            self._b_num = module.bias.numel()
            self._b_p = module.bias_p  # bias scaler
            self._b_ops = list(
                self._get_operands(
                    self._V1, self._V2, *self.__get_index_range(bias=True), self._size_n
                ).values()
            )

    def __get_index_range(self, bias: bool = False) -> Tuple[int]:
        """Return global weight or bias index range associated with given layer"""
        i1, j1 = divmod(self._offset + 1, self._size_n)
        if bias:
            i2, j2 = divmod(self._offset + self._b_num, self._size_n)
        else:
            i2, j2 = divmod(self._offset + self._w_num, self._size_n)
        return (i1, j1, i2, j2)

    @staticmethod
    def _get_operands(
        V1: Tensor, V2: Tensor, i1: int, j1: int, i2: int, j2: int, n: int
    ) -> dict:
        """Return dictionary of operands representing complete slices of V1.dot(V2) from
        range [i1, j1] to [i2, j2]. Matrix product maps to underlying matrix of size
        (n x n)"""
        ops = {}
        block_rows = get_block_rows(i1, j1, i2, j2, n)
        # Only one row, return whether complete or incomplete
        if i2 - i1 == 0:
            ops["top"] = (V1[i1, :], V2[:, j1 : j2 + 1])
            return ops
        # Has block rows
        if block_rows:
            ops["block"] = (V1[range(*block_rows), :], V2)
            # First row incomplete (from left)
            if i1 < block_rows[0]:
                ops["top"] = (V1[i1, :], V2[:, j1:])
            # Last row incomplete
            if i2 > (block_rows[1] - 1):
                ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
            return ops
        # Two rows, no blocks
        else:
            ops["top"] = (V1[i1, :], V2[:, j1:])
            ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
            return ops

    def __mm_map(self, matrices: List[Tensor]) -> Tensor:
        """Helper function for matrix multiplication, including scale parameter"""
        return self._w_p * torch.matmul(*matrices).view(-1, 1)

    def __call__(self, module: nn.Module, inputs: Tensor):
        if self._verbose:
            print("prehook activated: {} {}".format(self._name, self._module))

        # Load weights
        if len(self._w_ops) == 1:
            module.weight = self.__mm_map(*self._w_ops).reshape(self._w_size)
        else:
            m_products = tuple(map(self.__mm_map, self._w_ops))
            module.weight = torch.cat(m_products).reshape(self._w_size)

        # Load biases
        if self._bias:
            if len(self._b_ops) == 1:
                module.bias = self.__mm_map(*self._b_ops).reshape(self._b_size)
            else:
                m_products = tuple(map(self.__mm_map, self._b_ops))
                module.bias = torch.cat(m_products).reshape(self._b_size)


class UnloadLayer:
    """Forward posthook callable class with verbose switch"""

    verbose = False

    @classmethod
    def __call__(cls, module: nn.Module, inputs: Tensor, outputs: Tensor):
        if UnloadLayer.verbose:
            print("posthook activated: {}".format(module))
        # Unload weights and biases
        module.weight = None
        module.bias = None


class FeatherNet(nn.Module):
    """
    Compresses user-defined PyTorch models based on structured multi-hashing.

    Calculates matrix product V1 * V2 = V, and maps each element of V to global weight
    matrix. The size of V1 and V2 are determined based on compression. See README.md for
    an overview of structured mutli-hashing.
    """


    def __init__(
        self,
        module: nn.Module,
        compress: float = 0.5,
        exclude: tuple = (nn.BatchNorm2d),
        clone: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.module = copy.deepcopy(module) if clone else module
        self._verbose = verbose
        self._exclude = exclude
        self._prehooks = None
        self._posthooks = None
        self._prehook_callables = None
        self._posthook_callable = None

        # Find max compression range
        self._max_compress = self.get_max_compression()
        self.compress = compress

        # Unregister module Parameters, create scaler attributes, set weights
        # as tensors of prior data
        self.__unregister_params()

        self._size_n = ceil(sqrt(self.get_num_WandB()))
        self._size_m = ceil((self.compress * self._size_n) / 2)
        self._V1 = Parameter(torch.Tensor(self._size_n, self._size_m))
        self._V2 = Parameter(torch.Tensor(self._size_m, self._size_n))
        self._V = None

        # Normalize V1 and V2
        self.__norm_V()

    def __register_hooks(self) -> None:
        """Register forward pre and post hooks"""
        prehooks, posthooks, prehook_callables = [], [], []
        offset = -1
        for name, module in self._get_WorB_modules():
            # Create callable prehook object; see LoadLayer; update running V.view(-1, 1) index
            prehook_callable = LoadLayer(
                name, module, self._V1, self._V2, self._size_n, offset, self._verbose
            )
            offset += prehook_callable._w_num
            if getattr(module, "bias", None) is not None:
                offset += prehook_callable._b_num

            # Create callable posthook object
            posthook_callable = UnloadLayer()
            UnloadLayer.verbose = self._verbose

            # Register hooks
            prehook_handle = module.register_forward_pre_hook(prehook_callable)
            posthook_handle = module.register_forward_hook(posthook_callable)

            # Collect removable handles
            prehooks.append(prehook_handle)
            posthooks.append(posthook_handle)
            prehook_callables.append(prehook_callable)

        # Pass handles into attributes
        self._prehooks = prehooks
        self._posthooks = posthooks
        self._prehook_callables = prehook_callables
        self._posthook_callable = posthook_callable

    def __unregister_hooks(self, hooks: RemovableHandle) -> None:
        """Unregister forward pre and post hooks"""
        # Remove hooks
        if hooks is not None:
            for hook in hooks:
                hook.remove()

    def __unregister_params(self) -> None:
        """Delete params, set attributes as Tensors of prior data,
        register new params to scale weights and biases"""
        # fan_in will fail on BatchNorm2d.weight
        for name, module, kind in self._get_WandB_modules():
            try:
                data = module._parameters[kind].data
                if kind == "weight":
                    fan_in = torch.nn.init._calculate_correct_fan(data, "fan_in")
                # get bias fan_in from corresponding weight
                else:
                    fan_in = torch.nn.init._calculate_correct_fan(
                        getattr(module, "weight"), "fan_in"
                    )
                # Delete from parameter list to avoid loading into state dict
                del module._parameters[kind]
                scaler = 1 / sqrt(3 * fan_in)
                setattr(module, kind, data)
                # Add scale parameter to each weight or bias
                module.register_parameter(
                    kind + "_p", Parameter(torch.Tensor([scaler]))
                )
                if self._verbose:
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

    def __map_V_to_WandB(self) -> None:
        """Calculate V = V1*V2 and allocate to all weights and biases"""
        self._V = torch.matmul(self._V1, self._V2)
        V = self._V.view(-1, 1)  # V.is_contiguous() = True
        i = 0
        for name, module, kind in self._get_WandB_modules():
            v = getattr(module, kind)
            j = v.numel()  # elements in weight or bias
            v_new = V[i : i + j].reshape(v.size())  # confirmed contiguous

            # Scaler Parameter, e.g. nn.Linear.weight_p
            scaler = getattr(module, kind + "_p")
            # Update weights and biases, point to elems in V
            setattr(module, kind, scaler * v_new)
            i += j

    def __clear_WandB(self) -> None:
        """Set weights and biases as empty tensors"""
        for name, module, kind in self._get_WandB_modules():
            tensor_size = getattr(module, kind).size()
            setattr(module, kind, torch.empty(tensor_size))

    def __norm_V(self) -> None:
        """Normalize global weight matrix. Currently implemented only for uniform
        intializations"""
        # sigma = M**(-1/4); bound follows from uniform dist.
        bound = sqrt(12) / 2 * (self._size_m ** (-1 / 4))
        torch.nn.init.uniform_(self._V1, -bound, bound)
        torch.nn.init.uniform_(self._V2, -bound, bound)

    def _get_WandB(self) -> Iterator[Tuple[str, Tensor]]:
        """Helper function to return weight AND bias attributes in order"""
        for name, module, kind in self._get_WandB_modules():
            yield name + "." + kind, getattr(module, kind)

    def _get_WandB_modules(self) -> Iterator[Tuple[str, nn.Module, str]]:
        """Helper function to return weight AND bias modules in order.
        Adheres to `self.exclusion` list"""
        for name, module in self.named_modules():
            if isinstance(module, self._exclude):
                continue
            if getattr(module, "weight", None) is not None:
                yield name, module, "weight"
            if getattr(module, "bias", None) is not None:
                yield name, module, "bias"

    def _get_WorB_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        """Helper function to return weight OR bias modules in order
        Adheres to `self.exclusion` list"""
        for name, module in self.named_modules():
            if isinstance(module, (self._exclude, FeatherNet)):
                continue
            if getattr(module, "weight", None) is not None:
                yield name, module

    def get_max_compression(self) -> float:
        """Calculate maximum compression rate based on largest layer size"""
        max_layer_size, _ = self.get_max_num_WandB()
        return max_layer_size / self.get_num_WandB()

    def get_max_num_WandB(self) -> Tuple[int, nn.Module]:
        """Return size of largest layer's weights + biases"""
        w, b, size = 0, 0, 0
        layer = None
        for name, module in self._get_WorB_modules():
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
        return sum(v.numel() for name, v in self._get_WandB())

    def load_state_dict(self, *args, **kwargs) -> Dict:
        """Update weights and biases from stored V1, V2 values"""
        out = nn.Module.load_state_dict(self, *args, *kwargs)
        self.__map_V_to_WandB()
        return out

    def train(self, mode: bool = True) -> None:
        """Remove forward hooks, load weights and biases.
        Note: `self.eval()` calls `self.train(False)`"""
        self.training = mode
        self.__map_V_to_WandB()
        return nn.Module.train(self.module, mode)

    def deploy(self, mode: bool = True) -> None:
        """Whether in train or eval mode, activate deploy mode, i.e. weight layer
        caching. Note: `self.eval()` calls `self.train(False)`"""
        if mode:
            nn.Module.train(self.module, mode=False)
            self.training = False
            # Clear V weight matrix
            self._V = None
            # Add forward hooks
            self.__register_hooks()
            # Clear weights and biases
            self.__clear_WandB()

        # Remove forward hooks
        else:
            self.__unregister_hooks(self.prehooks)
            self.__unregister_hooks(self.posthooks)
            self.__unregister_hooks(self.prehook_outer)
            self.__map_V_to_WandB()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self.__map_V_to_WandB()
        output = self.module(x)
        if self._verbose:
            print("\tIn Model: input size", x.size(), "output size", output.size())
        return output


def tests():
    from feathermap.models.resnet import ResNet34, ResNet18

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def linear_test():
        lmodel = nn.Linear(2, 4).to(device)
        flmodel = FeatherNet(lmodel, compress=0.5)
        flmodel.__VtoWandB()
        print(flmodel.get_num_WandB(), flmodel._size_n, flmodel._size_m)
        print("V: {}".format(flmodel.V))
        flmodel.__WandBtoV()
        [print(name, v) for name, v in flmodel.named_parameters()]

    def res_test():
        def pic_gen():
            for i in range(2):
                yield torch.randn([1, 3, 32, 32])
        base_model = ResNet34().to(device)
        model = FeatherNet(base_model, compress=0.5, verbose=True).to(device)
        for name, module, kind in model._get_WandB_modules():
            p = getattr(module, kind)
            print(name, kind, p.size(), p.numel())
        with torch.no_grad():
            for x in pic_gen():
                model(x)
        model.deploy()
        with torch.no_grad():
            for x in pic_gen():
                model(x)

    res_test()


if __name__ == "__main__":
    try:
        tests()
    except KeyboardInterrupt:
        exit()
