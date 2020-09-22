import torch
import torch.nn as nn
from torch.nn import Parameter
from src.resnet import ResNet, ResidualBlock
from torch import Tensor, device, dtype
from collections import OrderedDict, namedtuple
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

# parameters(named_parameters(_named_members(named_modules(named_modules))))

# Device configuration
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(ResidualBlock, [2, 2, 2]).to(my_device)

test_params = list(model.parameters())
test_named_params = dict(model.named_parameters())

print(*test_named_params.keys(), sep="\n")


# ---------------------------------------------------------------------------- #
# PyTorch methods                                                              #
# ---------------------------------------------------------------------------- #


def named_modules(self, memo: Optional[Set["Module"]] = None, prefix: str = ""):
    r"""Returns an iterator over all modules in the network, yielding
    both the name of the module as well as the module itself.
    Yields:
        (string, Module): Tuple of name and module
    Note:
        Duplicate modules are returned only once. In the following
        example, ``l`` will be returned only once.
    Example::
        >>> l = nn.Linear(2, 2)
        >>> net = nn.Sequential(l, l)
        >>> for idx, m in enumerate(net.named_modules()):
                print(idx, '->', m)
        0 -> ('', Sequential(
          (0): Linear(in_features=2, out_features=2, bias=True)
          (1): Linear(in_features=2, out_features=2, bias=True)
        ))
        1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
    """

    if memo is None:
        memo = set()
    if self not in memo:
        memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            for m in module.named_modules(memo, submodule_prefix):
                yield m


def named_parameters(
    self, prefix: str = "", recurse: bool = True
) -> Iterator[Tuple[str, Tensor]]:
    r"""Returns an iterator over module parameters, yielding both the
    name of the parameter as well as the parameter itself.
    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.
    Yields:
        (string, Parameter): Tuple containing the name and parameter
    Example::
        >>> for name, param in self.named_parameters():
        >>>    if name in ['bias']:
        >>>        print(param.size())
    """
    gen = self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix,
        recurse=recurse,
    )
    for elem in gen:
        yield elem


def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    r"""Returns an iterator over module parameters.
    This is typically passed to an optimizer.
    Args:
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.
    Yields:
        Parameter: module parameter
    Example::
        >>> for param in model.parameters():
        >>>     print(type(param), param.size())
        <class 'torch.Tensor'> (20L,)
        <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
    """
    for name, param in self.named_parameters(recurse=recurse):
        yield param


def _named_members(self, get_members_fn, prefix="", recurse=True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v
