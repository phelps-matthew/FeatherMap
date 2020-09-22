"""
Implementation testing for hashable parameters function.
"""
# Many imports derived from PyTorch source
import torch
from torch.nn.modules import Module
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


# Device configuration
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(ResidualBlock, [2, 2, 2]).to(my_device)

# print(type(model.parameters()))
# print(model.parameters().__next__())
# print(next(model.named_parameters()))

test_params = list(model.parameters())
test_named_params = dict(model.named_parameters())

# print(*test_named_params.keys(), sep="\n")
# print(*model.named_modules(), sep='\n')
l = nn.Linear(2, 2)
net1 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net = nn.Sequential(l, l, net1)
# print(*dict(net.parameters()),sep='\n')

for name, module in model.named_modules():
    # print(name, type(module))
    continue

# print(l._parameters.items())


def _named_members_subset(
    self,
    get_members_fn,
    prefix="",
    exclude: tuple = (),
    recurse=True,
):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        if isinstance(module, exclude):
            continue
        # members from _parameters.items() are odict_items (possibly empty)
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v


def named_parameters_subset(
    self,
    prefix: str = "",
    exclude: tuple = (nn.BatchNorm2d),
    recurse: bool = True,
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
    gen = _named_members_subset(
        self,
        lambda module: module._parameters.items(),
        prefix=prefix,
        exclude=exclude,
        recurse=recurse,
    )
    for elem in gen:
        yield elem


def parameters_subset(
    self, exclude: tuple = (nn.BatchNorm2d), recurse: bool = True
) -> Iterator[Parameter]:
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
    for name, param in self.named_parameters_subset(
        exclude=exclude, recurse=recurse
    ):
        yield param


[
    print(name)
    for name, v in my_named_parameters(model, exclude=(nn.BatchNorm2d))
]
print("-" * 20)
print(*dict(model.named_parameters()).keys(), sep="\n")


# ---------------------------------------------------------------------------- #
# PyTorch methods                                                              #
# ---------------------------------------------------------------------------- #

# parameters(named_parameters(_named_members(named_modules(named_modules))))
# net._parameters only exists for explicitly defined layers
# (i.e. empty for nn.Sequential or nn.ResNet)
# for a given module, seeks __getattr__;


def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    for name, param in self.named_parameters(recurse=recurse):
        yield param


def named_parameters(
    self, prefix: str = "", recurse: bool = True
) -> Iterator[Tuple[str, Tensor]]:
    r"""Returns an iterator over module parameters, yielding both the
    name of the parameter as well as the parameter itself.
    Args:
        prefix (str): prefix to prepend to all parameter names.
    """
    gen = self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix,
        recurse=recurse,
    )
    for elem in gen:
        yield elem


def _named_members(self, get_members_fn, prefix="", recurse=True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    # iterator
    modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    for module_prefix, module in modules:
        # members from _parameters.items() are odict_items (possibly empty)
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v


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
    # non-duplication should not be issue; weights are same in ex.

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
