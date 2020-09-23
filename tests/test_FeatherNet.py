import torch
import torch.nn as nn
from torch.nn import Parameter
from src.resnet import ResNet, ResidualBlock
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


class FeatherNet(nn.Module):
    """Overrides parameters() method."""

    def __init__(self, module: nn.Module, compress: float = 1) -> None:
        super().__init__()
        self.module = module

    def _named_members(
        self,
        get_members_fn,
        prefix="",
        recurse=True,
        exclude: tuple = (),
    ):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        )
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

    def parameters(
        self, recurse: bool = True, exclude: tuple = (nn.BatchNorm2d)
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
        for name, param in self.named_parameters(
            exclude=exclude, recurse=recurse
        ):
            yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        exclude: tuple = (nn.BatchNorm2d),
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
            exclude=exclude,
        )
        for elem in gen:
            yield elem

    def hparams_to_W(self) -> Tensor:
        """Collect set of hashable params into n x n tensor"""
        pass

    def mhash(self, W: Tensor, compress: float) -> Tensor:
        """Create hash tensor (matrix product multi-hash)"""
        # dim_W = torch.size(W)[0]
        # target_size = dim_W*compress
        # M = Parameter(torch.Tensor(target_size, target_size))
        # MT = Parameter(torch.transpose(M))
        pass

    def reassign_hparams(self, W, M):
        """Deparameterize hash parameters, points towards hash tensor"""
        pass


def main():
    # parameters(named_parameters(_named_members(named_modules(named_modules))))

    # Device configuration

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(ResidualBlock, [2, 2, 2]).to(my_device)
    f_model = FeatherNet(model)

    # print(*dict(model.named_parameters()).keys(), sep="\n")
    # print(*dict(f_model.named_parameters()).keys(), sep="\n")
    print(*model.fc.named_parameters())
    del model.fc.weight

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
