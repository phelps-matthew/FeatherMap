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

    def get_weights(self, exclude: tuple = ()):
        for name, m in self.named_modules():
            try:
                if isinstance(m, exclude):
                    continue
                yield name, getattr(m, 'weight')
            except nn.modules.module.ModuleAttributeError:
                continue

    def get_bias(self, exclude: tuple = ()):
        for name, m in self.named_modules():
            try:
                if isinstance(m, exclude):
                    continue
                yield name, getattr(m, 'bias')
            except nn.modules.module.ModuleAttributeError:
                continue

    def del_params(self, exclude: tuple = ()):
        for name, m in self.named_modules():
            try:
                if isinstance(m, exclude):
                    continue
                del m._parameters["weight"]
                print("Deleting parameter: {}".format(name + ".weight"))
                setattr(m, "weight", Tensor([[1.0]]))
                print("Intialized to type Tensor: {}".format(name + ".weight"))
            except KeyError:
                continue
                # print("key error: {}".format(name))

    def parameter_pop(self, exclude: tuple = ()) -> str:
        """Delete weight or bias attribute and return attribute handle"""
        attrs = [
            ["self."] + module_attribute.rsplit(".", 1)
            for module_attribute, _ in self.named_parameters(exclude=exclude)
        ]
        for a in attrs:
            print(a)
            obj = eval("".join(a[:-1]))
            delattr(obj, a[-1])

    def parameters(
        self, recurse: bool = True, exclude: tuple = ()
    ) -> Iterator[Parameter]:
        for name, param in self.named_parameters(
            exclude=exclude, recurse=recurse
        ):
            yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        exclude: tuple = (),
    ) -> Iterator[Tuple[str, Tensor]]:
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
    # [print(name) for name, v in f_model.named_parameters()]
    # print(f_model.parameter_pop())
    # [print(name) for name, v in f_model.named_parameters()]
    # print(*list(filter(lambda x: not(isinstance(x, nn.BatchNorm2d)), f_model.modules())),sep='\n')
    [print(name) for name, v in f_model.named_parameters()]
    print("-" * 20)
    #f_model.del_params()
    #[print(name + '.weight') for name, m, in f_model.get_weights(exclude=(nn.BatchNorm2d))]
    #[print(name + '.bias') for name, m, in f_model.get_bias()]
    for name, weight in f_model.get_weights():
        print(name + '.weight')
        weight = Tensor([[1.0]])
    for name, weight in f_model.get_weights():
        print(weight)

    exit()

    # print(*dict(model.named_parameters()).keys(), sep="\n")
    # print(*dict(f_model.named_parameters()).keys(), sep="\n")

    del model.fc.weight


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
