import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import Parameter


class hashNet(nn.Module):
    """my class"""

    def __init__(self, module: nn.Module, compress: float) -> None:
        super().__init__()
        self.module = module
        pass

    def hparams(self, a):
        """Return generator of module params that are to be hashed.
        Ignore BatchNorm2d params"""
        a = [2, 2]
        return a

    def hparams_to_W(self) -> torch.tensor:
        """Collect set of hashable params into n x n tensor"""
        pass

    def mhash(self, W: torch.Tensor, compress: float) -> torch.Tensor:
        """Create hash tensor (matrix product multi-hash)"""
        # dim_W = torch.size(W)[0]
        # target_size = dim_W*compress
        # M = Parameter(torch.Tensor(target_size, target_size))
        # MT = Parameter(torch.transpose(M))

    def reassign_hparams(self, W, M):
        """Deparameterize hash parameters, points towards hash tensor"""
