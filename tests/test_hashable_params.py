import torch
import torch.nn as nn
from torch.nn import Parameter
from src.resnet import ResNet, ResidualBlock

# parameters(named_parameters(_named_members(named_modules(named_modules))))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

test_params = list(model.parameters())
test_named_params = list(model.named_parameters())

print(test_params)


class FeatherNet(nn.Module):
    """my class"""

    def __init__(self, module: nn.Module, compress: float) -> None:
        super().__init__()
        self.module = module
        pass

    def hashable_params(self, a):
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
