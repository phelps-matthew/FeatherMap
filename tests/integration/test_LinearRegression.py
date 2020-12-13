import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from math import ceil, sqrt
from torch.nn import Parameter
import torch.nn.functional as F


class LinReg(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = Parameter(torch.Tensor(output_size, input_size))
        self.bias = Parameter(torch.Tensor(output_size))

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return out


class LinRegHash(nn.Module):
    def __init__(self, input_size, output_size, compress=1.0):
        super().__init__()
        self.weight = torch.Tensor(output_size, input_size)
        self.bias = torch.Tensor(output_size)
        self.compress = compress
        self.size_n = ceil(sqrt(input_size * output_size + output_size))
        self.size_m = ceil((self.compress * self.size_n) / 2)
        self.V1 = Parameter(torch.Tensor(self.size_n, self.size_m))
        self.V2 = Parameter(torch.Tensor(self.size_m, self.size_n))

        self.norm_V()

    def norm_V(self):
        k = sqrt(12) / 2 * self.size_m ** (-1 / 4)
        torch.nn.init.uniform_(self.V1, -k, k)
        torch.nn.init.uniform_(self.V2, -k, k)

    def WtoV(self):
        self.V = torch.matmul(self.V1, self.V2)
        V = self.V.view(-1, 1)
        i = 0
        for kind in ("weight", "bias"):
            v = getattr(self, kind)
            j = v.numel()
            w = V[i : i + j].reshape(v.size())
            setattr(self, kind, w)
            i += j
        # i = 0
        # V = self.V.view(-1, 1)
        # v = self.weight.view(-1, 1)
        # for j in range(len(v)):
        #    v[j] = V[i]
        #    i += 1

    def forward(self, x):
        self.WtoV()
        out = F.linear(x, self.weight, self.bias)
        return out


def train(model, criterion, optimizer, x, y):
    # Forward
    loss = criterion(model(x), y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    torch.manual_seed(42)
    X = torch.Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    Y = 3.0 * X + torch.randn(X.size()) * 0.33

    model = LinRegHash(3, 3, compress=0.1)
    # model = LinReg(3, 3)
    loss = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch_size = 1
    epochs = 20

    for i in range(epochs):
        cost = 0.0
        num_batches = len(X) // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, X[start:end], Y[start:end])
        print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))

    print(list(model.named_parameters()))


if __name__ == "__main__":
    main()
