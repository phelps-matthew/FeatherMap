import torch
from feathermap.utils import timed
from math import sqrt


dim_in = 2 ** 14
dim_out = 2 ** 4
A = torch.randn(dim_in, dim_out)
B = torch.randn(dim_out, dim_in)
C = torch.rand(dim_in, dim_in)
D = torch.rand(dim_in, dim_in)
E = torch.rand(1, dim_out)
F = torch.rand(dim_out, dim_in)
G = torch.rand(int(sqrt(dim_in)), int(sqrt(dim_in)))
H = torch.rand(int(sqrt(dim_in)), int(sqrt(dim_in)))


@timed
def mam(a, b):
    for _ in range(10000):
        out = torch.mm(a, b)
    return out


def loop(a, b):
    for i in range(a.size(0)):
        for j in range(b.size(1)):
            yield a[i, :] @ b[:, j]


def loop2(a, b):
    for i in range(a.size(0)):
        for j in range(b.size(1)):
            yield 1


def tmm(a, b):
    c = torch.mm(a, b).view(-1, 1)
    return iter(c)


@timed
def run(c, dim_in):
    d = torch.empty(dim_in ** 2)
    for i in range(d.numel()):
        d[i] = next(c)


mam(E, F) # about 23% faster
mam(G, H)


# run(loop(A, B), dim_in)  # 739
# run(loop2(A, B), dim_in)  # 254
# run(tmm(A, B), dim_in)  # 289
