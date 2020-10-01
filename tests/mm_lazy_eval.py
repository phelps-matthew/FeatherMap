import torch
from feathermap.utils import timed


dim_in = 2**10
dim_out = 2**4
A = torch.randn(dim_in, dim_out)
B = torch.randn(dim_out, dim_in)


@timed
def loop(a, b):
    for i in range(a.size(0)):
        for j in range(b.size(1)):
            yield A[i, :] @ B[:, j]


@timed
def tmm(a, b):
    c = torch.mm(a, b).view(-1, 1)
    yield iter(c)


@timed
def run(c, dim_in):
    c = torch.Tensor(dim_in**2)
    for i, v in enumerate(c):
        c[i] = v


run(loop(A, B), dim_in)
run(tmm(A, B), dim_in)
# loop is approximately < 3% slower than tmm
