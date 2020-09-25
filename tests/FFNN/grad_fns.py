import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

V1 = Parameter(torch.randn(3, 3, requires_grad=True))
V2 = Parameter(torch.randn(3, 3, requires_grad=True))
W = torch.randn(2, 2)
bias = torch.zeros(2)


def update(V, W):
    V = torch.matmul(V1, V2.transpose(0, 1))
    i = 0
    V = V.view(-1, 1)
    W = W.view(-1, 1)
    for j in range(len(W)):
        W[j] = V[i]
        i += 1


def forward(x, W, bias):
    return F.linear(x, W, bias)


#print("V {}".format(V))
#print("W {}".format(W))
update(V, W)
#print("V {}".format(V))
#print("W {}".format(W))


x = torch.randn(2)
g = torch.ones(2)
#print(x)
#print(forward(x, W, bias).norm)
y = forward(x, W, bias)
print(y)
print(y.reshape(-1,1))
#loss_fn = F.cross_entropy(y.reshape(1, -1), torch.ones(1, 2))
# print(loss_fn)

forward(x, W, bias).backward(g)
