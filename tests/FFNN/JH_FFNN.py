from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math
from IPython.core.debugger import set_trace

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding="latin-1"
    )

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
exit()

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)


bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll


yb = y_train[0:bs]
print(loss_func(preds, yb))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


print(accuracy(preds, yb))


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

###############################################################################
# That's it: we've created and trained a minimal neural network (in this case, a
# logistic regression, since we have no hidden layers) entirely from scratch!
#
# Let's check the loss and accuracy and compare those to what we got

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# Using torch.nn.functional
# ------------------------------
#
# We will now refactor our code, so that it does the same thing as before, only
# we'll start taking advantage of PyTorch's ``nn`` classes to make it more concise
# and flexible. At each step from here, we should be making our code one or more
# of: shorter, more understandable, and/or more flexible.
#
# The first and easiest step is to make our code shorter by replacing our
# hand-written activation and loss functions with those from ``torch.nn.functional``
# (which is generally imported into the namespace ``F`` by convention). This module
# contains all the functions in the ``torch.nn`` library (whereas other parts of the
# library contain classes). As well as a wide range of loss and activation
# functions, you'll also find here some convenient functions for creating neural
# nets, such as pooling functions. (There are also functions for doing convolutions,
# linear layers, etc, but as we'll see, these are usually better handled using
# other parts of the library.)
#
# If you're using negative log likelihood loss and log softmax activation,
# then Pytorch provides a single function ``F.cross_entropy`` that combines
# the two. So we can even remove the activation function from our model.

import torch.nn.functional as F

loss_func = F.cross_entropy


def model(xb):
    return xb @ weights + bias


###############################################################################
# Note that we no longer call ``log_softmax`` in the ``model`` function. Let's
# confirm that our loss and accuracy are the same as before:

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

###############################################################################
# Refactor using nn.Module
# -----------------------------
# Next up, we'll use ``nn.Module`` and ``nn.Parameter``, for a clearer and more
# concise training loop. We subclass ``nn.Module`` (which itself is a class and
# able to keep track of state).  In this case, we want to create a class that
# holds our weights, bias, and method for the forward step.  ``nn.Module`` has a
# number of attributes and methods (such as ``.parameters()`` and ``.zero_grad()``)
# which we will be using.
#
# .. note:: ``nn.Module`` (uppercase M) is a PyTorch specific concept, and is a
#    class we'll be using a lot. ``nn.Module`` is not to be confused with the Python
#    concept of a (lowercase ``m``) `module <https://docs.python.org/3/tutorial/modules.html>`_,
#    which is a file of Python code that can be imported.

from torch import nn


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias


###############################################################################
# Since we're now using an object instead of just using a function, we
# first have to instantiate our model:

model = Mnist_Logistic()

###############################################################################
# Now we can calculate the loss in the same way as before. Note that
# ``nn.Module`` objects are used as if they are functions (i.e they are
# *callable*), but behind the scenes Pytorch will call our ``forward``
# method automatically.

print(loss_func(model(xb), yb))

###############################################################################
# Previously for our training loop we had to update the values for each parameter
# by name, and manually zero out the grads for each parameter separately, like this:
# ::
#   with torch.no_grad():
#       weights -= weights.grad * lr
#       bias -= bias.grad * lr
#       weights.grad.zero_()
#       bias.grad.zero_()
#
#
# Now we can take advantage of model.parameters() and model.zero_grad() (which
# are both defined by PyTorch for ``nn.Module``) to make those steps more concise
# and less prone to the error of forgetting some of our parameters, particularly
# if we had a more complicated model:
# ::
#   with torch.no_grad():
#       for p in model.parameters(): p -= p.grad * lr
#       model.zero_grad()
#
#
# We'll wrap our little training loop in a ``fit`` function so we can run it
# again later.


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


fit()

###############################################################################
# Let's double-check that our loss has gone down:

print(loss_func(model(xb), yb))
