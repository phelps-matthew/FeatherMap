import torch
from feathermap.utils import get_block_rows
from feathermap import feathernet
import unittest


def get_row_set(V1, V2, i1, j1, i2, j2, numels, n):
    ops = {}
    block_rows = get_block_rows(i1, j1, i2, j2, n)
    # Only one row, return whether complete or incomplete
    if i2 - i1 == 0:
        ops["top"] = (V1[i1, :], V2[:, j1 : j2 + 1])
        return ops
    # Has block rows
    if block_rows:
        ops["block"] = (V1[range(*block_rows), :], V2)
        # First row incomplete (from left)
        if i1 < block_rows[0]:
            ops["top"] = (V1[i1, :], V2[:, j1:])
        # Last row incomplete
        if i2 > (block_rows[1] - 1):
            ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
        return ops
    # Two rows, no blocks
    else:
        ops["top"] = (V1[i1, :], V2[:, j1:])
        ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
        return ops


def get_operand_slices(V, i1: int, j1: int, i2: int, j2: int, n: int) -> dict:
    """Return dictionary of operands representing complete slices of a matrix V"""
    ops = {}
    block_rows = get_block_rows(i1, j1, i2, j2, n)
    # Only one row, return whether complete or incomplete
    if i2 - i1 == 0:
        ops["top"] = V[i1, j1 : j2 + 1]
        return ops
    # Has block rows
    if block_rows:
        ops["block"] = V[range(*block_rows), :]
        # First row incomplete (from left)
        if i1 < block_rows[0]:
            ops["top"] = V[i1, j1:]
        # Last row incomplete
        if i2 > (block_rows[1] - 1):
            ops["bottom"] = V[i2, : j2 + 1]
        return ops
    # Two rows, no blocks
    else:
        ops["top"] = V[i1, j1:]
        ops["bottom"] = V[i2, : j2 + 1]
        return ops


def get_operands_str(i1: int, j1: int, i2: int, j2: int, n: int) -> dict:
    """Return dictionary of operands representing complete slices of V1.dot(V2) from range
    [i1, j1] to [i2, j2] corresponding to matrix of size (n x n)"""
    ops = {}
    block_rows = get_block_rows(i1, j1, i2, j2, n)
    # Only one row, return whether complete or incomplete
    if i2 - i1 == 0:
        ops["top"] = "(V1[{}, :], V2[:, {} : {} + 1])".format(i1, j1, j2)
        return ops
    # Has block rows
    if block_rows:
        ops["block"] = "(V1[range(*{}), :], V2)".format(block_rows)
        # First row incomplete (from left)
        if i1 < block_rows[0]:
            ops["top"] = "(V1[{}, :], V2[:, {}:])".format(i1, j1)
        # Last row incomplete
        if i2 > (block_rows[1] - 1):
            ops["bottom"] = "(V1[{}, :], V2[:, : {} + 1])".format(i2, j2)
        return ops
    # Two rows, no blocks
    else:
        ops["top"] = "(V1[{}, :], V2[:, {}:])".format(i1, j1)
        ops["bottom"] = "(V1[{}, :], V2[:, : {} + 1])".format(i2, j2)
        return ops


n, m = 3342, 1
V1 = torch.Tensor([list(range(n * q, n * (q + 1))) for q in range(m)]).reshape(n, m)
V2 = torch.Tensor([list(range(m * r, m * (r + 1))) for r in range(n)]).reshape(m, n)
V = torch.Tensor([list(range(n * q, n * (q + 1))) for q in range(n)])
# numels = 589824
n = 3342

# ------------------------------------------------ #
# Coverage Tests for block rows and operand slices #
# ------------------------------------------------ #

# All one row complete
print(get_block_rows(i1=42, j1=0, i2=42, j2=n - 1, n=n) == [42, 43])
print(get_operand_slices(V, i1=0, j1=0, i2=0, j2=n - 1, n=n))
# All two rows complete
print(get_block_rows(i1=652, j1=0, i2=653, j2=n - 1, n=n) == [652, 654])
print(get_operand_slices(V, i1=2, j1=0, i2=3, j2=n - 1, n=n))
# All 10 rows complete
print(get_block_rows(i1=652, j1=0, i2=662, j2=n - 1, n=n) == [652, 663])
print(get_operand_slices(V, i1=652, j1=0, i2=662, j2=n - 1, n=n))
# First row complete, second incomplete
print(get_block_rows(i1=0, j1=0, i2=1, j2=n - 100, n=n) == [0, 1])
print(get_operand_slices(V, i1=0, j1=0, i2=1, j2=n - 100, n=n))
# First row compete, last row incomplete
print(get_block_rows(i1=652, j1=0, i2=829, j2=105, n=n) == [652, 829])
print(get_operand_slices(V, i1=652, j1=0, i2=829, j2=105, n=n))
# First row incomplete (from right), no additional rows
print(get_block_rows(i1=1, j1=0, i2=1, j2=105, n=n) == [])
print(get_operand_slices(V, i1=1, j1=0, i2=1, j2=105, n=n))
# First row incomplete (from left), no additional rows
print(get_block_rows(i1=123, j1=123, i2=123, j2=n - 1, n=n) == [])
print(get_operand_slices(V, i1=123, j1=123, i2=123, j2=n - 1, n=n))
# First row incomplete (from left), second row incomplete
print(get_block_rows(i1=222, j1=1816, i2=223, j2=2018, n=n) == [])
print(get_operand_slices(V, i1=222, j1=1816, i2=223, j2=2018, n=n))
# First row incomplete (from left), last row incomplete
print(get_block_rows(i1=652, j1=1816, i2=829, j2=105, n=n) == [653, 829])
print(get_operand_slices(V, i1=652, j1=1816, i2=829, j2=105, n=n))
# First row incomplete (from left), second row complete
print(get_block_rows(i1=3000, j1=1816, i2=3001, j2=n - 1, n=n) == [3001, 3002])
print(get_operand_slices(V, i1=3000, j1=1816, i2=3001, j2=n - 1, n=n))
# First row incomplete (from left), last row complete
print(get_block_rows(i1=3000, j1=1816, i2=3091, j2=n - 1, n=n) == [3001, 3092])
print(get_operand_slices(V, i1=3000, j1=1816, i2=3091, j2=n - 1, n=n))
