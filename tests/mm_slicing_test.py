import torch


def get_block_rows(i, j, numels, n):
    j_start = j == 0
    rows = []
    for x in range(numels):
        if j > n:
            if j_start:
                rows.append(i)
            i += 1
            j = 0
            j_start = True
        j += 1
    return rows


 def get_row_set(V1, V2, i1, j1, i2, j2, numels, n):
    ops = {}
    block_rows = get_block_rows(i1, j1, numels, n)
    # Only one row, return whether complete or incomplete
    if i2 - i1 == 0:
        ops["top"] = (V1[i1, :], V2[:, j1 : j2 + 1])
        return ops
    # Has block rows
    if len(block_rows) != 0:
        ops["block"] = (V1[block_rows, :], V2)
        for row in range(i1, i2 + 1):
            if row not in block_rows:
                if row < min(block_rows):
                    ops["top"] = (V1[row, :], V2[:, j1:])
                else:
                    ops["bottom"] = (V1[row, :], V2[:, : j2 + 1])
        return ops
    # Two rows, no blocks
    else:
        ops["top"] = (V1[i1, :], V2[:, j1:])
        ops["bottom"] = (V1[i2, :], V2[:, : j2 + 1])
        return ops


def get_row_set_V(V, i1, j1, i2, j2, numels, n):
    ops = {}
    block_rows = get_block_rows(i1, j1, numels, n)
    # Only one row, return whether complete or incomplete
    if i2 - i1 == 0:
        ops["top"] = V[i1, j1 : j2 + 1]
        return ops
    # Has block rows
    if len(block_rows) != 0:
        ops["block"] = V[block_rows, :]
        for row in range(i1, i2 + 1):
            if row not in block_rows:
                if row < min(block_rows):
                    ops["top"] = V[row, j1:]
                else:
                    ops["bottom"] = V[row, : j2 + 1]
        return ops
    # Two rows, no blocks
    else:
        ops["top"] = V[i1, j1:]
        ops["bottom"] = V[i2, : j2 + 1]
        return ops


n, m = 5, 1
V1 = torch.Tensor([list(range(n * q, n * (q + 1))) for q in range(m)]).reshape(n, m)
V2 = torch.Tensor([list(range(m * r, m * (r + 1))) for r in range(n)]).reshape(m, n)
V = torch.Tensor([list(range(n * q, n * (q + 1))) for q in range(n)])
print(V1)
print(V2)
print(V)
numels = 18
(i1, j1, i2, j2) = (0, 1, 3, 3)
print(get_block_rows(i1, j1, numels, n))
print(get_row_set_V(V, i1, j1, i2, j2, numels, n))
print(get_row_set(V1, V2, i1, j1, i2, j2, numels, n))
