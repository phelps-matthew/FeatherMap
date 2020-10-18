"""Some helper functions for FeatherMap, including:
    - progress_bar: mimics xlua.progress.
    - timed: decorator for timing functions
    - get_block_rows: Get complete rows from range within matrix
"""
from timeit import default_timer as timer
import os
import sys
import time
from math import ceil
from typing import List


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 40.0
last_time = time.time()
begin_time = last_time


# current = batch idx, total = len(dataloader)
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: {:<4}".format(format_time(step_time)))
    L.append(" | Tot: {:<8}".format(format_time(tot_time)))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" {:>3}/{:<3} ".format(current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def timed(method):
    def time_me(*args, **kw):
        start = timer()
        result = method(*args, **kw)
        end = timer()
        print("{!r} duration (secs):  {:.4f}".format(method.__name__, end - start))
        return result

    return time_me


def get_block_rows(i1: int, j1: int, i2: int, j2: int, n: int) -> List[int]:
    """Return range of full (complete) rows from an (n x n) matrix, starting from row, col
    [i1, j1] and ending at [i2, j2]. E.g.

    | _ x x x |            | x x x |
    | x x x x |  ------>             + | x x x x |
    | x x x x |                        | x x x x | +
    | x x _ _ |                                      | x x |

    Necessary to make the most use of vectorized matrix multiplication. Sequentially
    calculating V[i, j] = V1[i, :] @ V2[:, j] leads to large latency.
    """
    row = []
    # Handle all cases for j1=0
    if j1 == 0:
        # All row(s) complete
        if j2 == n - 1:
            row.extend([i1, i2 + 1])
            return row
        # First row complete, last row incomplete
        elif i2 > i1:
            row.extend([i1, i2])
            return row
        # First row incomplete (from right), no additional rows
        else:
            return row
    # First row incomplete (from left), last row complete
    if j2 == n - 1 and i2 > i1:
        row.extend([i1 + 1, i2 + 1])
        return row
    # First row incomplete, last row incomplete; has at least one full row
    if i2 - i1 > 1:
        row.extend([i1 + 1, i2])
        return row
    # First row incomplete, second row incomplete
    return row
