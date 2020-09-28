from timeit import default_timer as timer
import torch


def timed(method):
    def time_me(*args, **kw):
        start = timer()
        result = method(*args, **kw)
        end = timer()
        print("{!r} duration (secs):  {:.4f}".format(method.__name__, end - start))
        return result

    return time_me


def print_gpu_status():
    """Print GPU torch cuda status"""
    try:
        cuda_status = [
            "torch.cuda.device(0): {}".format(torch.cuda.device(0)),
            "torch.cuda.device_count(): {}".format(torch.cuda.device_count()),
            "torch.cuda.get_device_name(0): {}".format(torch.cuda.get_device_name(0)),
            "torch.cuda_is_available: {}".format(torch.cuda.is_available()),
            "torch.cuda.current_device: {}".format(torch.cuda.current_device()),
        ]
        print(*cuda_status, sep="\n")
    except:
        print("Some torch.cuda functionality unavailable")
