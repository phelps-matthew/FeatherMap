from timeit import default_timer as timer
import torch
import logging
import matplotlib.pyplot as plt

# Disable font warnings from matplotlib
logging.getLogger("matplotlib.font_manager").disabled = True


def timed(method):
    def time_me(*args, **kw):
        start = timer()
        result = method(*args, **kw)
        end = timer()
        logging.info(
            "{!r} duration (secs):  {:.4f}".format(method.__name__, end - start)
        )
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
        logging.info(*cuda_status, sep="\n")
    except:
        logging.info("Some torch.cuda functionality unavailable")


def set_logger(filepath):
    logging.basicConfig(
        filename=str(filepath),
        filemode="w",  # will rewrite on each run
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation="spline16")

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
