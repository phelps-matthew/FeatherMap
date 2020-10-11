import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


compressions = np.array([0.01, 0.05, 0.1, 0.15, 0.2, 0.3])
# accuracies = np.array([71.99, 78.28, 80.99, 84.24, 85.07, 85.76])
accuracies = np.array([88.63, 91.63, 91.8, 92.11, 92.28, 92.89])
base_accuracy = 95.43

comp_fps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
fps = np.array([82.2, 79, 76.99, 73.46, 70.58])
base_fps = 88.02

sns.set_context("talk", font_scale=1.4)


def accuracy_graph():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(compressions, accuracies, "-o", markersize=15)
    ax.axhline(y=base_accuracy, linestyle="--")
    ax.set_xlabel("Model Size (fraction from original)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ResNet-34 on CIFAR10")
    ax.text(0, base_accuracy + 1, s="Base ResNet-34", ha="left")
    plt.xticks(np.arange(0, 0.35, 0.05))
    plt.yticks(np.arange(80, 105, 5))
    plt.show()


def fps_graph():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(comp_fps, fps, "-o")
    ax.axhline(y=base_fps, linestyle="--")
    ax.set_xlabel("Model Size (fraction from original)")
    ax.set_ylabel("FPS")
    ax.set_title("ResNet-34 on CIFAR10")
    ax.set_ylim((0, 100))
    ax.text(0.45, base_fps + 2, s="Base ResNet-34", ha="center")
    plt.yticks([20, 40, 60, 80, 100])
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.show()


if __name__ == "__main__":
   accuracy_graph()
