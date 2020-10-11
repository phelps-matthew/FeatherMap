import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


compressions = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25])
accuracies = np.array([88.63, 90.01, 90.72, 91.02, 91.75, 91.8, 92.11, 92.28, 92.41])
base_accuracy = 95.43


latency_delta = np.array([4.08, 4.29, 5.74, 5.45, 8.56, 13.37, 18.32, 23.44, 26.05])

sns.set_context("talk", font_scale=1.4)


def accuracy_graph():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(compressions, accuracies, "-o", markersize=15)
    ax.axhline(y=base_accuracy, linestyle="--")
    ax.set_xlabel("Model Size (fraction from original)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ResNet-34 Accuracy")
    ax.text(0.002, base_accuracy + 0.4, s="Base ResNet-34", ha="left")
    plt.xticks(np.arange(0, 0.30, 0.05))
    plt.yticks(np.arange(80, 105, 5))
    plt.show()


def latency_graph():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(compressions, latency_delta, "-o", markersize=15)
    ax.set_xlabel("Model Size (fraction from original)")
    ax.set_ylabel("Latency Increase (%)")
    ax.set_title("ResNet-34 Latency")
    ax.set_ylim((0, 27))
    # plt.yticks([20, 40, 60, 80, 100])
    plt.xticks(np.arange(0, 0.30, 0.05))
    plt.show()


def accuracy_latency_graph():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ln1 = ax1.plot(compressions, accuracies, "-o", markersize=15)
    ax1.axhline(y=base_accuracy, linestyle="--")
    ax1.text(0.002, base_accuracy + 0.4, s="Base ResNet-34", ha="left", fontsize=20)
    ax1.set_xticks(np.arange(0, 0.30, 0.05))
    ax1.set_xlabel("Model Size (fraction from original)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("ResNet-34")
    ax1.set_yticks(np.linspace(75, 100, 6))
    #ax1.legend(["Accuracy"], loc=1)

    ax2 = ax1.twinx()
    ax2.grid(None)
    ln2 = ax2.plot(compressions, latency_delta, "r-o", markersize=15)
    ax2.set_yticks(np.linspace(0, 50, 6))
    ax2.set_ylabel("Latency Increase (%)")
    ax1.legend(ln1+ln2, ["Accuracy", "Latency"], loc=1)

    plt.show()


if __name__ == "__main__":
    accuracy_latency_graph()
