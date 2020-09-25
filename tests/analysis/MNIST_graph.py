import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


compressions = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25])
accuracies = np.array([93.89, 96.03, 96.75, 96.96, 97.28, 97.19])
test = [
    [0.01, 93.89],
    [0.05, 96.03],
    [0.10, 96.75],
    [0.15, 96.96],
    [0.20, 97.28],
    [0.25, 97.19],
]
xy = list(zip(compressions, accuracies))
df = pd.DataFrame(
    [
        [0.01, 93.89],
        [0.05, 96.03],
        [0.10, 96.75],
        [0.15, 96.96],
        [0.20, 97.28],
        [0.25, 97.19],
    ],
    columns=["compression", "accuracy"],
)

sns.set_context("talk", font_scale=1.4)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(compressions, accuracies, "-o")
ax.axhline(y=97.9, linestyle="--")
ax.set_xlabel("Compression (fraction from original)")
ax.set_ylabel("Accuracy")
ax.set_title("MNIST Feedfoward NN (size = 500)")
ax.set_ylim((87, 100))
ax.text(0.125, 98.2, s="Full NN-500", ha='center')
# df.plot.scatter(x="compression", y="accuracy")
# print(data.head())
# ax = sns.lineplot(data=df, x="compression", y="accuracy", marker='o').set_title('MNIST Feedfoward NN')
# ax.axes[0][0].hlines( y = 20, color='black', linewidth=2, alpha=.7,xmin=97, xmax=98)
plt.show()
