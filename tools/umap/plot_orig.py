import numpy as np

import matplotlib.pyplot as plt

labels = [
    # "0_color_1k",
    "exp_7",
    "1_single",
    "2_multi",
    "real_50",
]

labels_jp = [
    # "flat",
    "exp_7",
    "single",
    "multi",
    "real",
]


for i, label in enumerate(labels):
    print(label)
    if i == 0:
        embedding = np.loadtxt(
            f"./outputs/test/patent_real/{label}_10k.csv", delimiter=","
        )
        target = [i for e in range(len(embedding))]
    else:
        _embedding = np.loadtxt(
            f"./outputs/test/patent_real/{label}_10k.csv", delimiter=","
        )
        embedding = np.concatenate([embedding, _embedding])

        _target = [i for e in range(len(_embedding))]
        target = np.concatenate([target, _target])


np.savetxt("./embedding.csv", embedding, delimiter=",")
np.savetxt("./target.csv", target, delimiter=",", fmt="%.0f")

x = embedding[:, 0]
y = embedding[:, 1]

# plt.rcParams["font.family"] = "Noto Serif CJK JP"

cmap = plt.get_cmap("tab10")
colors = [
    cmap(0),
    cmap(2),
    cmap(1),
    # cmap(4),
    # cmap(7),
    # cmap(6),
    # cmap(8),
    # cmap(5),
    cmap(3),
    # cmap(9),  # unity
]
handles = []

for n in np.unique(target):
    if n == len(labels) - 1:
        handle = plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[i],
            label=labels_jp[n],
            s=50,
            marker="x",
        )
    else:
        handle = plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[n],
            label=labels_jp[n],
            s=50,
            marker=".",
        )
    handles.append(handle)

mng = plt.get_current_fig_manager()
mng.window.showMaximized()


labels_jp_sort = [
    # "実験1",
    # "実験2",
    # "実験3",
    # "実験4",
    # "実験5",
    # "実験6",
    "近傍 1万枚",
    "ランダム 1万枚",
    # "実験8",
    "検証用データセット",
    # "Unity",
]

# handles_sort = [
#     handles[7],
#     handles[6],
#     handles[5],
#     handles[3],
#     handles[4],
#     handles[1],
#     handles[0],
#     handles[2],
#     handles[8],
#     # handles[9],
# ]
handles_sort = [handles[1], handles[0], handles[2]]

print("plotting...")
plt.rc("legend", fontsize=16)
# plt.xlim(-40, 35)
# plt.ylim(-5, 150)
# plt.legend(loc="upper left")
# lgnd = plt.legend(handles_sort, labels_jp_sort, loc="upper left")
lgnd = plt.legend(handles, labels_jp, loc="upper right")
size = 300
# for i in range(len(handles_sort)):
for i in range(len(handles)):
    lgnd.legendHandles[i].set_sizes([size])
plt.savefig("umap.png", format="png", dpi=300)
plt.show()
