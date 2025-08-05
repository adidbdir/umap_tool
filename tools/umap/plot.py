import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels", default="est", type=str, help="学習する画像のディレクトリ")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="outputs/train/test",
        type=str,
        help="学習パラメータの保存ディレクトリ",
    )
    parser.add_argument("-s", "--size", default="224", type=float, help="学習する画像のサイズ")
    parser.add_argument("-p", "--prefix", default="jpg", type=str, help="学習する画像の拡張子")
    return parser.parse_args()


labels = [
    "exp7_random_10k",
    # "exp7_10k",
    "exp7_dist_8_10k",
    # "exp4",
    # "exp8",
    # "exp6",
    # "exp5",
    # "exp3",
    # "exp2",
    # "exp1",
    "real",
    # "unity_low",
]

for i, label in enumerate(labels):
    print("Plotting: ", label)
    if i == 0:
        embedding = np.loadtxt(f"./outputs/{label}.csv", delimiter=",")
        target = [i for e in range(len(embedding))]
    else:
        _embedding = np.loadtxt(f"./outputs/{label}.csv", delimiter=",")
        embedding = np.concatenate([embedding, _embedding])

        _target = [i for e in range(len(_embedding))]
        target = np.concatenate([target, _target])


np.savetxt("./embedding.csv", embedding, delimiter=",")
np.savetxt("./target.csv", target, delimiter=",", fmt="%.0f")

x = embedding[:, 0]
y = embedding[:, 1]

plt.rcParams["font.family"] = "Noto Serif CJK JP"

cmap = plt.get_cmap("tab10")
colors = [
    cmap(0),
    cmap(2),
    # cmap(1),
    # cmap(4),
    # cmap(7),
    # cmap(6),
    # cmap(8),
    # cmap(5),
    cmap(3),
    cmap(9),  # unity
]
handles = []

# Define labels_jp based on the labels list for now
# You might want to customize these with actual Japanese translations
labels_jp = labels[:] 

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
plt.xlim(-40, 35)
plt.ylim(-5, 150)
# plt.legend(loc="upper left")
lgnd = plt.legend(handles_sort, labels_jp_sort, loc="upper left")
size = 300
for i in range(len(handles_sort)):
    lgnd.legendHandles[i].set_sizes([size])
plt.savefig("umap.png", format="png", dpi=300)
plt.show()
