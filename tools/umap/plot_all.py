import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import ConvexHull
import alphashape
from descartes import PolygonPatch


# matplotlib.use("Qt5Agg")

labels = [
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "exp5",
    "exp6",
    "exp7",
    "exp8",
    "real",
]

labels_jp = [
    # "00",
    # "01",
    # "10",
    # "11",
    "実験1",
    "実験2",
    "実験3",
    "実験4",
    "実験5",
    "実験6",
    "実験7",
    "実験8",
    # "実験8",
    # "実験9",
    # "実験10",
    # "Real7k",
    "検証用データセット",
]


for i, label in enumerate(labels):
    print(label)
    if i == 0:
        embedding = np.loadtxt(f"./outputs/visapp/{label}.csv", delimiter=",")
        target = [i for e in range(len(embedding))]
    else:
        _embedding = np.loadtxt(
            f"./outputs/visapp/{label}.csv", delimiter=","
        )
        embedding = np.concatenate([embedding, _embedding])

        _target = [i for e in range(len(_embedding))]
        target = np.concatenate([target, _target])


# np.savetxt("./embedding.csv", embedding, delimiter=",")
# np.savetxt("./target.csv", target, delimiter=",", fmt="%.0f")

x = embedding[:, 0]
y = embedding[:, 1]

plt.rcParams["font.family"] = "Noto Serif CJK JP"

cmap = plt.get_cmap("tab10")
colors = [
    cmap(4),
    cmap(2),
    cmap(6),
    cmap(8),
    cmap(5),
    cmap(0),
    cmap(1),
    cmap(7),
    cmap(3),
    # cmap(9),
    # "w",
    # "k",
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
# mng.window.showMaximized()


labels_jp_sort = [
    # "00",
    # "01",
    # "10",
    # "11",
    "実験1",
    "実験2",
    "実験3",
    "実験4",
    "実験5",
    "実験6",
    "実験7",
    "実験8",
    # "実験9",
    # "実験10",
    # "Real7k",
    "検証用データセット",
]
# labels_jp_sort = [
#     "exp1",
#     "exp2",
#     "exp3",
#     "exp4",
#     "exp5",
#     "exp6",
#     "exp7",
#     "exp8",
#     "exp9",
#     "exp10",
#     "real",
# ]

handles_sort = [
    handles[0],
    handles[1],
    handles[2],
    handles[3],
    handles[4],
    handles[5],
    handles[6],
    handles[7],
    handles[8],
]
print("plotting...")
plt.rc("legend", fontsize=16)
# plt.xlim(-40, 35)
# plt.ylim(-5, 150)
# plt.legend(loc="upper left")
lgnd = plt.legend(handles_sort, labels_jp_sort, loc="upper left")
size = 300
for i in range(len(handles_sort)):
    lgnd.legendHandles[i].set_sizes([size])
plt.savefig("umap.png", format="png", dpi=300)
plt.show()

# convex hull
alpha_list = [
    0.0763483071602129,
    0.14031828161952678,
    0.076104965938761,
    0.02990140678233643,
    0.10917604422118761,
    0.3445336848190006,
    0.276289408177455,
    0.5731529420964571,
    0.12755135942775575,
    0.3858457782175682,
    0.396922287023833,
]
fig, ax = plt.subplots()

plt.rc("legend", fontsize=16)
lgnd = plt.legend(handles_sort, labels_jp_sort, loc="upper left")
size = 300
for i in range(len(handles_sort)):
    lgnd.legendHandles[i].set_sizes([size])
for n in np.unique(target):
    n = 10
    print(n)
    points = np.column_stack((x[target == n], y[target == n]))

    alpha = 0.95 * alphashape.optimizealpha(points[:1000])
    print(alpha)
    alpha = alpha_list[n]
    # alpha = 0.0763483071602129
    # np.savetxt(f"./alpha_{n}.txt", np.array(alpha))
    hull = alphashape.alphashape(points, alpha)
    hull_pts = hull.exterio r.coords.xy

    # ax.scatter(hull_pts[0], hull_pts[1], color=colors[n])
    ax.scatter(
        hull_pts[0],
        hull_pts[1],
        alpha=0,
        color=colors[n],
        label=labels_jp[n],
        s=1,
        marker=".",
    )
    handles.append(handle)
    # ax.scatter(x[target == n], y[target == n], color="red")
    ax.add_patch(PolygonPatch(hull, fill=False, color=colors[n]))
    # plt.savefig(f"./{n}.png")

    # hull = ConvexHull(points, qhull_options="QJ")
    # points = hull.points
    # hull_points = points[hull.vertices]

    # hp = np.vstack((hull_points, hull_points[0]))
    # plt.plot(hp[:, 0], hp[:, 1])
    # # plt.scatter(points[:, 0], points[:, 1])
    # plt.show()

plt.show()
