import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Noto Sans CJK JPフォントを設定
plt.rcParams["font.family"] = "Noto Sans CJK JP"

labels = [
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "exp5",
    "exp6",
    "exp7",
    "exp8",
    "exp9",
    "exp10",
    "exp11",
    "exp12",
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
    "実験9",
    "実験10",
    "実験11",
    "実験12",
    # "Real7k",
    "検証用データセット",
]

# データの読み込み
for i, label in enumerate(labels):
    print(label)
    if i == 0:
        embedding = np.loadtxt(f"./outputs/visapp/{label}.csv", delimiter=",")
        target = [i for e in range(len(embedding))]
    else:
        _embedding = np.loadtxt(f"./outputs/visapp/{label}.csv", delimiter=",")
        embedding = np.concatenate([embedding, _embedding])
        _target = [i for e in range(len(_embedding))]
        target = np.concatenate([target, _target])

x = embedding[:, 0]
y = embedding[:, 1]

cmap = plt.get_cmap("tab10")
# colors = [cmap(i) for i in range(len(labels))]
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
    cmap(9),
    cmap(10),
    cmap(11),
    cmap(12),
    cmap(13),
    # "w",
    # "k",
]

# 1. 全ての要素を描画して軸範囲を取得
plt.figure()
for n in np.unique(target):
    if n == len(labels) - 1:
        plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[n],
            label=labels_jp[n],
            s=50,
            marker="x"  # 最後の要素は"✖"
        )
    else:
        plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[n],
            label=labels_jp[n],
            s=50,
            marker="."  # 他の要素は"."
        )

# 凡例を追加
plt.legend(loc="upper left", fontsize=16)

# 画像を保存
plt.savefig("all_classes.png", format="png", dpi=300)

# 現在の軸範囲を取得
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

plt.close()  # 全体プロットを閉じる

# 2. 取得した軸範囲を使って各クラスごとの画像を作成
for n in np.unique(target):
    plt.figure()
    
    # 現在のクラスだけをプロット
    if n == len(labels) - 1:
        plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[n],
            label=labels_jp[n],
            s=50,
            marker="x"  # 最後の要素は"✖"
        )
    else:
        plt.scatter(
            x[target == n],
            y[target == n],
            alpha=1,
            color=colors[n],
            label=labels_jp[n],
            s=50,
            marker="."  # 他の要素は"."
        )
    
    # 取得した軸範囲を設定
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # 凡例を追加
    plt.legend(loc="upper left", fontsize=16)
    
    # 各クラスごとに画像を保存
    plt.savefig(f"class_{labels_jp[n]}.png", format="png", dpi=300)
    plt.close()  # 毎回プロットを閉じる
