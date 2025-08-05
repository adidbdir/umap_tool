# Standard libraries
import glob
import json
from pathlib import Path
import argparse
import os
import tkinter as tk
from tkinter import filedialog
import statistics

# Third party libraries
import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns


def draw_kde(target_data, prefix=""):
    if type(target_data) is dict:
        values = list(target_data.values())
    elif type(target_data) is list:
        values = target_data
    # KDEプロットの描画
    sns.kdeplot(values)
    plt.xlim(0, 12)

    # タイトルとラベルの追加
    plt.title("Kernel Density Estimation of Values")
    plt.xlabel("Value")
    plt.ylabel("Density")

    # グラフの表示
    # plt.show()
    plt.savefig(f"{prefix}/statistics/glcm_kde.png")


def calc_glcm(img):
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=False,
        normed=True,
    )[:, :, 0, 0]

    sum = 0
    for i in range(len(glcm)):
        for j in range(len(glcm[i])):
            p = glcm[i][j]
            if p != 0:
                sum += p * np.log(p)
    glcm_entropy = -sum

    return glcm_entropy


def get_graph(xlabel="image id", prefix=""):
    plt.plot(np.loadtxt(f"{prefix}/statistics/glcm.txt"))
    plt.xlim(0, 12)
    plt.xlabel(xlabel)
    plt.ylabel("GLSM")
    plt.show()
    plt.savefig(f"{prefix}/statistics/glcm.png")


def run(paths, limit=-1, return_dict=True, save=True):
    """
    Parameters
    ---
    paths: list[str] or str
        list of image paths, or dir_path of images

    limit: int
        limit of image paths

    return_dict: bool
        return dict or list

    save: bool
        save entropy as json(dict) or txt(list)

    Return
    """

    if limit < 1:
        limit = len(paths)
    if type(paths) is str:
        types = ("jpg", "png")
        paths += [path for t in types for path in glob.glob(f"{paths}/*.{t}")]

    prefix = Path(paths[0]).parent

    entropy_dict = {}
    entropy_list = []

    # main process
    for p in tqdm.tqdm(paths[:limit]):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        entropy = calc_glcm(img)

        entropy_dict[p] = entropy
        entropy_list.append(entropy)

    # statistics
    if not len(entropy_list) > 2:
        statistics_dict = {
            "mean": statistics.mean(entropy_list),
            "median": 0,
            "variance": 0,
            "max": max(entropy_list),
            "min": min(entropy_list),
        }
    else:
        statistics_dict = {
            "mean": statistics.mean(entropy_list),
            "median": statistics.median(entropy_list),
            "variance": statistics.variance(entropy_list),
            "max": max(entropy_list),
            "min": min(entropy_list),
        }

    os.makedirs(f"{prefix}/statistics", exist_ok=True)
    plt.hist(entropy_list, bins=20, color=(0, 112 / 255, 192 / 255))
    plt.xlim(0, 12)
    plt.xticks(range(0, 12))
    plt.savefig(f"{prefix}/statistics/glcm_hist.png")
    plt.clf()
    draw_kde(entropy_list, prefix=prefix)

    # plt.savefig(f"{prefix}/glcm_hist.png")

    # save result as json or txt
    if save:
        if return_dict:
            with open(f"{prefix}/statistics/glcm.json", "w") as f:
                json.dump(entropy_dict, f, indent=4)
        else:
            np.savetxt(f"{prefix}/statistics/glcm.txt", entropy_list)

        with open(f"{prefix}/statistics/statistics.json", "w") as f:
            json.dump(statistics_dict, f, indent=4)
    else:
        print(statistics_dict)

    return entropy_dict if return_dict else entropy_list


def sorted_save(entropy_dict):
    file_keys = list(entropy_dict.keys())
    file_keys.sort()

    with open("glcm.txt", "w") as f:
        for file_key in file_keys:
            entropy = entropy_dict[file_key]
            f.write(str(entropy) + "\n")
    # get_graph()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="data path", type=str, default="")
    parser.add_argument("--gui", "-g", action="store_true")
    parser.add_argument(
        "--result-type", help="result type", type=str, choices=["dict", "list"]
    )
    parser.add_argument("--save", help="save file flag", action="store_true")
    # parser.add_argument("--no-limit", help="limit of images", action="store_true")
    return parser.parse_args()


def select_directory():
    # ルートウィンドウを作成
    root = tk.Tk()
    root.withdraw()  # ルートウィンドウを非表示にする

    # カレントディレクトリを取得
    current_directory = os.getcwd()

    # ファイルダイアログを開いてディレクトリを選択させる
    # initialdirにカレントディレクトリを設定
    directory_path = filedialog.askdirectory(initialdir=current_directory)

    # ルートウィンドウを閉じる
    root.destroy()

    return directory_path


if __name__ == "__main__":
    # argparse settings
    args = arg_parse()
    if args.gui == True:
        dir_ = select_directory()
    else:
        if args.data == "":
            dir_ = "../dummy"
        else:
            dir_ = args.data
    save = args.save
    result_dict = True if args.result_type == "dict" else False

    # set path
    path = dir_ + "/*.jpg"
    paths = glob.glob(path)

    # run
    # if not args.no_limit:
    entropy_dict = run(paths, save=save)
    # else:
    #     entropy_dict = run(
    #         paths,
    #         save=save,
    #         no_limit=args.no_limit,
    #     )
    sorted_save(entropy_dict)
    hoge = []
    for k, v in entropy_dict.items():
        hoge.append(v)
