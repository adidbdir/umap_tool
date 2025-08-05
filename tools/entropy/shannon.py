# Standard Library
import glob
import json
from pathlib import Path
from collections import Counter
import argparse
import statistics

# Third Party Library
import numba
import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np


@numba.njit(cache=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass


@numba.njit(cache=True)
def getIJ(final_img, ext_x, ext_y, width, height):
    IJ = []
    for i in numba.prange(ext_x, width - ext_x):
        for j in numba.prange(ext_y, height - ext_y):
            patch = final_img[j - ext_y : j + ext_y + 1, i - ext_x : i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)
    return IJ


def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    IJ = getIJ(final_img, ext_x, ext_y, new_width, new_height)

    Fij = Counter(IJ).items()

    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H


def get_graph(xlabel="image id"):
    plt.plot(np.loadtxt("shannon_entropy.txt"))
    plt.xlabel(xlabel)
    plt.ylabel("shanon entropy")
    plt.show()
    plt.savefig("shannon_entropy.png")


def run(paths, limit=1000, return_dict=True, save=True):
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

    for p in tqdm.tqdm(paths[:limit]):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        entropy = calcEntropy2dSpeedUp(img, 3, 3)

        # store result along return type
        if return_dict:
            entropy_dict[p] = entropy
        else:
            entropy_list.append(entropy)

    # save result as json or txt
    if save:
        if return_dict:
            with open(f"{prefix}/entropy.json", "w") as f:
                json.dump(entropy_dict, f, indent=4)
        else:
            np.savetxt(f"{prefix}/entropy.txt", entropy_list)

        # statistics
        statistics_dict = {
            "mean": statistics.mean(entropy_list),
            "median": statistics.median(entropy_list),
            "variance": statistics.variance(entropy_list),
            "max": max(entropy_list),
            "min": min(entropy_list),
        }
        with open(f"{prefix}/statistics.json", "w") as f:
            json.dump(statistics_dict, f, indent=4)

    return entropy_dict if return_dict else entropy_list


def sorted_save(entropy_dict):
    file_keys = list(entropy_dict.keys())
    file_keys.sort()

    with open("shannon_entropy.txt", "w") as f:
        for file_key in file_keys:
            entropy = entropy_dict[file_key]
            f.write(str(entropy) + "\n")

    get_graph()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="data path", type=str, default="")
    parser.add_argument(
        "--result-type", help="result type", type=str, choices=["dict", "list"]
    )
    parser.add_argument("--save", help="save file flag", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # argparse settings
    args = arg_parse()
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
    entropy_dict = run(paths, save=save)
    sorted_save(entropy_dict)
