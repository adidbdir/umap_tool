#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Trying to reproduce: https://arxiv.org/abs/1609.01117


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import statistics

# dir = "../dummy"
# path = dir + "/*.jpg"
# paths = glob.glob(path)
# images = []
# file path loop load opencv type of image data


def get_graph(xlabel="image id"):
    plt.plot(np.loadtxt("delentropy.txt"))
    plt.xlabel(xlabel)
    plt.ylabel("Delentropy")
    plt.show()
    plt.savefig("delentropy.png")


def calc_delentropy(image):
    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    if True:
        # see paper eq. (4)
        fx = (image[:, 2:] - image[:, :-2])[1:-1, :]
        fy = (image[2:, :] - image[:-2, :])[:, 1:-1]
    else:
        # throw away last row, because it seems to show some artifacts which it shouldn't really
        # Cleaning this up does not seem to work
        kernelDiffY = np.array([[-1, -1], [1, 1]])
        fx = signal.fftconvolve(image, kernelDiffY.T).astype(image.dtype)[:-1, :-1]
        fy = signal.fftconvolve(image, kernelDiffY).astype(image.dtype)[:-1, :-1]
    # print("fx in [{},{}], fy in [{},{}]".format(fx.min(), fx.max(), fy.min(), fy.max()))
    diffRange = np.max(
        [np.abs(fx.min()), np.abs(fx.max()), np.abs(fy.min()), np.abs(fy.max())]
    )
    if diffRange >= 200 and diffRange <= 255:
        diffRange = 255
    if diffRange >= 60000 and diffRange <= 65535:
        diffRange = 65535

    # see paper eq. (17)
    # The bin edges must be integers, that's why the number of bins and range depends on each other
    nBins = min(1024, 2 * diffRange + 1)
    if image.dtype == np.float64:
        nBins = 1024
    # print("Bins = {}, Range of Diff = {}".format(nBins, diffRange))
    # Centering the bins is necessary because else all value will lie on
    # the bin edges thereby leading to assymetric artifacts
    dbin = 0 if image.dtype == np.float64 else 0.5
    r = diffRange + dbin
    delDensity, xedges, yedges = np.histogram2d(
        fx.flatten(), fy.flatten(), bins=nBins, range=[[-r, r], [-r, r]]
    )
    if nBins == 2 * diffRange + 1:
        assert xedges[1] - xedges[0] == 1.0
        assert yedges[1] - yedges[0] == 1.0

    # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
    # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
    # assert( np.product( np.array( image.shape ) - 1 ) == np.sum( delDensity ) )
    delDensity = delDensity / np.sum(delDensity)  # see paper eq. (17)
    delDensity = delDensity.T
    # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
    # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
    entropy = -0.5 * np.sum(
        delDensity[delDensity.nonzero()] * np.log2(delDensity[delDensity.nonzero()])
    )  # see paper eq. (16)

    def draw_graphs(target_image_name=""):
        # gamma enhancements and inversion for better viewing pleasure
        delDensity = np.max(delDensity) - delDensity
        gamma = 1.0
        delDensity = (delDensity / np.max(delDensity)) ** gamma * np.max(delDensity)

        title_text = (
            "Example image " + str(target_image_name) + ", entropy="
            if not target_image_name == ""
            else "entropy="
        )
        ax = [
            fig.add_subplot(
                221,
                title=title_text + str(np.round(entropy, 3)),
            ),
            fig.add_subplot(
                222,
                title="x gradient of image (color range: ["
                + str(np.round(-diffRange, 3))
                + ","
                + str(np.round(diffRange, 3))
                + "])",
            ),
            fig.add_subplot(
                223,
                title="y gradient of image (color range: ["
                + str(np.round(-diffRange, 3))
                + ","
                + str(np.round(diffRange, 3))
                + "])",
            ),
            fig.add_subplot(
                224, title="Histogram of gradient (gamma corr. " + str(gamma) + " )"
            ),
        ]
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[1].imshow(fx, cmap=plt.cm.gray, vmin=-diffRange, vmax=diffRange)
        ax[2].imshow(fy, cmap=plt.cm.gray, vmin=-diffRange, vmax=diffRange)
        ax[3].imshow(
            delDensity,
            cmap=plt.cm.gray,
            vmin=0,
            interpolation="nearest",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )

        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        fig.savefig("delentropy.png")

    return entropy


def run(paths, limit=1000, return_dict=True, save=True):
    if limit < 1:
        limit = len(paths)

    if type(paths) is str:
        types = ("jpg", "png")
        paths += [path for t in types for path in glob.glob(f"{paths}/*.{t}")]
    prefix = Path(paths[0]).parent

    entropy_dict = {}
    entropy_list = []

    for p in tqdm(paths[:limit]):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        entropy = calc_delentropy(img)
        if return_dict:
            entropy_dict[p] = entropy
        else:
            entropy_list.append(entropy)

    # save result as json or txt
    if save:
        if return_dict:
            with open(f"{prefix}/delentropy.json", "w") as f:
                json.dump(entropy_dict, f, indent=4)
        else:
            np.savetxt(f"{prefix}/delentropy.txt", entropy_list)

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

    with open("delentropy.txt", "w") as f:
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
