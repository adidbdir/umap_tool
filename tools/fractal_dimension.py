# get_husdroff_fractal
import numpy as np
import matplotlib.pyplot as plt

# get_housdroff_fractal
from skimage import io
from scipy.stats import linregress


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def preprocess(file_name):
    """
    Parameters
    ---
    file_name: str

    Return
    ---
    image: np.ndarray, (matplotlib)
    """
    image = rgb2gray(plt.imread(file_name))
    return image


def get_housdroff_fractal(file_name):
    """
    Parameters
    ---
    target_image: matplotlib image
        grayscale image

    Return
    ---
    fractal_dimension: float

    Memo(Suzuki)
    ---
    # This program is offered by Ono
    # 0: point, 0D
    # 1: line, 1D
    # 2: plane, 2D
    # 3: volume, 3D
    # src: https://wagtail.cds.tohoku.ac.jp/coda/topics/fractals/fractal/fractal.html
    """

    # load and preprocess
    target_image = preprocess(file_name)

    # finding all the non-zero pixels
    positive_pixels = []

    for i in range(target_image.shape[0]):  # height
        for j in range(target_image.shape[1]):  # width
            if target_image[i, j] > 0:
                positive_pixels.append((i, j))
    positive_pixels = np.array(positive_pixels)

    Lx = target_image.shape[1]
    Ly = target_image.shape[0]

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    for scale in scales:
        # print("======= Scale :", scale)
        # computing the histogram
        H, _ = np.histogramdd(
            positive_pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale))
        )
        Ns.append(np.sum(H > 0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

    # save the figure
    drwaing = False
    if drwaing:
        plt.plot(np.log(scales), np.log(Ns), "o", mfc="none")
        plt.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
        plt.xlabel("log $\epsilon$")
        plt.ylabel("log N")
        plt.savefig("sierpinski_dimension.pdf")

        print(
            "The Hausdorff dimension is", -coeffs[0]
        )  # the fractal dimension is the OPPOSITE of the fitting coefficient
        # np.savetxt("scaling.txt", list(zip(scales, Ns)))
    return -coeffs[0]


# Function to perform boxcounting　by ChatGPT
def get_boxcount_fractal(binary_image, threshold=0.9):
    """
    by ChatGPT
    Prompt: フラクタル次元を計算するプログラムを作ってこの画像のフラクタル画像を計算して
    """
    # Load the image from file system
    image_path = file_name
    image = io.imread(image_path, as_gray=True)
    # Convert the image to binary
    thresh = np.mean(image)
    binary_image = image > thresh
    assert len(binary_image.shape) == 2

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    binary_image = binary_image < threshold

    # Minimal dimension of image
    p = min(binary_image.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(binary_image, size))

    # Fit the counts
    coeffs = linregress(np.log(sizes), np.log(counts))

    return -coeffs.slope


if __name__ == "__main__":
    file_name = "../117379057_226200615302832_533548947832366105_n.jpg"
    res = get_housdroff_fractal(file_name)
