import numpy as np
import argparse
import glob


def run(paths, limit=1000, return_dict=True, save=True):
    variance_dict = {}
    variance_list = []

    for p in paths:
        data = np.loadtxt(p, delimiter=",")
        variance = np.var(data, axis=0, ddof=1)

        if return_dict:
            variance_dict[p] = variance
        else:
            variance_list.append(variance)

    return variance_dict if return_dict else variance_list


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", help="data path", type=str, default="")
    parser.add_argument(
        "--result-type", help="result type", type=str, choices=["dict", "list"]
    )
    parser.add_argument("--save", help="save file flag", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    if args.data == "":
        dir_ = "../dummy"
    else:
        dir_ = args.data
    save = args.save
    result_dict = True if args.result_type == "dict" else False

    path = dir_ + "/*.csv"
    paths = glob.glob(path)

    variance_dict = run(paths, save=save)
    print(variance_dict)
