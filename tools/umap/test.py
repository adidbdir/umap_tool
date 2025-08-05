import argparse
import multiprocessing
import glob
import numpy as np
import tqdm

import os
import cv2

from umap.parametric_umap import load_ParametricUMAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        default="outputs/train/test",
        type=str,
        help="学習済みモデルのディレクトリ",
    )
    parser.add_argument(
        "-i", "--input-dir", default="data/test", type=str, help="テストする画像のディレクトリ"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="outputs/test/test.csv",
        type=str,
        help="テストパラメータの保存パス",
    )
    parser.add_argument("-s", "--size", default="64", type=int, help="テストする画像のサイズ")
    parser.add_argument("--max", default="10000", type=int, help="テストする画像の数")
    parser.add_argument("-p", "--parallel", default="1", type=int, help="並列処理の数")
    parser.add_argument("--prefix", default="jpg", type=str, help="学習する画像の拡張子")
    return parser.parse_args()


def create_embedding_and_target(
    proc_num,
    model_dir,
    paths,
    size,
    embedding_dict,
    path_dict,
):
    mapper = load_ParametricUMAP(model_dir)

    embedding = None
    p_list = []
    for p in tqdm.tqdm(paths):
        p_list.append(p)
        data = [cv2.resize(cv2.imread(str(p)), dsize=(size, size)) / 255.0]
        # data = [cv2.imread(str(p)).flatten().reshape(1, -1) / 255.0]

        _embedding = mapper.transform(data)
        if embedding is None:
            embedding = _embedding
        else:
            embedding = np.concatenate([embedding, _embedding])

    embedding_dict[proc_num] = embedding
    path_dict[proc_num] = p_list


def run(paths, model_dir, size, parallel):
    manager = multiprocessing.Manager()
    embedding_dict = manager.dict()
    path_dict = manager.dict()
    jobs = []
    div_paths = np.array_split(paths, parallel)
    for i in range(parallel):
        p = multiprocessing.Process(
            target=create_embedding_and_target,
            args=(i, model_dir, div_paths[i], size, embedding_dict, path_dict),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for i in range(parallel):
        if i == 0:
            embedding = embedding_dict[0]
        else:
            embedding = np.concatenate([embedding, embedding_dict[i]])

    return embedding


def main():
    args = parse_args()

    # paths = glob.glob(f"{args.input_dir}/*.{args.prefix}")
    paths = glob.glob(os.path.join(args.input_dir, '**', '*.jpg'), recursive=True)
    sorted_path = sorted(paths)[: args.max]
    embedding = run(sorted_path, args.model_dir, args.size, args.parallel)
    np.savetxt(args.output_path, embedding, delimiter=",")


if __name__ == "__main__":
    main()
