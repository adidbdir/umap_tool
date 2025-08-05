import os
import argparse
import glob
import numpy as np
import cv2
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf
from tensorflow.keras.applications import ResNet50 as VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", default="data/test", type=str, help="学習する画像のディレクトリ"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="outputs/train/test",
        type=str,
        help="学習パラメータの保存ディレクトリ",
    )
    parser.add_argument("-s", "--size", default="224", type=int, help="学習する画像のサイズ")
    parser.add_argument("-p", "--prefix", default="jpg", type=str, help="学習する画像の拡張子")
    return parser.parse_args()


def create_data_and_target(path, num, size=224):
    paths = glob.glob(path, recursive=True)  # recursive=Trueを追加
    data = np.concatenate(
        [
            cv2.resize(cv2.imread(str(p)), dsize=(size, size)).flatten().reshape(1, -1)
            / 255.0
            for p in paths
        ],
        axis=0,
    )
    target = [num for i in range(len(paths))]
    return data, target


def create_encoder(dims, n_components):
    encoder = VGG16(weights="imagenet", include_top=False, input_shape=dims)
    encoder.trainable = False
    x = GlobalAveragePooling2D()(encoder.output)
    encoder_output = Dense(n_components)(x)
    encoder_model = Model(encoder.input, encoder_output)
    return encoder_model


def main():
    args = parse_args()

    # 再帰的にサブディレクトリ内も探索するために**を追加
    data, _ = create_data_and_target(
        f"{args.input_dir}/**/*.{args.prefix}",  # **を追加
        0,
        args.size,
    )

    dims = (args.size, args.size, 3)
    n_components = 2
    encoder = create_encoder(dims, n_components)

    # early stopping
    keras_fit_kwargs = {
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=10 ** -2,
                patience=10,
                verbose=1,
            )
        ],
        "batch_size": 4
    }

    mapper = ParametricUMAP(
        encoder=encoder,
        dims=dims,
        n_components=n_components,
        verbose=True,
        autoencoder_loss=True,
        keras_fit_kwargs=keras_fit_kwargs,
    )
    mapper.fit_transform(data)
    # ディレクトリが存在しない場合に作成する
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mapper.save(args.output_dir)


if __name__ == "__main__":
    if len(tf.config.list_physical_devices("GPU")) == 0:
        print("WARNING!!! CPU mode!!!")
        input("start >>> ")
    main()
