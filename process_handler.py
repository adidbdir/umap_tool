import os
import subprocess

train = "outputs/train/"

# 実行する設定ファイルのリスト
dataset_files = [
    "dataset/visap_umap/1/data/**/**",
    "dataset/visap_umap/2/data/**/**",
    "dataset/visap_umap/3/data/**/**",
    "dataset/visap_umap/4/data/**/**",
    "dataset/visap_umap/5/data/**/**",
    "dataset/visap_umap/6/data/**/**",
    "dataset/visap_umap/7/data/**/**",
    "dataset/visap_umap/8/data/**/**",
    "dataset/real"
]

# 対応する出力ディレク'トリのリスト
output_dir = "outputs/visapp/"
output_files = [
    "exp1.csv",
    "exp2.csv",
    "exp3.csv",
    "exp4.csv",
    "exp5.csv",
    "exp6.csv",
    "exp7.csv",
    "exp8.csv",
    "real.csv",
]

# ディレクトリが存在しない場合に作成する
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# for文で設定ファイルと出力ディレクトリを繰り返し処理
for i in range(len(dataset_files)):
    dataset_file = dataset_files[i]
    output_file = output_files[i]
    # コマンドを組み立てて実行
    command = ['python', 'tools/umap/test.py', train, '-i', dataset_file, '-o', output_dir + output_file, '-s', "224"]
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    
    # エラーチェック
    if result.returncode != 0:
        print(f"Error encountered while running: {dataset_file}")
        break  # エラーが発生した場合はループを中断
    else:
        print(f"Completed: {dataset_file}")
