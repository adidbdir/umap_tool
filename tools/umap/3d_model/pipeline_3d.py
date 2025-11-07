import argparse
import json
import os
import subprocess
import sys
import time
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Train -> Test -> Plot pipeline for 3D UMAP")
    parser.add_argument("--train-config", required=True, type=str, help="Path to train config (json/yaml)")
    parser.add_argument("--test-input-dirs", nargs='+', required=True, help="Directories to test (recursive)")
    parser.add_argument("--test-prefix", type=str, default="obj", help="Extension for test files (default: from train config)")
    parser.add_argument("--test-max", type=int, default=100000, help="Max files per directory for test")
    parser.add_argument("--test-parallel", type=int, default=1, help="Parallel processes for test")
    parser.add_argument("--test-output-csv", type=str, default=None, help="Output CSV path for test embeddings")
    parser.add_argument("--plot-output-dir", type=str, default=None, help="Directory to save plot")
    parser.add_argument("--plot-title", type=str, default="3D Model UMAP Embeddings", help="Plot title")
    parser.add_argument("--plot-3d", action="store_true", help="Enable 3D plotting (requires umap_2)")
    parser.add_argument("--plot-show-names", action="store_true", help="Annotate points with file basenames")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable to invoke sub-steps")
    return parser.parse_args()


def run_cmd(cmd: List[str], env: dict = None) -> None:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def find_latest_subdir(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Output directory not found: {base_dir}")
    entries = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    subdirs = [d for d in entries if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No model directories found under: {base_dir}")
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def main():
    args = parse_args()

    # 1) Train
    # Disable XLA to avoid TF StatelessShuffle/XLA issue
    env = os.environ.copy()
    env["TF_ENABLE_XLA"] = "0"
    env["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

    train_cmd = [
        args.python_bin,
        os.path.join(os.path.dirname(__file__), "train_3d.py"),
        "--config", args.train_config,
    ]
    run_cmd(train_cmd, env=env)

    # Load train config to locate output directory
    with open(args.train_config, "r") as f:
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(f) if args.train_config.endswith((".yml", ".yaml")) else json.load(open(args.train_config))
        except Exception:
            f.seek(0)
            cfg = json.load(f)

    output_dir = cfg["output"]["dir"]
    model_dir = find_latest_subdir(output_dir)
    print(f"Detected model_dir: {model_dir}")

    # 2) Test
    test_prefix = args.test_prefix or cfg["input"].get("prefix", "obj")
    test_output_csv = args.test_output_csv or os.path.join(model_dir, "embeddings.csv")

    test_cmd = [
        args.python_bin,
        os.path.join(os.path.dirname(__file__), "test_3d.py"),
        model_dir,
        "-i",
        *args.test_input_dirs,
        "--prefix", test_prefix,
        "--max", str(args.test_max),
        "-p", str(args.test_parallel),
        "-o", test_output_csv,
    ]
    run_cmd(test_cmd, env=env)

    # 3) Plot
    plot_output_dir = args.plot_output_dir or os.path.join(model_dir, "plots")
    plot_cmd = [
        args.python_bin,
        os.path.join(os.path.dirname(__file__), "plot_3d.py"),
        "-i", test_output_csv,
        "-o", plot_output_dir,
        "--title", args.plot_title,
    ]
    if args.plot_3d:
        plot_cmd.append("--plot-3d")
    if args.plot_show_names:
        plot_cmd.append("--show-names")
    run_cmd(plot_cmd, env=env)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()


