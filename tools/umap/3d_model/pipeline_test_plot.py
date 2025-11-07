import argparse
import os
import subprocess
import sys
from typing import List
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Test -> Plot pipeline for 3D UMAP")
    parser.add_argument("model_dir", type=str, help="Path to trained model directory")
    parser.add_argument("--test-input-dirs", "-i", nargs='+', required=True, help="Directories to test (recursive)")
    parser.add_argument(
        "--test-prefix", "-p",
        type=str,
        default="any",
        help="File extension(s) to test (e.g., 'obj' or 'obj,ply,stl' or 'any')",
    )
    parser.add_argument("--test-max", type=int, default=100000, help="Max files per directory for test")
    parser.add_argument("--test-parallel", type=int, default=5, help="Parallel processes for test")
    parser.add_argument("--test-output-csv", type=str, default=None, help="Output CSV for embeddings (default: <model_dir>/embeddings.csv)")
    parser.add_argument("--plot-output-dir", type=str, default=None, help="Directory to save plot (default: <model_dir>/plots)")
    parser.add_argument("--plot-title", type=str, default="3D Model UMAP Embeddings", help="Plot title")
    parser.add_argument("--plot-3d", action="store_true", help="Enable 3D plotting (requires umap_2)")
    parser.add_argument("--plot-show-names", "-s",action="store_true", help="Annotate points with file basenames")
    parser.add_argument("--plot-name-folder", "-f", action="store_true", help="Use parent folder name when annotating points")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for deterministic sampling (propagated to test phase)")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable to invoke sub-steps")
    return parser.parse_args()


def run_cmd(cmd: List[str], env: dict = None) -> None:
    print("[RUN]", " ".join(cmd))
    # Stream output live so inner tqdm bars (from child) render properly
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main():
    args = parse_args()

    env = os.environ.copy()
    # Disable XLA just in case TF is used by feature extractor
    env["TF_ENABLE_XLA"] = "0"
    env["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

    # Resolve/validate model_dir (must contain model.pkl)
    model_dir = args.model_dir.rstrip('/')
    model_pkl = os.path.join(model_dir, 'model.pkl')
    if not os.path.exists(model_pkl):
        # Try to find a subdirectory that has model.pkl
        candidates = []
        if os.path.isdir(model_dir):
            for root, dirs, files in os.walk(model_dir):
                if 'model.pkl' in files:
                    candidates.append(root)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            print(f"Warning: model.pkl not found in {model_dir}. Using detected model_dir: {candidates[0]}")
            model_dir = candidates[0]
        else:
            raise FileNotFoundError(f"model.pkl not found under {args.model_dir}. Specify a trained model directory.")

    # Test
    test_output_csv = args.test_output_csv or os.path.join(model_dir, "embeddings.csv")
    test_cmd = [
        args.python_bin,
        os.path.join(os.path.dirname(__file__), "test_3d.py"),
        model_dir,
        "-i",
        *args.test_input_dirs,
        "--prefix", args.test_prefix,
        "--max", str(args.test_max),
        "-p", str(args.test_parallel),
        "-o", test_output_csv,
    ]
    if args.seed is not None:
        test_cmd += ["--seed", str(args.seed)]
    with tqdm(total=2, desc="Pipeline", unit="step") as pbar:
        # Step 1: Test (extract embeddings)
        run_cmd(test_cmd, env=env)
        pbar.update(1)

    # Plot
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
    if args.plot_name_folder:
        plot_cmd.append("--name-folder")
    # Step 2: Plot
    run_cmd(plot_cmd, env=env)
    # Advance final step in the same bar context above if still open
    try:
        pbar.update(1)  # type: ignore[name-defined]
    except Exception:
        pass

    print("Test->Plot pipeline completed successfully.")


if __name__ == "__main__":
    main()


