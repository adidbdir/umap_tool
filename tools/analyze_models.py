# 出力ファイルと各項目の意味
# dataset_stats.csv（1行=1モデル）
    # file_path: ファイルのフル/相対パス
    # ext: 拡張子（obj, glb, …）
    # file_size_bytes: ファイルサイズ[bytes]
    # num_vertices: 頂点数（軽微なクリーニング後の数）
    # num_faces: 面数（同上）
    # is_watertight: メッシュが水密かどうか（True/False）
    # extent_x / extent_y / extent_z: AABBの各軸長（モデル座標系の単位。スケールがまちまちなら単位は相対的）
    # bbox_diag: AABB対角長（= sqrt(x^2 + y^2 + z^2)）
    # surface_area: 表面積（単位は座標系の2乗。計算失敗時はNaN）
    # volume: 体積（単位は座標系の3乗。水密でない場合は原則NaN）
# 備考: 読み込み時に重複頂点/面やゼロ面積の除去、法線修正などの軽微なクリーニングを行ってから数値化しています。
# summary.csv（全体統計）
    # count, mean, std, min, 25%, 50%, 75%, max: 数値列の記述統計
    # unique, top, freq: 文字列列の記述統計（例: ext など）
    # NaN は統計から除外されます（例: volume が非水密でNaNのときは有効データのみで集計）
    # histograms.png
    # num_vertices／num_faces／bbox_diag／extent_x／extent_y／extent_z の分布ヒストグラム
    # means_std.png
    # num_vertices／num_faces／bbox_diag／surface_area／volume の平均±標準偏差バー
# 解釈のポイント
# スケール（単位）が統一されていないデータセットの場合、extent系やsurface/volumeは相対比較になります。必要に応じて正規化や単位統一を検討してください。
# volume は水密（is_watertight=True）で計算されます。False の行は NaN になり得ます。
# クリーニングにより頂点/面の重複が削られるため、原データと数がわずかに異なることがあります。
# 必要なら、追加指標（例: 三角形アスペクト比、法線一貫性、メッシュ連結成分数など）や、拡張子・フォルダ単位の集計グラフも拡張可能です。

import argparse
import os
import glob
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


DEFAULT_3D_EXTS: List[str] = ["obj", "glb", "gltf", "ply", "stl", "fbx", "dae"]


def parse_exts(value: str) -> List[str]:
    if value is None or value.strip().lower() in {"any", "*", "all", "3d", "all3d", ""}:
        return DEFAULT_3D_EXTS
    return sorted(set([p.strip().lower().lstrip('.') for p in value.split(',') if p.strip()]))


def collect_files(input_dir: str, extensions: List[str], max_files: int | None = None) -> List[str]:
    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(input_dir, "**", f"*.{ext}"), recursive=True))
    # de-duplicate while preserving order
    seen = set()
    deduped = [p for p in paths if not (p in seen or seen.add(p))]
    if max_files is not None and max_files > 0:
        return deduped[:max_files]
    return deduped


def load_mesh(path: str) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        # basic sanitization similar to train pipeline
        if hasattr(mesh, 'deduplicate_vertices'):
            mesh.deduplicate_vertices()
        if hasattr(mesh, 'deduplicate_faces'):
            mesh.deduplicate_faces()
        if hasattr(mesh, 'remove_zero_area_faces'):
            try:
                mesh.remove_zero_area_faces()
            except Exception:
                pass
        if hasattr(mesh, 'fix_face_normals'):
            try:
                mesh.fix_face_normals()
            except Exception:
                pass
        return mesh
    except Exception:
        return None


def compute_metrics(path: str, mesh: trimesh.Trimesh) -> Dict[str, Any]:
    file_size_bytes = None
    try:
        file_size_bytes = os.path.getsize(path)
    except Exception:
        pass

    num_vertices = int(getattr(mesh, 'vertices', np.empty((0, 3))).shape[0])
    num_faces = int(getattr(mesh, 'faces', np.empty((0, 3))).shape[0]) if hasattr(mesh, 'faces') else 0

    # bounding box metrics
    bbox = mesh.bounds if hasattr(mesh, 'bounds') else None
    if bbox is not None and bbox.shape == (2, 3):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox
        extent_x = float(max_x - min_x)
        extent_y = float(max_y - min_y)
        extent_z = float(max_z - min_z)
        bbox_diag = float(np.linalg.norm([extent_x, extent_y, extent_z]))
    else:
        extent_x = extent_y = extent_z = bbox_diag = np.nan

    # surface area and volume when available (volume may require watertight)
    try:
        surface_area = float(mesh.area) if hasattr(mesh, 'area') else np.nan
    except Exception:
        surface_area = np.nan
    try:
        volume = float(mesh.volume) if getattr(mesh, 'is_watertight', False) else np.nan
    except Exception:
        volume = np.nan

    return {
        "file_path": path,
        "ext": os.path.splitext(path)[1].lower().lstrip('.'),
        "file_size_bytes": file_size_bytes,
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "is_watertight": bool(getattr(mesh, 'is_watertight', False)),
        "extent_x": extent_x,
        "extent_y": extent_y,
        "extent_z": extent_z,
        "bbox_diag": bbox_diag,
        "surface_area": surface_area,
        "volume": volume,
    }


def analyze_path(path: str) -> Dict[str, Any] | None:
    """Worker function: load mesh and compute metrics for a single path."""
    mesh = load_mesh(path)
    if mesh is None:
        return None
    try:
        return compute_metrics(path, mesh)
    except Exception:
        return None


def save_plots(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Histograms for counts and size
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_list = axes.ravel()
    columns = [
        ("num_vertices", "Vertices"),
        ("num_faces", "Faces"),
        ("bbox_diag", "BBox Diagonal"),
        ("extent_x", "Extent X"),
        ("extent_y", "Extent Y"),
        ("extent_z", "Extent Z"),
    ]
    for ax, (col, title) in zip(ax_list, columns):
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.set_title(f"{title} (no data)")
            ax.axis('off')
            continue
        ax.hist(series, bins=30, color="steelblue", alpha=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path_hist = os.path.join(out_dir, "histograms.png")
    plt.savefig(path_hist, dpi=200)
    plt.close(fig)

    # Means with error bars (std)
    summary_cols = ["num_vertices", "num_faces", "bbox_diag", "surface_area", "volume"]
    means = df[summary_cols].mean(numeric_only=True)
    stds = df[summary_cols].std(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary_cols))
    ax2.bar(x, means.values, yerr=stds.values, capsize=4, color="coral", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_', ' ').title() for c in summary_cols], rotation=20)
    ax2.set_title("Dataset Means ± Std")
    ax2.grid(True, axis='y', alpha=0.3)
    path_bar = os.path.join(out_dir, "means_std.png")
    plt.tight_layout()
    plt.savefig(path_bar, dpi=200)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Analyze 3D models in a folder and summarize to CSV and plots")
    parser.add_argument("input_dir", type=str, help="Root folder containing 3D models (recursive)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/analysis_3d", help="Directory to save CSV and plots")
    parser.add_argument("--exts", type=str, default="any", help="Comma-separated exts or 'any'")
    parser.add_argument("--max", type=int, default=0, help="Max files to analyze (0 for all)")
    parser.add_argument("--no-append-input-name", action="store_false", help="Append basename(input_dir) to output_dir")
    parser.add_argument("--parallel", "-p", type=int, default=max(1, (multiprocessing.cpu_count() or 1) // 2), help="Number of worker processes (>=1)")
    args = parser.parse_args()

    extensions = parse_exts(args.exts)
    files = collect_files(args.input_dir, extensions, None if args.max <= 0 else args.max)
    if not files:
        print(f"No files found under {args.input_dir} for exts={extensions}")
        return

    print(f"Analyzing {len(files)} files with {args.parallel} worker(s) ...")
    records: List[Dict[str, Any]] = []
    failed: List[str] = []
    if args.parallel and args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            future_to_path = {ex.submit(analyze_path, p): p for p in files}
            for fut in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Loading and measuring"):
                p = future_to_path[fut]
                try:
                    rec = fut.result()
                    if rec is None:
                        failed.append(p)
                    else:
                        records.append(rec)
                except Exception:
                    failed.append(p)
    else:
        for path in tqdm(files, desc="Loading and measuring"):
            rec = analyze_path(path)
            if rec is None:
                failed.append(path)
            else:
                records.append(rec)

    if not records:
        print("No valid meshes could be analyzed.")
        return

    out_dir = args.output_dir
    if args.no_append_input_name:
        base = os.path.basename(os.path.normpath(args.input_dir))
        out_dir = os.path.join(out_dir, base)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(out_dir, "dataset_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Summary CSV (pandas<1.5互換)
    try:
        summary = df.describe(include='all', datetime_is_numeric=True)
    except TypeError:
        summary = df.describe(include='all')
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved summary to {summary_path}")

    save_plots(df, out_dir)
    if failed:
        failed_path = os.path.join(out_dir, "failed_paths.txt")
        with open(failed_path, 'w') as f:
            f.write("\n".join(failed))
        print(f"{len(failed)} files failed. List saved to {failed_path}")


if __name__ == "__main__":
    main()


