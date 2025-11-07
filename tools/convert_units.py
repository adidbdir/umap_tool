import argparse
import os
import glob
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

import numpy as np
import trimesh


DEFAULT_3D_EXTS: List[str] = ["obj", "glb", "gltf", "ply", "stl", "fbx", "dae"]


def parse_exts(value: str) -> List[str]:
    if value is None or value.strip().lower() in {"any", "*", "all", "3d", "all3d", ""}:
        return DEFAULT_3D_EXTS
    return sorted(set([p.strip().lower().lstrip('.') for p in value.split(',') if p.strip()]))


def collect_files(input_dir: str, extensions: List[str]) -> List[str]:
    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(input_dir, "**", f"*.{ext}"), recursive=True))
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def load_mesh(path: str) -> trimesh.Trimesh | None:
    try:
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception:
        return None


def export_mesh(mesh: trimesh.Trimesh, out_path: str) -> bool:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        mesh.export(out_path)
        return True
    except Exception:
        # Fallback: export as OBJ when native exporter is unavailable
        try:
            alt = os.path.splitext(out_path)[0] + ".obj"
            mesh.export(alt)
            return True
        except Exception:
            return False


def convert_one(src: str, input_root: str, output_root: str, scale: float) -> Tuple[str, bool, str]:
    rel = os.path.relpath(src, input_root)
    out_path = os.path.join(output_root, rel)
    mesh = load_mesh(src)
    if mesh is None:
        return (src, False, "load_failed")
    try:
        if scale != 1.0:
            mesh.apply_scale(scale)
        ok = export_mesh(mesh, out_path)
        return (src, ok, "ok" if ok else "export_failed")
    except Exception:
        return (src, False, "exception")


def unit_to_scale(from_unit: str) -> float:
    key = (from_unit or "m").strip().lower()
    if key in {"m", "meter", "metre", "meters"}:
        return 1.0
    if key in {"cm", "centimeter", "centimetre", "centimeters"}:
        return 0.01
    if key in {"mm", "millimeter", "millimetre", "millimeters"}:
        return 0.001
    raise ValueError(f"Unsupported unit: {from_unit}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D models under a folder to meters by scaling coordinates"
    )
    parser.add_argument("input_dir", type=str, help="Root folder (recursive)")
    parser.add_argument("output_dir", type=str, help="Output folder root")
    parser.add_argument("--exts", type=str, default="any", help="Comma-separated exts or 'any'")
    parser.add_argument(
        "--from-unit",
        type=str,
        default="cm",
        choices=["m", "cm", "mm"],
        help="Source unit of the coordinates. Converted to meters.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override scale factor to multiply vertex coords (e.g., 0.01 for cm->m)",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=max(1, (multiprocessing.cpu_count() or 1) // 2),
        help="Number of worker processes (>=1)",
    )
    args = parser.parse_args()

    exts = parse_exts(args.exts)
    files = collect_files(args.input_dir, exts)
    if not files:
        print(f"No files found under {args.input_dir} for exts={exts}")
        return

    scale = args.scale if args.scale is not None else unit_to_scale(args.from_unit)
    if scale <= 0:
        raise ValueError("Scale must be positive")

    print(
        f"Converting {len(files)} files to meters with scale={scale} using {args.parallel} worker(s)..."
    )

    results: List[Tuple[str, bool, str]] = []
    if args.parallel and args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            fut_map = {
                ex.submit(convert_one, src, args.input_dir, args.output_dir, scale): src
                for src in files
            }
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Converting"):
                try:
                    results.append(fut.result())
                except Exception:
                    results.append((fut_map[fut], False, "exception"))
    else:
        for src in tqdm(files, desc="Converting"):
            results.append(convert_one(src, args.input_dir, args.output_dir, scale))

    ok = sum(1 for _, s, _ in results if s)
    ng = len(results) - ok
    print(f"Done. success={ok} failed={ng}")
    if ng:
        failed = [src for src, s, _ in results if not s]
        out = os.path.join(args.output_dir, "failed_paths.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out, "w") as f:
            f.write("\n".join(failed))
        print(f"Failed paths written to {out}")


if __name__ == "__main__":
    main()


