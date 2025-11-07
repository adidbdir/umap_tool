import argparse
import os
import glob
import subprocess
from typing import List


DEFAULT_3D_EXTS: List[str] = ["obj", "glb", "gltf", "ply", "stl", "fbx", "dae"]


def parse_exts(value: str) -> List[str]:
    if value is None or value.strip().lower() in {"any", "*", "all", "3d", "all3d", ""}:
        return DEFAULT_3D_EXTS
    return sorted(set([p.strip().lower().lstrip('.') for p in value.split(',') if p.strip()]))


def collect_files(input_dir: str, extensions: List[str]) -> List[str]:
    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(input_dir, "**", f"*.{ext}"), recursive=True))
    # de-dup while preserving order
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def run_blender_render(blender_bin: str, src: str, out_dir: str, n_views: int) -> None:
    script = os.path.join(os.path.dirname(__file__), "render_obj_multi_views.py")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        blender_bin,
        "--background",
        "--python", script,
        "--",
        src,
        out_dir,
        str(n_views),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Recursively render thumbnails for 3D models in a folder")
    parser.add_argument("input_dir", type=str, help="Root folder to scan recursively")
    parser.add_argument("output_dir", type=str, help="Output folder for renders")
    parser.add_argument("--exts", type=str, default="any", help="Comma-separated exts or 'any'")
    parser.add_argument("--views", type=int, default=1, help="Number of views per model (1 for single thumb)")
    parser.add_argument("--blender", type=str, default="blender", help="Blender binary path")
    args = parser.parse_args()

    extensions = parse_exts(args.exts)
    files = collect_files(args.input_dir, extensions)
    if not files:
        print(f"No files found in {args.input_dir} for exts={extensions}")
        return

    print(f"Found {len(files)} files. Rendering with {args.views} view(s) each.")
    for i, src in enumerate(files, 1):
        # map source path into output subfolder mirroring the tree
        rel = os.path.relpath(src, args.input_dir)
        stem, _ = os.path.splitext(rel)
        out_dir = os.path.join(args.output_dir, stem)
        try:
            run_blender_render(args.blender, src, out_dir, args.views)
        except subprocess.CalledProcessError as e:
            print(f"Render failed for {src}: {e}")


if __name__ == "__main__":
    main()


