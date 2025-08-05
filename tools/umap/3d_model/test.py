import argparse
import multiprocessing
import glob
import numpy as np
import tqdm
import os
import trimesh  # Added for 3D model loading

from umap.parametric_umap import load_ParametricUMAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        default="outputs/train_3d/test",  # Changed default
        type=str,
        help="Directory of the trained 3D UMAP model",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="data/test_3d",
        type=str,
        help="Directory of 3D models to test",  # Changed default
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="outputs/test_3d/test.csv",  # Changed default
        type=str,
        help="Output path for the embedding CSV file",
    )
    parser.add_argument(
        "-n",
        "--n-points",
        default=2048,
        type=int,
        help="Number of points to sample from each 3D model",  # Added
    )
    parser.add_argument(
        "--max", default="10000", type=int, help="Maximum number of 3D models to test"
    )
    parser.add_argument(
        "-p", "--parallel", default="1", type=int, help="Number of parallel processes"
    )
    parser.add_argument(
        "--prefix", default="obj", type=str, help="Extension of 3D models to test"
    )  # Changed default
    return parser.parse_args()


def create_embedding_and_target(
    proc_num,
    model_dir,
    paths,
    n_points,  # Changed from size to n_points
    embedding_dict,
    path_dict,
):
    mapper = load_ParametricUMAP(model_dir)

    embedding_list = []
    p_list = []
    for p_str in tqdm.tqdm(paths, desc=f"Process {proc_num}"):
        try:
            mesh = trimesh.load_mesh(p_str)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            if (
                not hasattr(mesh, "sample")
                or not hasattr(mesh, "vertices")
                or not hasattr(mesh, "faces")
            ):
                print(
                    f"Warning (Proc {proc_num}): Could not load or process mesh {p_str} as a valid Trimesh object. Skipping."
                )
                continue

            if len(mesh.faces) > 0 and not mesh.is_watertight:
                # print(f"Warning (Proc {proc_num}): Mesh {p_str} is not watertight. Attempting to fill holes.")
                mesh.fill_holes()

            if len(mesh.vertices) == 0:
                print(
                    f"Warning (Proc {proc_num}): Mesh {p_str} has no vertices after processing. Skipping."
                )
                continue

            points = mesh.sample(n_points)
            if points.shape[0] == 0:
                print(
                    f"Warning (Proc {proc_num}): Mesh {p_str} resulted in 0 sampled points. Skipping."
                )
                continue

            data_for_transform = points.flatten().reshape(1, -1)
            _embedding = mapper.transform(data_for_transform)
            embedding_list.append(_embedding)
            p_list.append(p_str)
        except Exception as e:
            print(f"Error (Proc {proc_num}) processing file {p_str}: {e}. Skipping.")
            continue

    if embedding_list:
        embedding_dict[proc_num] = np.concatenate(embedding_list, axis=0)
    else:
        embedding_dict[proc_num] = np.array(
            []
        )  # Handle case with no successful embeddings
    path_dict[proc_num] = p_list


def run(paths, model_dir, n_points, parallel):  # Changed size to n_points
    manager = multiprocessing.Manager()
    embedding_dict = manager.dict()
    path_dict = manager.dict()  # Though not used in the return, kept for consistency
    jobs = []
    div_paths = (
        np.array_split(paths, parallel) if parallel > 0 and len(paths) > 0 else []
    )

    for i in range(parallel):
        if len(div_paths[i]) == 0:  # Skip if no paths for this process
            continue
        p = multiprocessing.Process(
            target=create_embedding_and_target,
            args=(i, model_dir, div_paths[i], n_points, embedding_dict, path_dict),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    final_embedding_list = []
    for i in range(parallel):
        if i in embedding_dict and embedding_dict[i].size > 0:
            final_embedding_list.append(embedding_dict[i])

    if not final_embedding_list:
        return np.array([])  # Return empty array if no embeddings were generated

    return np.concatenate(final_embedding_list, axis=0)


def main():
    args = parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Glob for 3D model files recursively
    path_pattern = os.path.join(args.input_dir, "**", f"*.{args.prefix}")
    paths = glob.glob(path_pattern, recursive=True)

    if not paths:
        print(f"No files found for pattern: {path_pattern}")
        return

    sorted_path = sorted(paths)[: args.max]
    print(f"Found {len(sorted_path)} files to process.")

    embedding = run(sorted_path, args.model_dir, args.n_points, args.parallel)

    if embedding.size == 0:
        print("No embeddings were generated. Output file will not be saved.")
        return

    np.savetxt(args.output_path, embedding, delimiter=",")
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
