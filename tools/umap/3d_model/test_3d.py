import argparse
import multiprocessing
import glob
import numpy as np
import tqdm
import os
import trimesh
import pandas as pd
import json
from typing import Tuple, List, Union
import hashlib
from umap.parametric_umap import load_ParametricUMAP

# Supported 3D model extensions for wildcard selection
DEFAULT_3D_EXTENSIONS: List[str] = [
    "obj", "glb", "gltf", "ply", "stl", "fbx", "dae"
]


def parse_extensions_arg(prefix: str) -> List[str]:
    """Parse --prefix argument into a list of extensions.

    - Accepts comma-separated values: "obj,ply,stl"
    - Accepts wildcards: any/*/all/3d/all3d -> DEFAULT_3D_EXTENSIONS
    - Normalizes by stripping spaces, lowercasing, and removing leading dots
    """
    if prefix is None:
        return DEFAULT_3D_EXTENSIONS
    value = prefix.strip().lower()
    if value in {"any", "*", "all", "3d", "all3d"} or value == "":
        return DEFAULT_3D_EXTENSIONS
    parts = [p.strip().lower().lstrip('.') for p in value.split(',') if p.strip()]
    return sorted(set(parts)) if parts else DEFAULT_3D_EXTENSIONS

# Import the feature extractors from train_3d.py
from feature_extractors import FeatureExtractor
from train_3d import create_feature_extractor, load_and_process_mesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory of the trained 3D UMAP model",
    )
    parser.add_argument(
        "-i", "--input-dirs", nargs='+', default=["data/test_3d"], 
        help="Directories of 3D models to test (can specify multiple directories)"
    )
    parser.add_argument(
        "--folder-labels", nargs='*', default=None,
        help="Custom labels for each input directory (optional, defaults to folder names)"
    )
    parser.add_argument(
        "-o", "--output-path", default="outputs/test_3d/embeddings_with_paths.csv", type=str,
        help="Output path for the CSV file containing embeddings and model paths",
    )
    parser.add_argument(
        "--max", default=10000, type=int, 
        help="Maximum number of 3D models to test per directory"
    )
    parser.add_argument(
        "-p", "--parallel", default=1, type=int, 
        help="Number of parallel processes"
    )
    parser.add_argument(
        "--prefix", default="obj", type=str,
        help="File extension(s) to test (e.g., 'obj' or 'obj,ply,stl' or 'any')"
    )
    # Feature extraction parameters (will be loaded from config if available)
    parser.add_argument(
        "--feature-extractor", default=None, type=str,
        choices=["pointcloud", "pointnet", "ulip", "llava"],
        help="Feature extraction method (auto-detected from model config if not specified)"
    )
    parser.add_argument(
        "--n-points", default=None, type=int,
        help="Number of points to sample (auto-detected from model config if not specified)"
    )
    parser.add_argument(
        "--pointnet-feature-dim", default=None, type=int,
        help="PointNet feature dimension (auto-detected from model config if not specified)"
    )
    parser.add_argument(
        "--seed", default=None, type=int,
        help="Global seed for deterministic sampling. Per-file seed is derived from seed^hash(path)."
    )
    return parser.parse_args()


def load_model_config(model_dir: str) -> dict:
    """Load model configuration from config.json if available."""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: No config.json found in {model_dir}. Using default parameters.")
        return {
            "feature_extractor": "pointcloud",
            "n_points": 2048,
            "pointnet_feature_dim": 1024
        }


def create_embedding_and_target(
    proc_num: int,
    model_dir: str,
    paths_with_labels: List[Tuple[str, str]],  # (path, label) tuples
    feature_extractor: FeatureExtractor,
    embedding_dict: dict,
    path_dict: dict,
    label_dict: dict,  # New: store labels corresponding to embeddings
    base_seed: int | None = None,
):
    """Process paths and create embeddings using specified feature extractor."""
    mapper = load_ParametricUMAP(model_dir)

    embedding_list = []
    processed_paths_list = []
    processed_labels_list = []
    
    for p_str, label in tqdm.tqdm(paths_with_labels, desc=f"Process {proc_num}"):
        mesh = load_and_process_mesh(p_str)
        if mesh is None:
            continue

        try:
            # Extract features using the specified feature extractor
            # Pass source_path so LLaVAExtractor can render multi-view images
            per_file_seed = None
            if base_seed is not None:
                # Stable 32-bit hash from path (independent of PYTHONHASHSEED)
                h = int(hashlib.md5(p_str.encode('utf-8'), usedforsecurity=False).hexdigest()[:8], 16)
                per_file_seed = base_seed ^ (h & 0xFFFFFFFF)
            features = feature_extractor.extract_features(mesh, source_path=p_str, seed=per_file_seed)
            
            # Reshape for UMAP transform
            data_for_transform = features.reshape(1, -1)
            _embedding = mapper.transform(data_for_transform)
            
            embedding_list.append(_embedding)
            processed_paths_list.append(p_str)
            processed_labels_list.append(label)
            
        except Exception as e:
            print(f"Error (Proc {proc_num}) processing file {p_str}: {e}. Skipping.")
            continue

    if embedding_list:
        embedding_dict[proc_num] = np.concatenate(embedding_list, axis=0)
        path_dict[proc_num] = processed_paths_list
        label_dict[proc_num] = processed_labels_list
    else:
        embedding_dict[proc_num] = np.array([])
        path_dict[proc_num] = []
        label_dict[proc_num] = []


def collect_paths_from_directories(input_dirs: List[str], folder_labels: List[str], prefix: str, max_per_dir: int) -> List[Tuple[str, str]]:
    """Collect paths from multiple directories with their corresponding labels."""
    all_paths_with_labels = []
    extensions = parse_extensions_arg(prefix)
    
    for i, input_dir in enumerate(input_dirs):
        label = folder_labels[i] if i < len(folder_labels) else os.path.basename(input_dir.rstrip('/'))
        
        # Find files in this directory across selected extensions
        paths: List[str] = []
        for ext in extensions:
            path_pattern = os.path.join(input_dir, "**", f"*.{ext}")
            paths.extend(glob.glob(path_pattern, recursive=True))
        # De-duplicate while preserving order
        seen = set()
        paths = [p for p in paths if not (p in seen or seen.add(p))]
        
        if not paths:
            print(f"Warning: No files found in {input_dir} for extensions: {extensions}")
            continue
            
        # Limit number of files per directory
        sorted_paths = sorted(paths)[:max_per_dir]
        print(f"Found {len(sorted_paths)} files in {input_dir} (label: {label}, exts: {extensions})")
        
        # Add (path, label) tuples
        for path in sorted_paths:
            all_paths_with_labels.append((path, label))
    
    return all_paths_with_labels


def run(paths_with_labels: List[Tuple[str, str]], model_dir: str, feature_extractor: FeatureExtractor, parallel: int, base_seed: int | None) -> Tuple[np.ndarray, List[str], List[str]]:
    """Run embedding extraction with multiprocessing."""
    manager = multiprocessing.Manager()
    embedding_dict = manager.dict()
    path_dict = manager.dict()
    label_dict = manager.dict()
    jobs = []
    
    if parallel > 0 and len(paths_with_labels) > 0:
        div_paths_with_labels = np.array_split(paths_with_labels, parallel)
    else:
        div_paths_with_labels = []

    for i in range(parallel):
        if len(div_paths_with_labels[i]) == 0:
            continue
        p = multiprocessing.Process(
            target=create_embedding_and_target,
            args=(i, model_dir, div_paths_with_labels[i], feature_extractor, embedding_dict, path_dict, label_dict, base_seed),
        )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    final_embedding_list = []
    final_path_list = []
    final_label_list = []
    for i in range(parallel):
        if i in embedding_dict and embedding_dict[i].size > 0:
            final_embedding_list.append(embedding_dict[i])
            if i in path_dict and i in label_dict:
                final_path_list.extend(path_dict[i])
                final_label_list.extend(label_dict[i])

    if not final_embedding_list:
        return np.array([]), [], []

    return np.concatenate(final_embedding_list, axis=0), final_path_list, final_label_list


def main():
    args = parse_args()

    # Load model configuration
    config = load_model_config(args.model_dir)
    
    # Use command line arguments if provided, otherwise use config values
    feature_extractor_type = args.feature_extractor or config.get("feature_extractor", "pointcloud")
    n_points = args.n_points or config.get("n_points", 2048)
    pointnet_feature_dim = args.pointnet_feature_dim or config.get("pointnet_feature_dim", 1024)
    
    # Setup folder labels
    if args.folder_labels and len(args.folder_labels) != len(args.input_dirs):
        print(f"Warning: Number of folder labels ({len(args.folder_labels)}) does not match number of input directories ({len(args.input_dirs)}). Using directory names.")
        folder_labels = [os.path.basename(d.rstrip('/')) for d in args.input_dirs]
    elif args.folder_labels:
        folder_labels = args.folder_labels
    else:
        folder_labels = [os.path.basename(d.rstrip('/')) for d in args.input_dirs]
    
    print(f"Configuration:")
    print(f"  Model directory: {args.model_dir}")
    print(f"  Feature Extractor: {feature_extractor_type}")
    print(f"  N Points: {n_points}")
    print(f"  Input directories: {args.input_dirs}")
    print(f"  Folder labels: {folder_labels}")
    print(f"  Extensions: {parse_extensions_arg(args.prefix)}")
    if feature_extractor_type == "pointnet":
        print(f"  PointNet Feature Dim: {pointnet_feature_dim}")

    # Create feature extractor matching the trained model
    feature_extractor = create_feature_extractor(
        feature_extractor_type,
        n_points,
        pointnet_feature_dim=pointnet_feature_dim
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Collect paths from all directories with labels
    paths_with_labels = collect_paths_from_directories(
        args.input_dirs, folder_labels, args.prefix, args.max
    )

    if not paths_with_labels:
        print("No files found in any of the specified directories.")
        return

    print(f"Total files to process: {len(paths_with_labels)}")

    # Extract embeddings
    embeddings, processed_model_paths, processed_labels = run(
        paths_with_labels, args.model_dir, feature_extractor, args.parallel, args.seed
    )

    if embeddings.size == 0:
        print("No embeddings were generated. Output file will not be saved.")
        return

    # Validate data consistency
    if len(processed_model_paths) != embeddings.shape[0] or len(processed_labels) != embeddings.shape[0]:
        print(
            f"Warning: Mismatch between embeddings ({embeddings.shape[0]}), "
            f"paths ({len(processed_model_paths)}), and labels ({len(processed_labels)}). "
            f"Truncating to shortest length."
        )
        min_len = min(len(processed_model_paths), len(processed_labels), embeddings.shape[0])
        processed_model_paths = processed_model_paths[:min_len]
        processed_labels = processed_labels[:min_len]
        embeddings = embeddings[:min_len, :]

    # Create DataFrame and save to CSV
    df_data = {
        "model_path": processed_model_paths,
        "folder_label": processed_labels  # Add folder labels for color coding
    }
    for i in range(embeddings.shape[1]):
        df_data[f"umap_{i}"] = embeddings[:, i]

    df = pd.DataFrame(df_data)
    df.to_csv(args.output_path, index=False)
    
    # Print summary statistics
    print(f"Successfully processed {len(processed_model_paths)} models")
    print(f"Label distribution:")
    for label in set(processed_labels):
        count = processed_labels.count(label)
        print(f"  {label}: {count} models")
    
    print(f"Saved embeddings and model paths to {args.output_path}")
    
    # Save test configuration for reference
    test_config = {
        "model_dir": args.model_dir,
        "feature_extractor": feature_extractor_type,
        "n_points": n_points,
        "input_dirs": args.input_dirs,
        "folder_labels": folder_labels,
        "num_processed": len(processed_model_paths),
        "output_path": args.output_path,
        "label_counts": {label: processed_labels.count(label) for label in set(processed_labels)}
    }
    
    config_output_path = args.output_path.replace(".csv", "_config.json")
    with open(config_output_path, "w") as f:
        json.dump(test_config, f, indent=2)
    
    print(f"Saved test configuration to {config_output_path}")


if __name__ == "__main__":
    main()