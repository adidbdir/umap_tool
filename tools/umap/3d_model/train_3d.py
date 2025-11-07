import os

# GPU設定を調整
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# XLA無効化（StatelessShuffle未対応エラー対策）
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

from feature_extractors import FeatureExtractor, FeatureExtractorRegistry
from config import load_config, apply_overrides, validate_config

# Other imports
import argparse
import glob
import numpy as np
import trimesh
from typing import Tuple, List, Union
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
try:
    # Avoid graph/XLA compilation path entirely
    tf.config.run_functions_eagerly(True)
except Exception:
    pass


class EncoderFactory:
    """Factory class for creating different types of encoders."""
    
    @staticmethod
    def create_mlp_encoder(input_dim: int, n_components: int, hidden_dims: List[int] = None) -> tf.keras.Model:
        """Create a simple MLP encoder."""
        from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout

        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        input_layer = Input(shape=(input_dim,), name='mlp_input')
        x = input_layer
        
        for i, dim in enumerate(hidden_dims):
            x = Dense(dim, activation="relu", name=f'mlp_dense_{i+1}')(x)
            x = BatchNormalization(name=f'mlp_bn_{i+1}')(x)
            x = Dropout(0.2, name=f'mlp_dropout_{i+1}')(x)
        
        encoder_output = Dense(n_components, name='mlp_output')(x)
        return tf.keras.Model(input_layer, encoder_output, name='mlp_encoder')
    
    @staticmethod
    def create_pointnet_encoder(input_dim: int, n_components: int, n_points: int) -> tf.keras.Model:
        """Create a PointNet-based encoder."""
        # For PointNet, we expect the input to be already processed by PointNetExtractor
        # So we just need a final projection layer
        from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout

        input_layer = Input(shape=(input_dim,), name='pointnet_features_input')
        x = Dense(512, activation="relu", name='pointnet_dense_1')(input_layer)
        x = BatchNormalization(name='pointnet_bn_1')(x)
        x = Dropout(0.3, name='pointnet_dropout_1')(x)
        
        x = Dense(256, activation="relu", name='pointnet_dense_2')(x)
        x = BatchNormalization(name='pointnet_bn_2')(x)
        x = Dropout(0.2, name='pointnet_dropout_2')(x)
        
        encoder_output = Dense(n_components, name='pointnet_output')(x)
        return tf.keras.Model(input_layer, encoder_output, name='pointnet_projection_encoder')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", default="data/test_3d", type=str, 
        help="Directory of 3D models to train"
    )
    parser.add_argument(
        "-o", "--output-dir", default="outputs/train_3d/test", type=str,
        help="Directory to save training parameters"
    )
    parser.add_argument(
        "-n", "--n-points", default=2048, type=int, 
        help="Number of points to sample from each 3D model"
    )
    parser.add_argument(
        "-p", "--prefix", default="obj", type=str, 
        help="Extension of 3D models to train (e.g., obj, glb, ply, stl)"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--set", type=str, nargs='*', default=[], help="Override config (key=value, dotted path)")
    # Backward-compatible flags (optional). Users should prefer config file + --set
    parser.add_argument("--feature-extractor", default=None, type=str, choices=["pointcloud", "pointnet", "ulip", "llava"], help="Override: extractor type")
    parser.add_argument("--pointnet-feature-dim", default=None, type=int, help="Override: PointNet feature dim")
    parser.add_argument("--ulip-ckpt-path", type=str, default=None, help="Override: ULIP checkpoint path")
    parser.add_argument("--ulip-model", type=str, default=None, help="Override: ULIP model name")
    parser.add_argument("--llava-model-path", type=str, default=None, help="Override: LLaVA model path")
    parser.add_argument("--llava-model-name", type=str, default=None, help="Override: LLaVA model name")
    parser.add_argument("--llava-n-views", type=int, default=None, help="Override: LLaVA n views")
    parser.add_argument("--blender-bin", type=str, default=None, help="Override: Blender binary")
    parser.add_argument(
        "--encoder-type", default="mlp", type=str,
        choices=["mlp", "pointnet"],
        help="Type of encoder to use for UMAP"
    )
    return parser.parse_args()


def create_feature_extractor(extractor_type: str, n_points: int, **kwargs) -> FeatureExtractor:
    """Create feature extractor via registry with normalized kwargs."""
    extractor_key = extractor_type.lower().strip()
    if extractor_key == "pointcloud":
        return FeatureExtractorRegistry.create("pointcloud", n_points=n_points)
    if extractor_key == "pointnet":
        feature_dim = kwargs.get("pointnet_feature_dim", 1024)
        return FeatureExtractorRegistry.create("pointnet", n_points=n_points, feature_dim=feature_dim)
    if extractor_key == "ulip":
        ckpt_path = kwargs.get("ulip_ckpt_path")
        if ckpt_path is None:
            raise ValueError("--ulip-ckpt-path is required when using --feature-extractor=ulip")
        model_name = kwargs.get("ulip_model", "ULIP_PointBERT")
        batch_size = kwargs.get("batch_size", 1)
        return FeatureExtractorRegistry.create(
            "ulip",
            ckpt_path=ckpt_path,
            model_name=model_name,
            n_points=n_points,
            batch_size=batch_size,
        )
    if extractor_key == "llava":
        # LLaVA can fallback to OpenCLIP if model path is not provided
        return FeatureExtractorRegistry.create(
            "llava",
            model_path=kwargs.get("llava_model_path"),
            model_name=kwargs.get("llava_model_name", "llava-v1.5-7b"),
            n_views=kwargs.get("llava_n_views", 8),
            blender_path=kwargs.get("blender_bin", "blender"),
        )
    raise ValueError(f"Unknown feature extractor type: {extractor_type}")


def load_and_process_mesh(mesh_path: str) -> Union[trimesh.Trimesh, None]:
    """Load and preprocess a 3D mesh file."""
    try:
        mesh = trimesh.load_mesh(mesh_path)
        
        # Convert scene to single mesh if necessary
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # Validate mesh
        if not hasattr(mesh, 'sample') or not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"Warning: Could not load or process mesh {mesh_path} as a valid Trimesh object. Skipping.")
            return None

        # trimesh 4.6.12対応のメッシュ修復
        if len(mesh.faces) > 0:
            # 重複した頂点を削除
            if hasattr(mesh, 'deduplicate_vertices'):
                mesh.deduplicate_vertices()
            
            # 重複した面を削除
            if hasattr(mesh, 'deduplicate_faces'):
                mesh.deduplicate_faces()
            
            # 無効な面を削除
            if hasattr(mesh, 'remove_zero_area_faces'):
                mesh.remove_zero_area_faces()
            
            # 穴を埋める
            if not mesh.is_watertight:
                print(f"Warning: Mesh {mesh_path} is not watertight. Filling holes.")
                if hasattr(mesh, 'fill_holes'):
                    mesh.fill_holes()
                
            # メッシュを正規化
            if hasattr(mesh, 'fix_face_normals'):
                mesh.fix_face_normals()
        
        # Check if mesh has vertices after processing
        if len(mesh.vertices) == 0:
            print(f"Warning: Mesh {mesh_path} has no vertices after processing. Skipping.")
            return None
            
        return mesh
        
    except Exception as e:
        print(f"Error processing file {mesh_path}: {e}. Skipping.")
        return None


def create_data_and_target(path_pattern: str, feature_extractor: FeatureExtractor) -> Tuple[np.ndarray, List[int]]:
    """Create training data using the specified feature extractor."""
    # 複数のファイル拡張子に対応
    if path_pattern.endswith('/**/*.obj'):
        base_pattern = path_pattern.replace('/**/*.obj', '')
        # サポートする3Dファイル形式
        supported_extensions = ['obj', 'glb', 'gltf', 'ply', 'stl', 'fbx', 'dae']
        paths = []
        for ext in supported_extensions:
            ext_pattern = f"{base_pattern}/**/*.{ext}"
            ext_paths = glob.glob(ext_pattern, recursive=True)
            paths.extend(ext_paths)
    else:
        # 特定の拡張子が指定されていない場合は、すべてのサポート形式を検索
        base_pattern = path_pattern.replace('/**/*.obj', '').replace('/**/*.glb', '').replace('/**/*.gltf', '').replace('/**/*.ply', '').replace('/**/*.stl', '').replace('/**/*.fbx', '').replace('/**/*.dae', '')
        if base_pattern == path_pattern:
            # パターンが変更されていない場合（特定の拡張子が指定されていない）
            supported_extensions = ['obj', 'glb', 'gltf', 'ply', 'stl', 'fbx', 'dae']
            paths = []
            for ext in supported_extensions:
                ext_pattern = f"{base_pattern}/**/*.{ext}"
                ext_paths = glob.glob(ext_pattern, recursive=True)
                paths.extend(ext_paths)
        else:
            paths = glob.glob(path_pattern, recursive=True)
    
    if not paths:
        raise ValueError(f"No files found for pattern: {path_pattern}")
        
    data_list = []
    processed_count = 0
    
    print(f"Processing {len(paths)} files using {feature_extractor.get_name()}...")
    print(f"Found files with extensions: {list(set([os.path.splitext(p)[1] for p in paths]))}")
    
    for i, mesh_path in enumerate(paths):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(paths)} files processed")
            
        mesh = load_and_process_mesh(mesh_path)
        if mesh is None:
            continue
            
        try:
            features = feature_extractor.extract_features(mesh)
            data_list.append(features.reshape(1, -1))
            processed_count += 1
        except Exception as e:
            print(f"Error extracting features from {mesh_path}: {e}. Skipping.")
            continue
    
    if not data_list:
        raise ValueError("No valid 3D models could be processed.")

    print(f"Successfully processed {processed_count}/{len(paths)} files")
    
    data = np.concatenate(data_list, axis=0)
    target = [0 for _ in range(data.shape[0])]  # Placeholder target
    return data, target


def main():
    args = parse_args()

    # Build config
    cfg = {
        "input": {
            "dir": args.input_dir,
            "prefix": args.prefix,
            "n_points": args.n_points,
        },
        "output": {"dir": args.output_dir},
        "umap": {"encoder_type": "mlp", "n_components": 2},
        "extractor": {
            "type": "pointcloud",
            "pointnet": {"feature_dim": 1024},
            "ulip": {"ckpt_path": "external/ULIP/weights/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt", "model": "ULIP_PointBERT"},
            "llava": {"model_path": None, "model_name": "llava-v1.5-7b", "n_views": 8, "blender_bin": "blender"},
        },
    }

    if args.config:
        cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    validate_config(cfg)

    # Backward-compatible overrides
    if args.feature_extractor:
        cfg["extractor"]["type"] = args.feature_extractor
    if args.pointnet_feature_dim is not None:
        cfg["extractor"]["pointnet"]["feature_dim"] = args.pointnet_feature_dim
    if args.ulip_ckpt_path:
        cfg["extractor"]["ulip"]["ckpt_path"] = args.ulip_ckpt_path
    if args.ulip_model:
        cfg["extractor"]["ulip"]["model"] = args.ulip_model
    if args.llava_model_path:
        cfg["extractor"]["llava"]["model_path"] = args.llava_model_path
    if args.llava_model_name:
        cfg["extractor"]["llava"]["model_name"] = args.llava_model_name
    if args.llava_n_views is not None:
        cfg["extractor"]["llava"]["n_views"] = args.llava_n_views
    if args.blender_bin:
        cfg["extractor"]["llava"]["blender_bin"] = args.blender_bin
    
    print("Configuration:")
    print(f"  Feature Extractor: {cfg['extractor']['type']}")
    print(f"  Encoder Type: {cfg['umap']['encoder_type']}")
    print(f"  N Points: {cfg['input']['n_points']}")
    print(f"  File Extension: {cfg['input']['prefix']}")
    if cfg['extractor']['type'] == "pointnet":
        print(f"  PointNet Feature Dim: {cfg['extractor']['pointnet']['feature_dim']}")
    if cfg['extractor']['type'] == "ulip":
        print(f"  ULIP Checkpoint Path: {cfg['extractor']['ulip']['ckpt_path']}")
        print(f"  ULIP Model Name: {cfg['extractor']['ulip']['model']}")
    if cfg['extractor']['type'] == "llava":
        print(f"  LLaVA Model Path: {cfg['extractor']['llava']['model_path']}")
        print(f"  LLaVA Model Name: {cfg['extractor']['llava']['model_name']}")
        print(f"  LLaVA N Views: {cfg['extractor']['llava']['n_views']}")

    # Create feature extractor
    feature_extractor = create_feature_extractor(
        cfg['extractor']['type'], 
        cfg['input']['n_points'],
        pointnet_feature_dim=cfg['extractor']['pointnet']['feature_dim'],
        ulip_ckpt_path=cfg['extractor']['ulip']['ckpt_path'],
        ulip_model=cfg['extractor']['ulip']['model'],
        llava_model_path=cfg['extractor']['llava']['model_path'],
        llava_model_name=cfg['extractor']['llava']['model_name'],
        llava_n_views=cfg['extractor']['llava']['n_views'],
        blender_bin=cfg['extractor']['llava']['blender_bin'],
    )

    # Extract features from all models with LLaVA support (pass source_path)
    # Recursively collect 3D files from all subdirectories.
    supported_extensions = ['obj', 'glb', 'gltf', 'ply', 'stl', 'fbx', 'dae']
    prefix = (cfg['input']['prefix'] or '').lower()
    exts = supported_extensions if prefix in ['', 'any', '*', 'all'] else [prefix]

    paths: List[str] = []
    for ext in exts:
        pattern = f"{cfg['input']['dir']}/**/*.{ext}"
        paths.extend(glob.glob(pattern, recursive=True))
    # De-duplicate while preserving order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if not paths:
        raise ValueError(
            f"No files found. dir={cfg['input']['dir']} extensions={exts}"
        )

    data_list = []
    for i, mesh_path in enumerate(paths):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(paths)} files processed")
        mesh = load_and_process_mesh(mesh_path)
        if mesh is None:
            continue
        kwargs = {}
        if cfg['extractor']['type'] == 'llava':
            kwargs['source_path'] = mesh_path
        try:
            features = feature_extractor.extract_features(mesh, **kwargs)
            data_list.append(features.reshape(1, -1))
        except Exception as e:
            print(f"Error extracting features from {mesh_path}: {e}. Skipping.")
            continue
    if not data_list:
        raise ValueError("No valid 3D models could be processed.")
    data = np.concatenate(data_list, axis=0)

    if data.shape[0] == 0:
        print("No data to train on. Exiting.")
        return

    # Get feature dimension and create encoder
    input_dim = feature_extractor.get_feature_dim()
    n_components = int(cfg['umap'].get('n_components', 2))  # 2 or 3 for visualization
    
    print(f"Input dimension: {input_dim}")
    print(f"Data shape: {data.shape}")
    print(f"UMAP n_components: {n_components}")

    # Create encoder based on specified type
    if cfg['umap']['encoder_type'] == "mlp":
        encoder = EncoderFactory.create_mlp_encoder(input_dim, n_components)
    elif cfg['umap']['encoder_type'] == "pointnet":
        encoder = EncoderFactory.create_pointnet_encoder(input_dim, n_components, args.n_points)
    else:
        raise ValueError(f"Unknown encoder type: {cfg['umap']['encoder_type']}")

    print("Encoder architecture:")
    encoder.summary()

    # Setup training parameters
    keras_fit_kwargs = {
        "callbacks": [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=10 ** -3,
                patience=15,
                verbose=1,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ],
        "batch_size": 1,  # バッチサイズを1に変更
        "shuffle": False,  # datasetシャッフル由来のstateless_shuffle回避の一助
    }

    # Create and train UMAP model
    mapper = ParametricUMAP(
        encoder=encoder,
        dims=(input_dim,),
        n_components=n_components,
        verbose=True,
        autoencoder_loss=True,
        keras_fit_kwargs=keras_fit_kwargs,
    )
    
    print(f"Fitting UMAP on data with shape: {data.shape}")
    mapper.fit_transform(data)

    # Save model with descriptive name
    model_name = f"{feature_extractor.get_name()}_{cfg['umap']['encoder_type']}"
    output_path = os.path.join(cfg['output']['dir'], model_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    mapper.save(output_path)
    
    # Save configuration (for test pipeline auto-detection)
    config = {
        "feature_extractor": cfg['extractor']['type'],
        "n_points": cfg['input']['n_points'],
        "pointnet_feature_dim": cfg['extractor']['pointnet']['feature_dim'],
        "umap": cfg['umap'],
        "input": cfg['input'],
        "output": cfg['output'],
        "extractor": cfg['extractor'],
        "input_dim": input_dim,
        "n_components": n_components,
        "model_name": model_name
    }
    
    import json
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved UMAP model to {output_path}")
    print(f"Model configuration saved to {output_path}/config.json")


if __name__ == "__main__":
    if len(tf.config.list_physical_devices("GPU")) == 0:
        print("WARNING!!! CPU mode!!!")
    main() 