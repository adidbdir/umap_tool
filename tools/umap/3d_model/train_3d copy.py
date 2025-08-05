import os
import argparse
import glob
import numpy as np
import trimesh
from abc import ABC, abstractmethod
from typing import Tuple, List, Union
from umap.parametric_umap import ParametricUMAP

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.models import Model


class FeatureExtractor(ABC):
    """Abstract base class for 3D model feature extractors."""
    
    @abstractmethod
    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        """Extract features from a 3D mesh."""
        pass
    
    @abstractmethod
    def get_feature_dim(self, **kwargs) -> int:
        """Get the dimension of extracted features."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the feature extractor."""
        pass


class PointCloudExtractor(FeatureExtractor):
    """Point cloud sampling based feature extractor."""
    
    def __init__(self, n_points: int = 2048):
        self.n_points = n_points
    
    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        """Extract features by sampling points from mesh surface."""
        points = mesh.sample(self.n_points)
        return points.flatten()
    
    def get_feature_dim(self, **kwargs) -> int:
        return self.n_points * 3
    
    def get_name(self) -> str:
        return f"PointCloud_{self.n_points}"


class PointNetExtractor(FeatureExtractor):
    """PointNet-based feature extractor."""
    
    def __init__(self, n_points: int = 2048, feature_dim: int = 1024):
        self.n_points = n_points
        self.feature_dim = feature_dim
        self._pointnet_model = None
    
    def _build_pointnet_encoder(self) -> Model:
        """Build PointNet encoder architecture."""
        # Input: (batch_size, n_points, 3)
        input_points = Input(shape=(self.n_points, 3), name='point_cloud_input')
        
        # T-Net for input transformation (simplified version)
        # In full PointNet, this would be a transformation network
        x = input_points
        
        # Point-wise MLPs
        x = Conv1D(64, 1, activation='relu', name='conv1d_1')(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, 1, activation='relu', name='conv1d_2')(x)
        x = BatchNormalization()(x)
        
        # Feature transformation T-Net would go here in full implementation
        
        x = Conv1D(64, 1, activation='relu', name='conv1d_3')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 1, activation='relu', name='conv1d_4')(x)
        x = BatchNormalization()(x)
        x = Conv1D(self.feature_dim, 1, activation='relu', name='conv1d_5')(x)
        x = BatchNormalization()(x)
        
        # Global feature aggregation
        global_features = GlobalMaxPooling1D(name='global_max_pool')(x)
        
        model = Model(inputs=input_points, outputs=global_features, name='pointnet_encoder')
        return model
    
    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        """Extract features using PointNet architecture."""
        # Sample points from mesh
        points = mesh.sample(self.n_points)
        
        # Initialize PointNet model if not already done
        if self._pointnet_model is None:
            self._pointnet_model = self._build_pointnet_encoder()
        
        # Reshape for PointNet input: (1, n_points, 3)
        points_batch = points.reshape(1, self.n_points, 3)
        
        # Extract features
        features = self._pointnet_model.predict(points_batch, verbose=0)
        return features.flatten()
    
    def get_feature_dim(self, **kwargs) -> int:
        return self.feature_dim
    
    def get_name(self) -> str:
        return f"PointNet_{self.n_points}_{self.feature_dim}"


class EncoderFactory:
    """Factory class for creating different types of encoders."""
    
    @staticmethod
    def create_mlp_encoder(input_dim: int, n_components: int, hidden_dims: List[int] = None) -> Model:
        """Create a simple MLP encoder."""
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        input_layer = Input(shape=(input_dim,), name='mlp_input')
        x = input_layer
        
        for i, dim in enumerate(hidden_dims):
            x = Dense(dim, activation="relu", name=f'mlp_dense_{i+1}')(x)
            x = BatchNormalization(name=f'mlp_bn_{i+1}')(x)
            x = Dropout(0.2, name=f'mlp_dropout_{i+1}')(x)
        
        encoder_output = Dense(n_components, name='mlp_output')(x)
        return Model(input_layer, encoder_output, name='mlp_encoder')
    
    @staticmethod
    def create_pointnet_encoder(input_dim: int, n_components: int, n_points: int) -> Model:
        """Create a PointNet-based encoder."""
        # For PointNet, we expect the input to be already processed by PointNetExtractor
        # So we just need a final projection layer
        input_layer = Input(shape=(input_dim,), name='pointnet_features_input')
        x = Dense(512, activation="relu", name='pointnet_dense_1')(input_layer)
        x = BatchNormalization(name='pointnet_bn_1')(x)
        x = Dropout(0.3, name='pointnet_dropout_1')(x)
        
        x = Dense(256, activation="relu", name='pointnet_dense_2')(x)
        x = BatchNormalization(name='pointnet_bn_2')(x)
        x = Dropout(0.2, name='pointnet_dropout_2')(x)
        
        encoder_output = Dense(n_components, name='pointnet_output')(x)
        return Model(input_layer, encoder_output, name='pointnet_projection_encoder')


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
        help="Extension of 3D models to train"
    )
    parser.add_argument(
        "--feature-extractor", default="pointcloud", type=str, 
        choices=["pointcloud", "pointnet"],
        help="Feature extraction method to use"
    )
    parser.add_argument(
        "--pointnet-feature-dim", default=1024, type=int,
        help="Feature dimension for PointNet (only used with --feature-extractor=pointnet)"
    )
    parser.add_argument(
        "--encoder-type", default="mlp", type=str,
        choices=["mlp", "pointnet"],
        help="Type of encoder to use for UMAP"
    )
    return parser.parse_args()


def create_feature_extractor(extractor_type: str, n_points: int, **kwargs) -> FeatureExtractor:
    """Factory function to create feature extractors."""
    if extractor_type == "pointcloud":
        return PointCloudExtractor(n_points=n_points)
    elif extractor_type == "pointnet":
        feature_dim = kwargs.get("pointnet_feature_dim", 1024)
        return PointNetExtractor(n_points=n_points, feature_dim=feature_dim)
    else:
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

        # Fix non-watertight meshes
        if len(mesh.faces) > 0 and not mesh.is_watertight:
            print(f"Warning: Mesh {mesh_path} is not watertight. Filling holes.")
            mesh.fill_holes()
        
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
    paths = glob.glob(path_pattern, recursive=True)
    if not paths:
        raise ValueError(f"No files found for pattern: {path_pattern}")
        
    data_list = []
    processed_count = 0
    
    print(f"Processing {len(paths)} files using {feature_extractor.get_name()}...")
    
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
    
    print(f"Configuration:")
    print(f"  Feature Extractor: {args.feature_extractor}")
    print(f"  Encoder Type: {args.encoder_type}")
    print(f"  N Points: {args.n_points}")
    if args.feature_extractor == "pointnet":
        print(f"  PointNet Feature Dim: {args.pointnet_feature_dim}")

    # Create feature extractor
    feature_extractor = create_feature_extractor(
        args.feature_extractor, 
        args.n_points,
        pointnet_feature_dim=args.pointnet_feature_dim
    )

    # Extract features from all models
    data, _ = create_data_and_target(
        f"{args.input_dir}/**/*.{args.prefix}",
        feature_extractor
    )

    if data.shape[0] == 0:
        print("No data to train on. Exiting.")
        return

    # Get feature dimension and create encoder
    input_dim = feature_extractor.get_feature_dim()
    n_components = 2  # For UMAP, typically 2 or 3 for visualization
    
    print(f"Input dimension: {input_dim}")
    print(f"Data shape: {data.shape}")

    # Create encoder based on specified type
    if args.encoder_type == "mlp":
        encoder = EncoderFactory.create_mlp_encoder(input_dim, n_components)
    elif args.encoder_type == "pointnet":
        encoder = EncoderFactory.create_pointnet_encoder(input_dim, n_components, args.n_points)
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder_type}")

    print(f"Encoder architecture:")
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
        "batch_size": 8,  # Keep batch_size, remove epochs to avoid conflict
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
    model_name = f"{feature_extractor.get_name()}_{args.encoder_type}"
    output_path = os.path.join(args.output_dir, model_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    mapper.save(output_path)
    
    # Save configuration
    config = {
        "feature_extractor": args.feature_extractor,
        "encoder_type": args.encoder_type,
        "n_points": args.n_points,
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