import os
import sys
import glob
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Type, Optional

import numpy as np
import trimesh
import tempfile
import subprocess
from PIL import Image

import torch

# Public API of this module:
# - FeatureExtractor (ABC)
# - PointCloudExtractor, PointNetExtractor, ULIPExtractor (optional)
# - FeatureExtractorRegistry: register/get/create utilities


class FeatureExtractor(ABC):
    """Abstract base class for 3D model feature extractors."""

    @abstractmethod
    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        """Extract features from a 3D mesh and return a 1D numpy array."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_dim(self, **kwargs) -> int:
        """Return the dimensionality of extracted features."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable extractor name."""
        raise NotImplementedError


class PointCloudExtractor(FeatureExtractor):
    """Point cloud sampling based feature extractor."""

    def __init__(self, n_points: int = 2048):
        self.n_points = n_points

    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        # Deterministic sampling if seed provided
        seed = kwargs.get("seed")
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(int(seed) & 0xFFFFFFFF)
            try:
                points = mesh.sample(self.n_points)
            finally:
                np.random.set_state(state)
        else:
            points = mesh.sample(self.n_points)
        return points.flatten()

    def get_feature_dim(self, **kwargs) -> int:
        return self.n_points * 3

    def get_name(self) -> str:
        return f"PointCloud_{self.n_points}"


class PointNetExtractor(FeatureExtractor):
    """Lightweight PointNet-like encoder with Keras to get global features."""

    def __init__(self, n_points: int = 2048, feature_dim: int = 1024):
        from tensorflow.keras.layers import (
            Dense,
            Input,
            Conv1D,
            GlobalMaxPooling1D,
            BatchNormalization,
        )
        from tensorflow.keras.models import Model

        self.n_points = n_points
        self.feature_dim = feature_dim
        self._model: Optional[Model] = None
        # Lazy build; defer model construction until first use to keep imports light

    def _build_encoder(self):
        from tensorflow.keras.layers import (
            Dense,
            Input,
            Conv1D,
            GlobalMaxPooling1D,
            BatchNormalization,
        )
        from tensorflow.keras.models import Model

        input_points = Input(shape=(self.n_points, 3), name="point_cloud_input")
        x = input_points
        x = Conv1D(64, 1, activation="relu", name="conv1d_1")(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, 1, activation="relu", name="conv1d_2")(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, 1, activation="relu", name="conv1d_3")(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 1, activation="relu", name="conv1d_4")(x)
        x = BatchNormalization()(x)
        x = Conv1D(self.feature_dim, 1, activation="relu", name="conv1d_5")(x)
        x = BatchNormalization()(x)
        global_features = GlobalMaxPooling1D(name="global_max_pool")(x)
        self._model = Model(inputs=input_points, outputs=global_features, name="pointnet_encoder")

    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        # Deterministic sampling if seed provided
        seed = kwargs.get("seed")
        if seed is not None:
            state = np.random.get_state()
            np.random.seed(int(seed) & 0xFFFFFFFF)
            try:
                points = mesh.sample(self.n_points)
            finally:
                np.random.set_state(state)
        else:
            points = mesh.sample(self.n_points)
        if self._model is None:
            self._build_encoder()
        batch = points.reshape(1, self.n_points, 3)
        features = self._model.predict(batch, verbose=0)
        return features.flatten()

    def get_feature_dim(self, **kwargs) -> int:
        return self.feature_dim

    def get_name(self) -> str:
        return f"PointNet_{self.n_points}_{self.feature_dim}"


class ULIPExtractor(FeatureExtractor):
    """ULIP-based feature extractor for 3D models (lazy imports)."""

    def __init__(
        self,
        ckpt_path: str,
        model_name: str = "ULIP_PointBERT",
        n_points: int = 8192,
        batch_size: int = 1,
    ):
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.n_points = n_points
        self.batch_size = batch_size
        self._ulip_model = None
        self._tokenizer = None
        self._args = None

    def _setup_ulip(self):
        if self._ulip_model is not None:
            return

        # PyTorch compatibility fix for ULIP
        import torch

        class MockTorchSix:
            string_classes = (str,)

        torch._six = MockTorchSix()

        # ULIP path
        ulip_dir = os.path.join(os.path.dirname(__file__), "../../../external/ULIP")
        sys.path.append(ulip_dir)

        import models.ULIP_models as models
        from utils.tokenizer import SimpleTokenizer

        # Args mock
        class Args:
            def __init__(self, n_points):
                self.validate_dataset_name = "customdata"
                self.pretrain_dataset_prompt = "shapenet_64"
                self.validate_dataset_prompt = "modelnet40_64"
                self.npoints = n_points
                self.use_height = False
                self.evaluate_3d = False
                self.pc_size = n_points
                self.pc_augm_scale = 0.0
                self.pc_augm_rot = False
                self.pc_augm_mirror_prob = 0.0
                self.pc_augm_jitter = False
                self.pc_augm_shift = False
                self.pc_augm_scale_range = [0.8, 1.2]
                self.pc_augm_rot_range = [-0.1, 0.1]
                self.pc_augm_jitter_std = 0.01
                self.pc_augm_shift_range = [-0.1, 0.1]

        self._args = Args(self.n_points)

        # Load model
        import torch
        from utils.utils import get_model

        # Validate checkpoint file and load (supports .pt/.pth and .safetensors)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"ULIP checkpoint not found: {self.ckpt_path}")
        try:
            file_size = os.path.getsize(self.ckpt_path)
            if file_size < 1024 * 1024:
                print(f"Warning: Checkpoint file is unusually small ({file_size} bytes). It may be an HTML error page.")
        except Exception:
            pass

        state = None
        try:
            if self.ckpt_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file as safe_load
                except Exception as e:
                    raise RuntimeError("safetensors is required for .safetensors checkpoints: pip install safetensors") from e
                state = safe_load(self.ckpt_path)
            else:
                state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load ULIP checkpoint: {self.ckpt_path}. Error: {e}") from e

        # Extract state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        elif isinstance(state, dict):
            state_dict = state
        else:
            raise RuntimeError("Unexpected checkpoint format: expected dict or dict with 'state_dict' key")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self._ulip_model = getattr(models, self.model_name)(args=self._args)
        self._ulip_model.load_state_dict(state_dict, strict=False)
        if torch.cuda.is_available():
            self._ulip_model.cuda().eval()
        else:
            self._ulip_model.eval()

        self._tokenizer = SimpleTokenizer()
        self._get_model = get_model

    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        import torch

        self._setup_ulip()
        points = mesh.sample(self.n_points)
        tensor = torch.from_numpy(points).float().unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            features = self._get_model(self._ulip_model).encode_pc(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.detach().cpu().numpy().flatten()
        return features

    def get_feature_dim(self, **kwargs) -> int:
        import torch

        self._setup_ulip()
        dummy = torch.randn(1, self.n_points, 3)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
        with torch.no_grad():
            feat = self._get_model(self._ulip_model).encode_pc(dummy)
        return int(feat.shape[-1])

    def get_name(self) -> str:
        return f"ULIP_{self.model_name}_{self.n_points}_{self.get_feature_dim()}d"


class LLaVAExtractor(FeatureExtractor):
    """LLaVA-based image embedding from multi-view renders of 3D meshes.

    Renders N views via Blender, encodes each image with LLaVA vision tower if available.
    Falls back to OpenCLIP ViT-L/14 embeddings if LLaVA is not fully configured.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "llava-v1.5-7b",
        n_views: int = 8,
        blender_path: str = "blender",
        vit_variant: str = "ViT-L-14",
        vit_pretrained: str = "openai",
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.n_views = n_views
        self.blender_path = blender_path
        self.vit_variant = vit_variant
        self.vit_pretrained = vit_pretrained

        self._llava_model = None
        self._image_processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_model = None
        self._clip_preprocess = None

    def _ensure_llava(self):
        if self._llava_model is not None:
            return True
        try:
            llava_dir = os.path.join(os.path.dirname(__file__), "../../../external/LLaVA")
            sys.path.append(llava_dir)
            from llava.model.builder import load_pretrained_model  # type: ignore

            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=self.model_name,
                device_map=self._device,
            )
            self._llava_model = model
            self._image_processor = image_processor
            return True
        except Exception:
            return False

    def _ensure_openclip(self):
        if self._clip_model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.vit_variant, pretrained=self.vit_pretrained
        )
        model = model.to(self._device)
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess

    def _render_views(self, source_path: str) -> List[str]:
        script_path = os.path.join(os.path.dirname(__file__), "render_obj_multi_views.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                self.blender_path,
                "--background",
                "--python", script_path,
                "--",
                source_path,
                tmpdir,
                str(self.n_views),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Collect rendered images
            paths = [
                os.path.join(tmpdir, f)
                for f in sorted(os.listdir(tmpdir))
                if f.endswith(".png")
            ]
            # Copy to permanent temp files to persist after context exits
            keep_paths: List[str] = []
            for p in paths:
                dst_fd, dst_path = tempfile.mkstemp(suffix=".png")
                os.close(dst_fd)
                with open(p, "rb") as rf, open(dst_path, "wb") as wf:
                    wf.write(rf.read())
                keep_paths.append(dst_path)
            return keep_paths

    def _encode_images_llava(self, image_paths: List[str]) -> np.ndarray:
        assert self._llava_model is not None and self._image_processor is not None
        from llava.mm_utils import process_images  # type: ignore

        feats: List[torch.Tensor] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            image_tensor = process_images([img], self._image_processor, self._llava_model.config)
            image_tensor = image_tensor.to(self._device)
            with torch.no_grad():
                # Try common API; fall back to vision tower features
                if hasattr(self._llava_model, "encode_images"):
                    emb = self._llava_model.encode_images(image_tensor)
                else:
                    vt = self._llava_model.get_vision_tower()
                    if hasattr(vt, "forward_features"):
                        emb = vt.forward_features(image_tensor)
                        if isinstance(emb, (list, tuple)):
                            emb = emb[0]
                        emb = emb.mean(dim=1)
                    else:
                        emb = vt(image_tensor)
                        if emb.ndim == 3:
                            emb = emb.mean(dim=1)
                # Normalize
                emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.detach().cpu())
        stacked = torch.cat(feats, dim=0)
        mean_feat = stacked.mean(dim=0)
        return mean_feat.cpu().numpy().flatten()

    def _encode_images_openclip(self, image_paths: List[str]) -> np.ndarray:
        self._ensure_openclip()
        assert self._clip_model is not None and self._clip_preprocess is not None
        feats: List[torch.Tensor] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensor = self._clip_preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._clip_model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.detach().cpu())
        stacked = torch.cat(feats, dim=0)
        mean_feat = stacked.mean(dim=0)
        return mean_feat.cpu().numpy().flatten()

    def extract_features(self, mesh: trimesh.Trimesh, **kwargs) -> np.ndarray:
        source_path = kwargs.get("source_path")
        if not source_path or not os.path.exists(source_path):
            raise ValueError("LLaVAExtractor requires 'source_path' to existing 3D file for rendering.")
        image_paths = self._render_views(source_path)
        try_llava = self._ensure_llava()
        try:
            if try_llava:
                return self._encode_images_llava(image_paths)
            return self._encode_images_openclip(image_paths)
        finally:
            # Cleanup temporary images
            for p in image_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def get_feature_dim(self, **kwargs) -> int:
        # Determine dimension without full render by using a dummy black image
        try_llava = self._ensure_llava()
        if try_llava:
            # Use model hidden size if available
            hidden_size = getattr(self._llava_model.config, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)
        # Fallback to OpenCLIP embedding dim
        self._ensure_openclip()
        assert self._clip_model is not None
        return int(self._clip_model.visual.output_dim)

    def get_name(self) -> str:
        return f"LLaVA_{self.model_name}_{self.n_views}views"


class FeatureExtractorRegistry:
    """Simple registry for feature extractors.

    Usage:
      FeatureExtractorRegistry.register("pointcloud", PointCloudExtractor)
      FeatureExtractorRegistry.create("pointcloud", n_points=2048)
    """

    _registry: Dict[str, Type[FeatureExtractor]] = {}

    @classmethod
    def register(cls, key: str, extractor_cls: Type[FeatureExtractor]) -> None:
        key = key.lower().strip()
        cls._registry[key] = extractor_cls

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def create(cls, key: str, **kwargs) -> FeatureExtractor:
        extractor_cls = cls._registry.get(key.lower().strip())
        if extractor_cls is None:
            raise ValueError(f"Unknown feature extractor: {key}. Available: {cls.available()}")
        return extractor_cls(**kwargs)


# Register built-ins
FeatureExtractorRegistry.register("pointcloud", PointCloudExtractor)
FeatureExtractorRegistry.register("pointnet", PointNetExtractor)
FeatureExtractorRegistry.register("ulip", ULIPExtractor)
FeatureExtractorRegistry.register("llava", LLaVAExtractor)


