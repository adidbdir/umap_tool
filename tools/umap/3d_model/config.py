import json
import os
from typing import Any, Dict, List


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    _, ext = os.path.splitext(path)
    with open(path, "r") as f:
        if ext.lower() in [".yml", ".yaml"]:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("YAML config requested but PyYAML is not installed.") from e
            return yaml.safe_load(f)
        return json.load(f)


def infer_scalar(value: str) -> Any:
    # Try to cast to bool/int/float; fallback to string
    lv = value.lower()
    if lv == "true":
        return True
    if lv == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        set_by_dotted_path(cfg, key.strip(), infer_scalar(raw.strip()))
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    # Minimal schema validation with informative errors
    def require(path: str, typ):
        ref = cfg
        for p in path.split('.'):
            if p not in ref:
                raise ValueError(f"Missing config key: {path}")
            ref = ref[p]
        if typ is not None and not isinstance(ref, typ):
            raise TypeError(f"Invalid type for {path}: expected {typ}, got {type(ref)}")

    require('input', dict)
    require('input.dir', str)
    require('input.prefix', str)
    require('input.n_points', int)

    require('output', dict)
    require('output.dir', str)

    require('umap', dict)
    require('umap.encoder_type', str)
    if cfg['umap']['encoder_type'] not in ['mlp', 'pointnet']:
        raise ValueError("umap.encoder_type must be 'mlp' or 'pointnet'")

    require('extractor', dict)
    require('extractor.type', str)
    if cfg['extractor']['type'] not in ['pointcloud', 'pointnet', 'ulip', 'llava']:
        raise ValueError("extractor.type must be one of: pointcloud, pointnet, ulip, llava")

    # Per-extractor fields are optional but validated when chosen
    if cfg['extractor']['type'] == 'pointnet':
        require('extractor.pointnet', dict)
        require('extractor.pointnet.feature_dim', int)
    if cfg['extractor']['type'] == 'ulip':
        require('extractor.ulip', dict)
        require('extractor.ulip.ckpt_path', str)
        require('extractor.ulip.model', str)
    if cfg['extractor']['type'] == 'llava':
        require('extractor.llava', dict)
        # model_path may be None when falling back to open_clip
        require('extractor.llava.model_name', str)
        require('extractor.llava.n_views', int)
        require('extractor.llava.blender_bin', str)


