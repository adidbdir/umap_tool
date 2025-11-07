import argparse
from typing import Any

from feature_extractors import FeatureExtractorRegistry


def list_extractors(_: Any) -> None:
    print("Available feature extractors:")
    for key in FeatureExtractorRegistry.available():
        print(f"- {key}")


def info_extractor(args: Any) -> None:
    key = args.name
    if key not in FeatureExtractorRegistry.available():
        print(f"Unknown extractor: {key}")
        print("Available:", FeatureExtractorRegistry.available())
        return
    # Try to instantiate with common defaults to show dim
    kwargs = {}
    if key == "pointcloud":
        kwargs = {"n_points": args.n_points}
    elif key == "pointnet":
        kwargs = {"n_points": args.n_points, "feature_dim": args.pointnet_feature_dim}
    elif key == "ulip":
        if not args.ulip_ckpt_path:
            print("--ulip-ckpt-path is required for ULIP info")
            return
        kwargs = {
            "ckpt_path": args.ulip_ckpt_path,
            "model_name": args.ulip_model,
            "n_points": args.n_points,
        }
    extractor = FeatureExtractorRegistry.create(key, **kwargs)
    print(f"Name: {extractor.get_name()}")
    print(f"Feature dim: {extractor.get_feature_dim()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature extractor management CLI")
    sub = parser.add_subparsers(dest="command")

    sub_list = sub.add_parser("list", help="List available extractors")
    sub_list.set_defaults(func=list_extractors)

    sub_info = sub.add_parser("info", help="Show info for a specific extractor")
    sub_info.add_argument("name", type=str, help="Extractor name")
    sub_info.add_argument("--n-points", type=int, default=2048)
    sub_info.add_argument("--pointnet-feature-dim", type=int, default=1024)
    sub_info.add_argument("--ulip-ckpt-path", type=str, default=None)
    sub_info.add_argument("--ulip-model", type=str, default="ULIP_PointBERT")
    sub_info.set_defaults(func=info_extractor)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()


