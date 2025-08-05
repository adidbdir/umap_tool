#!/usr/bin/env python
"""
download_objaverse_cc_sample.py
---------------------------------
Download a CC‑BY/CC0 subset of Objaverse‑XL (default 10 000 objects),
filter by file type (default: glb), save metadata, and print summary.

Dependencies:
    pip install objaverse pandas tqdm
"""

import objaverse.xl as oxl
import pandas as pd
import re, argparse, multiprocessing
from pathlib import Path

# ---------- licence filter -------------------------------------------------
_BAD_CC_TOKENS = ("noncommercial", "non-commercial", "nc", "nd", "share alike", "sa")


def is_cc_by_or_cc0(lic: str) -> bool:
    """Return True for CC‑BY (without NC/ND/SA) or CC0 licence strings."""
    if not isinstance(lic, str):
        return False
    l = lic.lower()

    # CC0 / public domain
    if ("creative commons zero" in l) or ("cc0" in l) or ("public domain" in l):
        return True

    # CC‑BY (reject NC / ND / SA variants)
    if "creative commons" in l and (("attribution" in l) or re.search(r"\bby\b", l)):
        return not any(tok in l for tok in _BAD_CC_TOKENS)

    return False


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download a CC‑BY/CC0 subset of Objaverse‑XL "
        "(default 10 000 objects) and summarise metadata."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10_000,
        help="Number of objects to download (default: 10 000)",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="./objaverse_cc_sample",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="Parallel processes (default: cpu count)",
    )
    parser.add_argument(
        "-f",
        "--filetype",
        default="glb",
        help="Desired file type filter (e.g. glb, obj, ply). "
        "Use 'any' to disable (default: glb)",
    )
    args = parser.parse_args()

    # 1. annotations (cached after first run)
    print("Fetching Objaverse‑XL annotations …")
    ann = oxl.get_annotations(download_dir="~/.objaverse")

    # 2. licence + optional file‑type filter
    print("Filtering by licence and file type …")
    mask = ann["license"].apply(is_cc_by_or_cc0)
    subset = ann[mask]
    if args.filetype.lower() != "any":
        subset = subset[subset["fileType"].str.lower() == args.filetype.lower()]

    total = len(subset)
    if total == 0:
        raise RuntimeError("No objects match the given criteria.")
    print(f"{total:,} objects pass the filters.")

    # 3. random sample
    n = min(args.num, total)
    sample_df = subset.sample(n, random_state=42).reset_index(drop=True)
    print(f"Sampling {n:,} objects …")

    # 4. download
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    procs = (
        args.processes if args.processes is not None else multiprocessing.cpu_count()
    )
    print(f"Downloading to {out_dir} with {procs} processes …")
    oxl.download_objects(
        objects=sample_df,
        download_dir=str(out_dir),
        processes=procs,
        save_repo_format="files",  # GitHub repos saved as loose files (no zip)
    )

    # 5. save metadata CSV
    csv_path = out_dir / "metadata_sample.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")

    # 6. print summary tables
    print("\n=== Licence distribution ===")
    print(sample_df["license"].value_counts())

    print("\n=== File type distribution ===")
    print(sample_df["fileType"].value_counts())

    print("\nDone.")


if __name__ == "__main__":
    main()
