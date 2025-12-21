#!/usr/bin/env python3
"""
Download datasets from the Hugging Face Hub into this repo's ./dataset directory.

Example:
  python3 downloader.py --repo_id tonyzhao/aloha_sim_transfer_cube --repo_type dataset

By default this downloads into:
  <repo_root>/dataset/<repo_name>/
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional


def _split_csv(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(",")]
        out.extend([p for p in parts if p])
    return out


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_dataset_dir(repo_root: Path) -> Path:
    return repo_root / "dataset"


def _resolve_dest_dir(
    repo_id: str,
    dataset_dir: Path,
    out_dir: Optional[Path],
    local_subdir: Optional[str],
) -> Path:
    if out_dir is not None:
        return out_dir
    repo_name = repo_id.split("/")[-1]
    subdir = local_subdir if local_subdir is not None else repo_name
    return dataset_dir / subdir


def download_from_huggingface(
    *,
    repo_id: str,
    dest_dir: Path,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  pip install huggingface_hub\n"
            "If the repo is private, also set HF_TOKEN or pass --token."
        ) from exc

    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    sig = inspect.signature(snapshot_download)
    params = sig.parameters

    kwargs = {"repo_id": repo_id}
    if "repo_type" in params:
        kwargs["repo_type"] = repo_type
    if revision is not None and "revision" in params:
        kwargs["revision"] = revision
    if cache_dir is not None and "cache_dir" in params:
        kwargs["cache_dir"] = str(cache_dir)
    if allow_patterns and "allow_patterns" in params:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns and "ignore_patterns" in params:
        kwargs["ignore_patterns"] = ignore_patterns
    if max_workers is not None and "max_workers" in params:
        kwargs["max_workers"] = max_workers

    if token is not None:
        if "token" in params:
            kwargs["token"] = token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = token

    if "resume_download" in params:
        kwargs["resume_download"] = True

    if "local_dir" in params:
        kwargs["local_dir"] = str(dest_dir)
        if "local_dir_use_symlinks" in params:
            kwargs["local_dir_use_symlinks"] = False
        snapshot_path = Path(snapshot_download(**kwargs))
        return snapshot_path

    snapshot_path = Path(snapshot_download(**kwargs))
    shutil.copytree(snapshot_path, dest_dir, dirs_exist_ok=True)
    return dest_dir


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    default_dataset = _default_dataset_dir(repo_root)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo_id",
        default=os.environ.get("HF_REPO_ID"),
        help="Hugging Face repo id, e.g. 'org/name' (or set HF_REPO_ID).",
    )
    parser.add_argument(
        "--repo_type",
        default="dataset",
        choices=("dataset", "model", "space"),
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Repo revision (branch, tag, or commit).",
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=default_dataset,
        help=f"Dataset root directory (default: {default_dataset}).",
    )
    parser.add_argument(
        "--local_subdir",
        default=None,
        help="Subdirectory under --dataset_dir (default: repo name).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Exact output directory (overrides --dataset_dir/--local_subdir).",
    )

    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Hugging Face cache dir (default: <dataset_dir>/.hf-cache).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN).",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Allow patterns (glob), repeatable or comma-separated.",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Ignore patterns (glob), repeatable or comma-separated.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Max parallel workers for download (if supported by your huggingface_hub).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print paths and exit.")
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce output and progress bars."
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("Missing --repo_id (or set HF_REPO_ID).")

    if args.quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    dataset_dir = args.dataset_dir.resolve()
    dest_dir = _resolve_dest_dir(
        repo_id=args.repo_id,
        dataset_dir=dataset_dir,
        out_dir=args.out_dir,
        local_subdir=args.local_subdir,
    )
    cache_dir = (
        args.cache_dir if args.cache_dir is not None else (dataset_dir / ".hf-cache")
    )

    allow_patterns = _split_csv(args.allow)
    ignore_patterns = _split_csv(args.ignore)

    if args.dry_run:
        if not args.quiet:
            print(f"repo_id:     {args.repo_id}")
            print(f"repo_type:   {args.repo_type}")
            print(f"revision:    {args.revision}")
            print(f"dataset_dir: {dataset_dir}")
            print(f"dest_dir:    {dest_dir}")
            print(f"cache_dir:   {cache_dir}")
            if allow_patterns:
                print(f"allow:       {allow_patterns}")
            if ignore_patterns:
                print(f"ignore:      {ignore_patterns}")
        return 0

    if not args.quiet:
        print(f"Downloading {args.repo_type} '{args.repo_id}' -> {dest_dir}")

    downloaded_path = download_from_huggingface(
        repo_id=args.repo_id,
        dest_dir=dest_dir,
        repo_type=args.repo_type,
        revision=args.revision,
        token=args.token,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns or None,
        ignore_patterns=ignore_patterns or None,
        max_workers=args.max_workers,
    )

    if not args.quiet:
        print(f"Done. Local path: {downloaded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
