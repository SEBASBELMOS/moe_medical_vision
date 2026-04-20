#!/usr/bin/env python3
"""Rebuild router bundles and k-NN artifacts from cached embeddings.

This script is designed to repair the router artifact contract without
retraining experts or touching any running GPU job.

Supported sources:
1. Modern combined cache in `data/processed/router_embeddings/`
2. Legacy per-dataset cache in `embeddings/`

It can:
- validate which datasets are present,
- rebuild `Z_train.npz` / `Z_val.npz`,
- export `router_pca.pkl`, `router_knn.index`, `router_knn_labels.pkl`,
- write a manifest with counts and sources.

By default it is strict: all 5 expert domains must be present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Missing dependency: faiss. Install faiss-cpu first.") from exc

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("Missing dependency: scikit-learn. Install it first.") from exc


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    expert_id: int
    aliases: tuple[str, ...]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("nih_chestxray", 0, ("nih_chestxray",)),
    DatasetSpec("isic2019", 1, ("isic2019",)),
    DatasetSpec("osteoarthritis", 2, ("osteoarthritis",)),
    DatasetSpec("luna16", 3, ("luna16",)),
    DatasetSpec(
        "pancreatic", 4, ("pancreatic", "pancreatic_fast", "pancreatic_cancer")
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default="/workspace/moe_medical_vision",
        help="Project root that contains embeddings/, checkpoints/ and data/processed/.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=32,
        help="PCA dimension for FAISS router export.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only inspect availability and print manifest. Do not write artifacts.",
    )
    parser.add_argument(
        "--allow-missing",
        nargs="*",
        default=(),
        help="Dataset names allowed to be missing. Example: --allow-missing pancreatic",
    )
    return parser.parse_args()


def load_legacy_split(
    legacy_dir: Path, spec: DatasetSpec, split: str
) -> tuple[np.ndarray, np.ndarray] | None:
    for alias in spec.aliases:
        z_path = legacy_dir / f"Z_{split}_{alias}.npy"
        y_path = legacy_dir / f"y_{split}_{alias}.npy"
        if z_path.exists() and y_path.exists():
            z = np.load(z_path)
            y_task = np.load(y_path)
            return z, y_task
    return None


def load_modern_split(
    router_dir: Path, spec: DatasetSpec, split: str
) -> tuple[np.ndarray, np.ndarray] | None:
    for alias in spec.aliases:
        npz_path = router_dir / f"Z_{split}_{alias}.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            return data["z"], data["y_task"]
    return None


def discover_split(project_root: Path, split: str) -> tuple[list[dict], list[str]]:
    router_dir = project_root / "data" / "processed" / "router_embeddings"
    legacy_dir = project_root / "embeddings"

    discovered: list[dict] = []
    missing: list[str] = []

    for spec in DATASETS:
        loaded = None
        source = None

        if router_dir.exists():
            loaded = load_modern_split(router_dir, spec, split)
            if loaded is not None:
                source = "modern"

        if loaded is None and legacy_dir.exists():
            loaded = load_legacy_split(legacy_dir, spec, split)
            if loaded is not None:
                source = "legacy"

        if loaded is None:
            missing.append(spec.name)
            continue

        z, y_task = loaded
        y_expert = np.full(len(z), spec.expert_id, dtype=np.int32)
        discovered.append(
            {
                "dataset": spec.name,
                "expert_id": spec.expert_id,
                "source": source,
                "z": z,
                "y_task": y_task,
                "y_expert": y_expert,
                "count": int(len(z)),
                "dim": int(z.shape[1]),
            }
        )

    return discovered, missing


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


def write_combined_npz(
    out_dir: Path, split: str, discovered: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.concatenate([item["z"] for item in discovered], axis=0).astype(np.float32)
    y_expert = np.concatenate([item["y_expert"] for item in discovered], axis=0).astype(
        np.int32
    )
    y_task = np.concatenate([item["y_task"] for item in discovered], axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"Z_{split}.npz", z=z, y_expert=y_expert, y_task=y_task
    )
    return z, y_expert, y_task


def export_knn_router(
    checkpoint_dir: Path, z_train: np.ndarray, y_train_expert: np.ndarray, pca_dim: int
) -> dict:
    z_train_norm = normalize_rows(z_train.astype(np.float32))
    pca = PCA(n_components=pca_dim, random_state=42)
    z_train_pca = pca.fit_transform(z_train_norm).astype(np.float32)
    faiss.normalize_L2(z_train_pca)

    index = faiss.IndexFlatIP(pca_dim)
    index.add(z_train_pca)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, checkpoint_dir / "router_pca.pkl")
    faiss.write_index(index, str(checkpoint_dir / "router_knn.index"))
    joblib.dump(
        y_train_expert.astype(np.int32), checkpoint_dir / "router_knn_labels.pkl"
    )

    mem_mb = len(z_train_pca) * pca_dim * 4 / (1024**2)
    return {
        "router": "k-NN + PCA + FAISS",
        "n_train": int(len(z_train_pca)),
        "pca_dim": int(pca_dim),
        "estimated_index_mb": round(mem_mb, 2),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    out_dir = project_root / "data" / "processed" / "router_embeddings"
    checkpoint_dir = project_root / "checkpoints"
    allow_missing = set(args.allow_missing)

    train_discovered, train_missing = discover_split(project_root, "train")
    val_discovered, val_missing = discover_split(project_root, "val")

    missing = sorted(set(train_missing) | set(val_missing))
    blocking_missing = [name for name in missing if name not in allow_missing]

    manifest = {
        "project_root": str(project_root),
        "train": [
            {
                "dataset": item["dataset"],
                "expert_id": item["expert_id"],
                "source": item["source"],
                "count": item["count"],
                "dim": item["dim"],
            }
            for item in train_discovered
        ],
        "val": [
            {
                "dataset": item["dataset"],
                "expert_id": item["expert_id"],
                "source": item["source"],
                "count": item["count"],
                "dim": item["dim"],
            }
            for item in val_discovered
        ],
        "missing_datasets": missing,
        "blocking_missing": blocking_missing,
        "allow_missing": sorted(allow_missing),
    }

    print(json.dumps(manifest, indent=2))

    if args.validate_only:
        return 0 if not blocking_missing else 2

    if blocking_missing:
        print(
            "Cannot export router artifacts because required datasets are missing: "
            + ", ".join(blocking_missing)
        )
        return 2

    z_train, y_train_expert, _ = write_combined_npz(out_dir, "train", train_discovered)
    z_val, y_val_expert, _ = write_combined_npz(out_dir, "val", val_discovered)

    export_meta = export_knn_router(
        checkpoint_dir, z_train, y_train_expert, args.pca_dim
    )
    manifest["export"] = {
        **export_meta,
        "train_bundle": str(out_dir / "Z_train.npz"),
        "val_bundle": str(out_dir / "Z_val.npz"),
        "router_pca": str(checkpoint_dir / "router_pca.pkl"),
        "router_index": str(checkpoint_dir / "router_knn.index"),
        "router_labels": str(checkpoint_dir / "router_knn_labels.pkl"),
        "y_expert_counts": np.bincount(y_train_expert, minlength=5).tolist(),
        "val_expert_counts": np.bincount(y_val_expert, minlength=5).tolist(),
    }

    manifest_path = out_dir / "router_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
