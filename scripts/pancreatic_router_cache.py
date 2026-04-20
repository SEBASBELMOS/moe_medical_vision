#!/usr/bin/env python3
"""Build missing pancreatic router cache safely.

This script exists to unblock the 5-expert router export when the project is
missing only the pancreatic cache.

It can do two independent steps:
1. Precompute `/data/processed/pancreatic_fast/*.npz`
2. Extract pancreatic CLS embeddings for the router cache

Designed for careful operation in shared environments:
- validation mode
- low default workers
- no training
- can run CPU-only if needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SIZE_3D = (64, 64, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/workspace/moe_medical_vision")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument(
        "--extract-from-raw",
        action="store_true",
        help="Extract pancreatic embeddings directly from raw dataset instead of pancreatic_fast.",
    )
    return parser.parse_args()


def resize_volume(vol_np: np.ndarray, size=SIZE_3D) -> np.ndarray:
    t = torch.from_numpy(vol_np).float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=size, mode="trilinear", align_corners=False)
    return t.squeeze(0).numpy()


class PancFastDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.files = sorted(root.glob(f"{split}_*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        return {
            "image": torch.from_numpy(d["volume"][0].copy()).float().unsqueeze(0),
            "label": int(d["label"]),
        }


def validate_paths(project_root: Path) -> dict:
    raw_root = project_root / "data" / "raw" / "pancreatic"
    fast_root = project_root / "data" / "processed" / "pancreatic_fast"
    emb_root = project_root / "embeddings"
    modern_root = project_root / "data" / "processed" / "router_embeddings"
    return {
        "raw_pancreatic_exists": raw_root.exists(),
        "fast_pancreatic_exists": fast_root.exists(),
        "legacy_embeddings_exist": (emb_root / "Z_train_pancreatic.npy").exists(),
        "modern_embeddings_exist": (modern_root / "Z_train_pancreatic.npz").exists(),
        "raw_root": str(raw_root),
        "fast_root": str(fast_root),
        "embeddings_root": str(emb_root),
        "router_embeddings_root": str(modern_root),
    }


def build_fast_cache(project_root: Path) -> dict:
    import sys

    sys.path.insert(0, str(project_root / "src"))
    from data.datasets import PancreaticCancerDataset

    raw_root = project_root / "data" / "raw" / "pancreatic"
    fast_root = project_root / "data" / "processed" / "pancreatic_fast"
    fast_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "train_new": 0,
        "train_skipped": 0,
        "train_errors": 0,
        "val_new": 0,
        "val_skipped": 0,
        "val_errors": 0,
    }

    for split in ("train", "val"):
        ds = PancreaticCancerDataset(str(raw_root), split=split, transform=None)
        total = len(ds)
        print(f"[precompute:{split}] total={total}", flush=True)
        for i in range(len(ds)):
            row = ds.df.iloc[i]
            out_file = fast_root / f"{split}_{i:05d}.npz"
            if out_file.exists():
                summary[f"{split}_skipped"] += 1
                if (summary[f"{split}_skipped"] + summary[f"{split}_new"]) % 25 == 0:
                    print(
                        f"[precompute:{split}] done={summary[f'{split}_new']} "
                        f"skipped={summary[f'{split}_skipped']} errors={summary[f'{split}_errors']} "
                        f"progress={i + 1}/{total}",
                        flush=True,
                    )
                continue
            try:
                vol_tensor = ds._load_volume(row["filename"])
                vol = resize_volume(vol_tensor.numpy(), SIZE_3D)
                np.savez_compressed(out_file, volume=vol, label=int(row["label"]))
                summary[f"{split}_new"] += 1
                if summary[f"{split}_new"] % 10 == 0:
                    print(
                        f"[precompute:{split}] new={summary[f'{split}_new']} "
                        f"skipped={summary[f'{split}_skipped']} errors={summary[f'{split}_errors']} "
                        f"progress={i + 1}/{total}",
                        flush=True,
                    )
            except Exception as exc:
                summary[f"{split}_errors"] += 1
                print(
                    f"[precompute:{split}] ERROR idx={i} file={row['filename']} err={exc}",
                    flush=True,
                )

        print(
            f"[precompute:{split}] finished new={summary[f'{split}_new']} "
            f"skipped={summary[f'{split}_skipped']} errors={summary[f'{split}_errors']}",
            flush=True,
        )

    return summary


def extract_pancreatic_embeddings(
    project_root: Path, device: str, batch_size: int, num_workers: int
) -> dict:
    import timm

    fast_root = project_root / "data" / "processed" / "pancreatic_fast"
    emb_root = project_root / "embeddings"
    modern_root = project_root / "data" / "processed" / "router_embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)
    modern_root.mkdir(parents=True, exist_ok=True)

    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    summary = {}
    for split in ("train", "val"):
        ds = PancFastDataset(fast_root, split)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        print(f"[extract:{split}] files={len(ds)} batch_size={batch_size}", flush=True)
        z_all = []
        y_task = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                x = batch["image"].to(device)
                x = x.squeeze(1)
                mip = torch.stack(
                    [x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1
                )
                x = F.interpolate(
                    mip, size=(224, 224), mode="bilinear", align_corners=False
                )
                z = backbone(x)
                z_all.append(z.cpu().numpy())
                y_task.extend(batch["label"].tolist())
                if batch_idx % 10 == 0:
                    seen = min(batch_idx * batch_size, len(ds))
                    print(
                        f"[extract:{split}] batches={batch_idx} seen={seen}/{len(ds)}",
                        flush=True,
                    )

        z_np = np.concatenate(z_all, axis=0).astype(np.float32)
        y_task_np = np.asarray(y_task, dtype=np.int64)
        y_expert_np = np.full(len(z_np), 4, dtype=np.int32)

        np.save(emb_root / f"Z_{split}_pancreatic.npy", z_np)
        np.save(emb_root / f"y_{split}_pancreatic.npy", y_task_np)
        np.savez_compressed(
            modern_root / f"Z_{split}_pancreatic.npz",
            z=z_np,
            y_expert=y_expert_np,
            y_task=y_task_np,
        )

        summary[split] = {
            "count": int(len(z_np)),
            "dim": int(z_np.shape[1]),
        }
        print(
            f"[extract:{split}] saved count={len(z_np)} dim={z_np.shape[1]}", flush=True
        )

    return summary


def extract_pancreatic_embeddings_from_raw(
    project_root: Path, device: str, batch_size: int, num_workers: int
) -> dict:
    import sys
    import timm

    sys.path.insert(0, str(project_root / "src"))
    from data.datasets import get_dataloader

    raw_root = project_root / "data" / "raw" / "pancreatic"
    emb_root = project_root / "embeddings"
    modern_root = project_root / "data" / "processed" / "router_embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)
    modern_root.mkdir(parents=True, exist_ok=True)

    backbone = timm.create_model(
        "vit_tiny_patch16_224", pretrained=True, num_classes=0
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    summary = {}
    for split in ("train", "val"):
        loader, ds = get_dataloader(
            "pancreatic",
            str(raw_root),
            split=split,
            batch_size=min(batch_size, 4),
            num_workers=num_workers,
            transform=None,
        )
        print(
            f"[extract-raw:{split}] files={len(ds)} batch_size={min(batch_size, 4)}",
            flush=True,
        )
        z_all = []
        y_task = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, start=1):
                x = batch["image"].to(device)
                x = x.squeeze(1)
                mip = torch.stack(
                    [x.max(dim=1)[0], x.max(dim=2)[0], x.max(dim=3)[0]], dim=1
                )
                x = F.interpolate(
                    mip, size=(224, 224), mode="bilinear", align_corners=False
                )
                z = backbone(x)
                z_all.append(z.cpu().numpy())
                y_task.extend(batch["label"].tolist())
                if batch_idx % 10 == 0:
                    seen = min(batch_idx * min(batch_size, 4), len(ds))
                    print(
                        f"[extract-raw:{split}] batches={batch_idx} seen={seen}/{len(ds)}",
                        flush=True,
                    )

        z_np = np.concatenate(z_all, axis=0).astype(np.float32)
        y_task_np = np.asarray(y_task, dtype=np.int64)
        y_expert_np = np.full(len(z_np), 4, dtype=np.int32)

        np.save(emb_root / f"Z_{split}_pancreatic.npy", z_np)
        np.save(emb_root / f"y_{split}_pancreatic.npy", y_task_np)
        np.savez_compressed(
            modern_root / f"Z_{split}_pancreatic.npz",
            z=z_np,
            y_expert=y_expert_np,
            y_task=y_task_np,
        )

        summary[split] = {"count": int(len(z_np)), "dim": int(z_np.shape[1])}
        print(
            f"[extract-raw:{split}] saved count={len(z_np)} dim={z_np.shape[1]}",
            flush=True,
        )

    return summary


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)

    validation = validate_paths(project_root)
    print(json.dumps(validation, indent=2))
    if args.validate_only:
        return 0

    result = {"validation": validation}
    if not args.extract_only:
        result["precompute"] = build_fast_cache(project_root)

    if not args.precompute_only:
        if args.extract_from_raw:
            result["extract"] = extract_pancreatic_embeddings_from_raw(
                project_root=project_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            result["extract"] = extract_pancreatic_embeddings(
                project_root=project_root,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
