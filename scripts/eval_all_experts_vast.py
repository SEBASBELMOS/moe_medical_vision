from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, resnet34, ResNet34_Weights
from torchvision.models import EfficientNet_B0_Weights

ROOT = Path("/workspace/moe_medical_vision")
sys.path.insert(0, str(ROOT / "src"))

from data.datasets import NIHChestXray14Dataset, ISIC2019Dataset, OsteoarthritisDataset  # noqa: E402
from data.datasets import PancreaticCancerDataset  # noqa: E402
from models.experts_3d import build_pancreatic_expert  # noqa: E402
from nih_preprocessing_v2 import NIHXrayTransform  # noqa: E402
import timm  # noqa: E402
import SimpleITK as sitk  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = CKPT_DIR / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class NIHDenseNetHead(nn.Module):
    def __init__(self, num_classes: int = 14, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet121", pretrained=False, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


class LunaMIPFast(Dataset):
    def __init__(self, root: Path, split: str = "val"):
        self.files = sorted(root.glob(f"{split}_*.npz"))
        self.labels = np.array(
            [int(np.load(f)["label"]) for f in self.files], dtype=np.int64
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        vol = torch.from_numpy(d["volume"][0].copy()).float()

        def norm(t: torch.Tensor) -> torch.Tensor:
            t_min, t_max = t.min(), t.max()
            if t_max > t_min:
                return (t - t_min) / (t_max - t_min)
            return t

        mip_max = norm(vol.max(dim=0)[0])
        mip_mean = norm(vol.mean(dim=0))
        mip_std = norm(vol.std(dim=0))
        img = torch.stack([mip_max, mip_mean, mip_std], dim=0)
        img = F.interpolate(
            img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return {
            "image": img,
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class PancFastDataset(Dataset):
    def __init__(self, root: Path, split: str = "val"):
        self.files = sorted(root.glob(f"{split}_*.npz"))
        self.labels = np.array(
            [int(np.load(f)["label"]) for f in self.files], dtype=np.int64
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])
        x = torch.from_numpy(d["volume"].copy()).float()
        y = torch.tensor(int(d["label"]), dtype=torch.long)
        return {"image": x, "label": y}


class LunaMIPRaw(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "val",
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        val_frac: float = 0.2,
        seed: int = 42,
    ):
        import pandas as pd

        ann_csv = root / "annotations.csv"
        series_with_nodule = set()
        if ann_csv.exists():
            ann_df = pd.read_csv(ann_csv)
            series_with_nodule = set(ann_df["seriesuid"].unique())

        mhd_dir = root / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
        mhd_files = (
            sorted(mhd_dir.glob("*.mhd"))
            if mhd_dir.exists()
            else sorted(root.rglob("*.mhd"))
        )
        all_samples = [(f, 1 if f.stem in series_with_nodule else 0) for f in mhd_files]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_samples))
        n_val = int(len(all_samples) * val_frac)
        sel = idx[:n_val] if split == "val" else idx[n_val:]
        self.samples = [all_samples[i] for i in sel]
        self.hu_min = hu_min
        self.hu_max = hu_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        itk_img = sitk.ReadImage(str(path))
        vol = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min)
        vol = torch.from_numpy(vol)

        def norm(t: torch.Tensor) -> torch.Tensor:
            t_min, t_max = t.min(), t.max()
            if t_max > t_min:
                return (t - t_min) / (t_max - t_min)
            return t

        mip_max = norm(vol.max(dim=0)[0])
        mip_mean = norm(vol.mean(dim=0))
        mip_std = norm(vol.std(dim=0))
        img = torch.stack([mip_max, mip_mean, mip_std], dim=0)
        img = F.interpolate(
            img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return {"image": img, "label": torch.tensor(int(label), dtype=torch.long)}


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro": float(roc_auc_score(y_true, y_prob_pos))
        if len(np.unique(y_true)) > 1
        else None,
        "confusion_matrix": cm.tolist(),
    }


def multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    auc = None
    present = np.unique(y_true)
    try:
        auc = float(
            roc_auc_score(
                y_true,
                y_prob[:, present],
                labels=present,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro_ovr": auc,
    }


def multilabel_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray
) -> dict:
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int64)
    per_class_auc = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if len(np.unique(col)) > 1:
            per_class_auc.append(float(roc_auc_score(col, y_prob[:, i])))
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_samples": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "precision_samples": float(
            precision_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "recall_samples": float(
            recall_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "auc_macro": float(np.mean(per_class_auc)) if per_class_auc else None,
        "thresholds": thresholds.tolist(),
    }


def evaluate_nih() -> dict:
    ckpt_candidates = [
        CKPT_DIR / "expert1_nih_enriched_best.pth",
        CKPT_DIR / "expert1_nih_best.pth",
        CKPT_DIR / "expert1_nih_final_best.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError("No NIH checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = NIHDenseNetHead(num_classes=14, dropout=0.2).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    thresholds = np.array(ckpt.get("thresholds", [0.5] * 14), dtype=np.float32)
    ds = NIHChestXray14Dataset(
        root=ROOT / "data/raw/nih",
        split="val",
        transform=NIHXrayTransform(split="val"),
        val_frac=0.15,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    probs, labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            p = torch.sigmoid(model(x)).cpu().numpy()
            labels.append(y)
            probs.append(p)

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    metrics = multilabel_metrics(y_true, y_prob, thresholds)
    metrics.update(
        {
            "checkpoint": ckpt_path.name,
            "best_f1_checkpoint": float(ckpt.get("best_f1", -1)),
            "best_auc_checkpoint": float(ckpt.get("best_auc", -1))
            if "best_auc" in ckpt
            else None,
            "epoch": int(ckpt.get("epoch", -1)),
            "expert_name": ckpt.get("expert_name", "expert1_nih"),
        }
    )
    return metrics


def evaluate_isic() -> dict:
    ckpt_candidates = [
        CKPT_DIR / "expert2_isic_best_fixed.pth",
        CKPT_DIR / "expert2_isic_best.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError("No ISIC checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=9).to(
        DEVICE
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ds = ISIC2019Dataset(
        root=ROOT / "data/raw/isic", split="val", val_frac=0.15, seed=42
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    probs, labels, preds = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            probs.append(p)
            labels.append(y)
            preds.append(pred)

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    metrics = multiclass_metrics(y_true, y_pred, y_prob)
    metrics.update(
        {
            "checkpoint": ckpt_path.name,
            "best_f1_checkpoint": float(ckpt.get("best_f1", -1)),
            "epoch": int(ckpt.get("epoch", -1)),
        }
    )
    return metrics


def evaluate_oa() -> dict:
    ckpt_path = CKPT_DIR / "expert3_oa_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = OsteoarthritisDataset(
        root=ROOT / "data/raw/osteoporosis/KLGrade/KLGrade",
        split="val",
        val_frac=0.2,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    probs, labels, preds = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            probs.append(p)
            labels.append(y)
            preds.append(pred)

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    y_pred = np.concatenate(preds)
    metrics = multiclass_metrics(y_true, y_pred, y_prob)
    metrics.update(
        {
            "checkpoint": ckpt_path.name,
            "best_f1_checkpoint": float(ckpt.get("best_f1", -1)),
            "epoch": int(ckpt.get("epoch", -1)),
        }
    )
    return metrics


def evaluate_luna_mip() -> dict:
    ckpt_path = CKPT_DIR / "expert4_luna16_MIP_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 2))
    model = model.to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    fast_root = ROOT / "data/processed/luna16_fast"
    if fast_root.exists() and list(fast_root.glob("val_*.npz")):
        ds = LunaMIPFast(fast_root, split="val")
    else:
        ds = LunaMIPRaw(ROOT / "data/raw/luna16", split="val")
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    prob_pos, labels, preds = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            prob_pos.append(p[:, 1])
            labels.append(y)
            preds.append(pred)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob_pos = np.concatenate(prob_pos)
    metrics = binary_metrics(y_true, y_pred, y_prob_pos)
    metrics.update(
        {
            "checkpoint": ckpt_path.name,
            "best_f1_checkpoint": float(ckpt.get("best_val_f1", -1)),
            "epoch": int(ckpt.get("epoch", -1)),
        }
    )
    return metrics


def evaluate_pancreatic() -> dict:
    ckpt_path = CKPT_DIR / "expert5_pancreatic_FAST_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_pancreatic_expert(
        pretrained=False, use_gradient_checkpointing=False
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    fast_root = ROOT / "data/processed/pancreatic_fast"
    if fast_root.exists() and list(fast_root.glob("val_*.npz")):
        ds = PancFastDataset(fast_root, split="val")
    else:
        ds = PancreaticCancerDataset(
            root=ROOT / "data/raw/pancreatic", split="val", val_frac=0.2, seed=42
        )
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    prob_pos, labels, preds = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(DEVICE, non_blocking=True)
            y = batch["label"].cpu().numpy()
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()
            pred = p.argmax(axis=1)
            prob_pos.append(p[:, 1])
            labels.append(y)
            preds.append(pred)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob_pos = np.concatenate(prob_pos)
    metrics = binary_metrics(y_true, y_pred, y_prob_pos)
    metrics.update(
        {
            "checkpoint": ckpt_path.name,
            "best_f1_checkpoint": float(ckpt.get("best_val_f1", -1)),
            "epoch": int(ckpt.get("epoch", -1)),
        }
    )
    return metrics


def main():
    summary = {
        "device": DEVICE,
        "experts": {
            "expert1_nih": evaluate_nih(),
            "expert2_isic": evaluate_isic(),
            "expert3_oa": evaluate_oa(),
            "expert4_luna_mip": evaluate_luna_mip(),
            "expert5_pancreatic": evaluate_pancreatic(),
        },
    }
    out_path = OUT_DIR / "all_experts_eval_vast.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
