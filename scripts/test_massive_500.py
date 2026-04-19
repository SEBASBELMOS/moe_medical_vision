import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import faiss
import joblib

sys.path.insert(0, "/workspace/moe_medical_vision/src")
from models.moe_system import MoE_System
from data.datasets import get_dataloader

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Cargando MoE_System y los 5 Expertos Definitivos...")
model = MoE_System(device=device)
model.load_all_weights("/workspace/moe_medical_vision/checkpoints")
model.eval()


def process_3d_npz(file_path):
    d = np.load(file_path)
    vol = torch.from_numpy(d["volume"][0].copy()).float()
    y = int(d["label"])
    x_3d = vol.unsqueeze(0).unsqueeze(0)
    mip = torch.stack([vol.max(dim=0)[0], vol.mean(dim=0), vol.std(dim=0)], dim=0)
    x_224 = F.interpolate(
        mip.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    )
    return x_224, x_3d, y


results = {
    i: {
        "router_correct": 0,
        "total": 0,
        "y_true": [],
        "y_pred": [],
        "classes_pred": set(),
    }
    for i in range(5)
}

print(
    "\nIniciando Test Masivo de 500 Imágenes (100 por Experto)... Esto tomará un minuto.\n"
)

fast_dirs = {
    3: "/workspace/moe_medical_vision/data/processed/luna16_highres",
    4: "/workspace/moe_medical_vision/data/processed/pancreatic_fast",
}

for exp_id, root in fast_dirs.items():
    files = list(Path(root).glob("val_*.npz"))
    selected = random.sample(files, min(100, len(files)))
    for f in selected:
        x_224, x_3d, y_true = process_3d_npz(f)
        with torch.no_grad():
            out, best_idx, _ = model(x_224.to(device), x_3d=x_3d.to(device))

        results[exp_id]["total"] += 1
        if best_idx == exp_id:
            results[exp_id]["router_correct"] += 1
        pred = torch.argmax(out, dim=-1).item()
        results[exp_id]["y_true"].append(y_true)
        results[exp_id]["y_pred"].append(pred)
        results[exp_id]["classes_pred"].add(pred)

ds_names = {
    0: ("nih_chestxray", "/workspace/moe_medical_vision/data/raw/nih"),
    1: ("isic2019", "/workspace/moe_medical_vision/data/raw/isic"),
    2: ("osteoarthritis", "/workspace/moe_medical_vision/data/raw/osteoporosis/train"),
}

for exp_id, (name, root) in ds_names.items():
    try:
        loader, ds = get_dataloader(
            name, root, split="val", batch_size=1, num_workers=0
        )
    except Exception:
        continue
    indices = random.sample(range(len(ds)), min(100, len(ds)))

    for idx in indices:
        item = ds[idx]
        x_224 = item["image"].unsqueeze(0)
        if x_224.shape[1] == 1:
            x_224 = x_224.repeat(1, 3, 1, 1)
        if x_224.shape[-1] != 224:
            x_224 = F.interpolate(
                x_224, size=(224, 224), mode="bilinear", align_corners=False
            )

        if exp_id == 0:
            y_true_vec = item["label"].numpy()
        else:
            y_true = item["label"]

        with torch.no_grad():
            out, best_idx, _ = model(x_224.to(device))

        results[exp_id]["total"] += 1
        if best_idx == exp_id:
            results[exp_id]["router_correct"] += 1
        pred = torch.argmax(out, dim=-1).item()

        if exp_id == 0:
            results[exp_id]["y_true"].append(y_true_vec)
        else:
            results[exp_id]["y_true"].append(y_true)
        results[exp_id]["y_pred"].append(pred)
        results[exp_id]["classes_pred"].add(pred)

NAMES = {
    0: "NIH ChestX-ray14",
    1: "ISIC 2019 (Piel)",
    2: "Osteoarthritis",
    3: "LUNA16 (Pulmón)",
    4: "Pancreatic Cancer",
}
print("=" * 80)
print("  RESULTADOS FINALES DEL PIPELINE END-TO-END (500 IMAGENES)")
print("=" * 80)

total_router = sum(r["router_correct"] for r in results.values())
total_imgs = sum(r["total"] for r in results.values())

for exp_id in range(5):
    r = results[exp_id]
    n = r["total"]
    if n == 0:
        continue
    print(f"\n--- Experto {exp_id}: {NAMES[exp_id]} ---")
    print(
        f"➜ Router Accuracy: {r['router_correct']}/{n} ({(r['router_correct'] / n) * 100:.1f}%)"
    )

    if exp_id == 0:
        correct_preds = 0
        for yt, yp in zip(r["y_true"], r["y_pred"]):
            if yt[yp] == 1:
                correct_preds += 1
        print(
            f"➜ Expert Accuracy (Hit Multi-label): {correct_preds}/{n} ({(correct_preds / n) * 100:.1f}%)"
        )
    else:
        acc = accuracy_score(r["y_true"], r["y_pred"])
        f1 = f1_score(r["y_true"], r["y_pred"], average="macro")
        print(f"➜ Expert Accuracy: {acc * 100:.1f}% | Expert F1 Macro: {f1:.4f}")

    if len(r["classes_pred"]) == 1:
        print(
            f"  ⚠️ ¡ALERTA! El modelo predijo la misma clase ({list(r['classes_pred'])[0]}) para las {n} imágenes."
        )
    else:
        print(f"  ✅ Modelo sano. Predijo {len(r['classes_pred'])} clases diferentes.")

print("\n" + "=" * 80)
print(
    f"ROUTING GLOBAL DEL SISTEMA: {total_router}/{total_imgs} ({(total_router / total_imgs) * 100:.1f}%)"
)
print("=" * 80)
