import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm

np.random.seed(42)

RAW_ROOT = Path("/workspace/moe_medical_vision/data/raw/luna16")
MHD_DIR = RAW_ROOT / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
CAND_CSV = RAW_ROOT / "candidates.csv"
OUT_DIR = Path("/workspace/moe_medical_vision/data/processed/luna16_highres")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Cargando candidatos desde CSV...")
df = pd.read_csv(CAND_CSV)
available_mhd = {p.stem: p for p in MHD_DIR.glob("*.mhd")}
df = df[df["seriesuid"].isin(available_mhd.keys())].copy()

# SPLIT POR PACIENTE (No por candidato, evita leakage)
unique_patients = df["seriesuid"].unique()
np.random.shuffle(unique_patients)
n_train = int(len(unique_patients) * 0.8)
train_patients = set(unique_patients[:n_train])
val_patients = set(unique_patients[n_train:])

print(
    f"Total Pacientes: {len(unique_patients)} (Train: {len(train_patients)}, Val: {len(val_patients)})"
)

df_train = df[df["seriesuid"].isin(train_patients)]
df_val = df[df["seriesuid"].isin(val_patients)]


# RATIO 1:2 (1 Positivo por cada 2 Negativos)
def select_candidates(split_df, ratio=2):
    pos = split_df[split_df["class"] == 1]
    neg = split_df[split_df["class"] == 0]
    n_neg = min(len(neg), len(pos) * ratio)
    neg_sampled = neg.sample(n=n_neg, random_state=42)
    return pd.concat([pos, neg_sampled]).sample(frac=1.0, random_state=42)


train_selected = select_candidates(df_train, ratio=2)
val_selected = select_candidates(df_val, ratio=2)

print(
    f"Patches de Train: {len(train_selected)} (Nódulos: {train_selected['class'].sum()})"
)
print(f"Patches de Val: {len(val_selected)} (Nódulos: {val_selected['class'].sum()})")

all_selected = pd.concat([train_selected, val_selected])
all_selected["split"] = np.where(
    all_selected["seriesuid"].isin(train_patients), "train", "val"
)

# AGRUPAR POR PACIENTE PARA LEER EL MHD UNA SOLA VEZ
grouped = all_selected.groupby("seriesuid")


def extract_patch(vol, z, y, x, patch_size=64, pad_value=-1000.0):
    half = patch_size // 2
    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    patch = np.full((patch_size, patch_size, patch_size), pad_value, dtype=np.float32)

    sz0, sz1 = max(0, z0), min(vol.shape[0], z1)
    sy0, sy1 = max(0, y0), min(vol.shape[1], y1)
    sx0, sx1 = max(0, x0), min(vol.shape[2], x1)

    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    dz1 = dz0 + (sz1 - sz0)
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)

    patch[dz0:dz1, dy0:dy1, dx0:dx1] = vol[sz0:sz1, sy0:sy1, sx0:sx1]
    return patch


metadata = []

print("\nExtrayendo parches en Alta Resolucion y Normalizando HU...")
counter_train = 0
counter_val = 0

for seriesuid, group in tqdm(grouped, total=len(grouped)):
    mhd_path = available_mhd[seriesuid]
    try:
        img = sitk.ReadImage(str(mhd_path))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)

        for _, row in group.iterrows():
            coord = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]))
            idx = img.TransformPhysicalPointToIndex(coord)
            x, y, z = idx[0], idx[1], idx[2]

            patch = extract_patch(vol, z, y, x, patch_size=64, pad_value=-1000.0)

            # Normalizacion HU y conversion a escala 0-1
            patch = np.clip(patch, -1000.0, 400.0)
            patch = (patch + 1000.0) / 1400.0

            split = row["split"]
            label = int(row["class"])

            if split == "train":
                filename = f"train_{counter_train:05d}.npz"
                counter_train += 1
            else:
                filename = f"val_{counter_val:05d}.npz"
                counter_val += 1

            out_path = OUT_DIR / filename
            np.savez_compressed(
                out_path,
                volume=np.expand_dims(patch, axis=0),
                label=label,
                seriesuid=seriesuid,
            )

            metadata.append(
                {
                    "filename": filename,
                    "seriesuid": seriesuid,
                    "class": label,
                    "split": split,
                }
            )
    except Exception as e:
        print(f"Error procesando {seriesuid}: {e}")

pd.DataFrame(metadata).to_csv(OUT_DIR / "metadata.csv", index=False)
print(
    f"\nExtraccion completada con exito. Total generados: {len(metadata)} parches HD."
)
