import pandas as pd
from pathlib import Path

RAW_ROOT = Path("/workspace/moe_medical_vision/data/raw/luna16")
MHD_DIR = RAW_ROOT / "seg-lungs-LUNA16" / "seg-lungs-LUNA16"
CAND_CSV = RAW_ROOT / "candidates.csv"

print("Escanenado archivos .mhd disponibles...")
available_mhd = set([p.stem for p in MHD_DIR.glob("*.mhd")])
print(f"Total .mhd files disponibles: {len(available_mhd)}")

if not CAND_CSV.exists():
    print(f"ERROR: No se encuentra {CAND_CSV}")
else:
    df = pd.read_csv(CAND_CSV)
    df_avail = df[df["seriesuid"].isin(available_mhd)]

    pos = df_avail[df_avail["class"] == 1]
    neg = df_avail[df_avail["class"] == 0]

    print(f"\nTotal candidatos en los archivos disponibles: {len(df_avail)}")
    print(f"  -> Positivos (Nódulos reales): {len(pos)}")
    print(f"  -> Negativos (Falsos positivos): {len(neg)}")
