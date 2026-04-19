#!/usr/bin/env python3
import os, sys, subprocess, requests, zipfile, tarfile
from pathlib import Path
RAW_DIR = Path("/workspace/moe_medical_vision/data/raw")

def get_folder_size(folder):
    total = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def count_files(folder):
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        count += len(filenames)
    return count

def download_kaggle(slug, dest):
    print(f"[KAGGLE] {slug}")
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("  FALLO: define KAGGLE_USERNAME y KAGGLE_KEY en el entorno")
        return False
    try:
        d = RAW_DIR / dest
        d.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(d), "--unzip"], capture_output=True, text=True, timeout=600)
        print(f"  OK" if r.returncode == 0 else f"  FALLO: {r.stderr[:100]}")
        return r.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def download_zenodo(rid, dest):
    print(f"[ZENODO] {rid}")
    try:
        d = RAW_DIR / dest
        d.mkdir(parents=True, exist_ok=True)
        resp = requests.get(f"https://zenodo.org/api/records/{rid}", timeout=30)
        resp.raise_for_status()
        for f in resp.json().get("files", []):
            fn = f.get("filename", "")
            if any(fn.endswith(ext) for ext in [".zip", ".tar", ".gz", ".tar.gz"]):
                url = f"https://zenodo.org/records/{rid}/files/{fn}"
                print(f"  Descargando {fn}")
                subprocess.run(["wget", "-q", "-O", str(d/fn), url], timeout=3600)
                if fn.endswith(".zip"):
                    zipfile.ZipFile(d/fn).extractall(d)
                elif fn.endswith((".tar", ".gz", ".tar.gz")):
                    try:
                        tarfile.open(d/fn).extractall(d)
                    except:
                        pass
                print(f"  OK {fn}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def setup_hf():
    print("[HUGGINGFACE]")
    if not os.environ.get("HF_TOKEN"):
        print("  FALLO: define HF_TOKEN en el entorno")
        return False
    try:
        from huggingface_hub import HfApi
        url = HfApi().create_repo(name="moe-medical-vision-raw", repo_type="dataset", private=True, exist_ok=True)
        print(f"  OK: {url}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

results = {}
results["luna16"] = download_kaggle("fanbyprinciple/luna-lung-cancer-dataset", "luna16")
results["osteoporosis"] = download_kaggle("dhruvacube/osteoarthritis", "osteoporosis")
results["pancreatic"] = download_zenodo("13715870", "pancreatic")
results["hf_repo"] = setup_hf()

print("="*60)
print(" REPORTE DE INTEGRIDAD DE DATOS")
print("="*60)
for ds in ["luna16", "osteoporosis", "pancreatic"]:
    f = RAW_DIR / ds
    n = count_files(f) if f.exists() else 0
    s = get_folder_size(f) / (1024**3) if f.exists() else 0
    print(f"{ds:<20} {n:<12} {s:.4f} GB")
print("="*60)
for k, v in results.items():
    print(f"  {k}: {'OK' if v else 'FALLO'}")
