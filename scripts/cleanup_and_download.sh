#!/bin/bash
# ==============================================================================
# MOE Medical Vision - Cleanup & Resume Downloads
# ==============================================================================

echo "============================================================"
echo "  MOE Medical Vision - Cleanup & Resume"
echo "============================================================"

# Check disk space first
echo ""
echo "[1] Espacio en disco:"
df -h /

# Create directories if needed
echo ""
echo "[2] Creando directorios..."
mkdir -p /workspace/moe_medical_vision/data/raw/{nih,isic,pancreatic}

# Install kagglehub if not installed
echo ""
echo "[3] Verificando dependencias..."
pip show kagglehub > /dev/null 2>&1 || pip install kagglehub -q

# ==============================================================================
# DOWNLOAD FUNCTIONS
# ==============================================================================

download_nih() {
    echo ""
    echo "[DOWNLOAD] NIH ChestX-ray14 (~45GB)"
    echo "============================================================"
    
    NIH_DIR="/workspace/moe_medical_vision/data/raw/nih"
    
    # Remove old if exists
    if [ -d "$NIH_DIR" ] && [ "$(ls -A $NIH_DIR)" ]; then
        echo "Eliminando NIH anterior..."
        rm -rf "$NIH_DIR"
        mkdir -p "$NIH_DIR"
    fi
    
    # Download using kagglehub
    python3 << EOF
import kagglehub
from pathlib import Path
import shutil

NIH_DIR = Path("$NIH_DIR")
print(f"Descargando NIH a {NIH_DIR}...")
path = kagglehub.dataset_download("nih-chest-xrays/data")
print(f"Descarga completada: {path}")

# Move files
for item in Path(path).iterdir():
    dst = NIH_DIR / item.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(item), str(dst))
Path(path).rmdir()

print("NIH descargado!")
EOF
}

download_isic() {
    echo ""
    echo "[DOWNLOAD] ISIC 2019 (~5GB)"
    echo "============================================================"
    
    ISIC_DIR="/workspace/moe_medical_vision/data/raw/isic"
    kaggle datasets download -d andrewmvd/isic-2019 -p "$ISIC_DIR" --unzip
}

download_pancreatic() {
    echo ""
    echo "[DOWNLOAD] Pancreatic Cancer (~46GB from Zenodo)"
    echo "============================================================"
    
    DEST="/workspace/moe_medical_vision/data/raw/pancreatic"
    ZIP_FILE="$DEST/batch_1.zip"
    
    wget --progress=bar:force -O "$ZIP_FILE" \
        "https://zenodo.org/records/13715870/files/batch_1.zip"
    
    if [ -f "$ZIP_FILE" ]; then
        echo "Extrayendo..."
        unzip -q "$ZIP_FILE" -d "$DEST"
        rm "$ZIP_FILE"  # Free space
        echo "Pancreatic Cancer listo!"
    fi
}

# ==============================================================================
# MAIN MENU
# ==============================================================================

echo ""
echo "Que deseas hacer?"
echo "  1) Descargar NIH (kagglehub con progress)"
echo "  2) Descargar ISIC (Kaggle CLI)"
echo "  3) Descargar Pancreatic (Zenodo ~46GB)"
echo "  4) Descargar todo"
echo "  5) Solo reporte de integridad"

read -p "Opcion (1-5): " opcion

case $opcion in
    1) download_nih ;;
    2) download_isic ;;
    3) download_pancreatic ;;
    4) 
        download_nih
        download_isic  
        download_pancreatic
        ;;
    5) 
        echo ""
        echo "[REPORTE DE INTEGRIDAD]"
        echo "============================================================"
        for ds in nih isic osteoporosis luna16 pancreatic; do
            DIR="/workspace/moe_medical_vision/data/raw/$ds"
            if [ -d "$DIR" ] && [ "$(ls -A $DIR)" ]; then
                SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
                COUNT=$(find "$DIR" -type f 2>/dev/null | wc -l)
                echo "$ds: $COUNT archivos, $SIZE"
            else
                echo "$ds: Vacio"
            fi
        done
        ;;
    *) echo "Opcion invalida" ;;
esac

# Final report
echo ""
echo "============================================================"
echo "[FINAL] Espacio en disco:"
df -h /
echo "============================================================"
