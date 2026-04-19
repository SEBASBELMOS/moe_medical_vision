#!/bin/bash
cd /workspace/moe_medical_vision

echo 'Matando procesos viejos...'
pkill -f 'run_luna_vivit_fast.py' || true
pkill -f '03_extract_cls_tokens.py' || true

echo 'Lanzando ViViT...'
nohup python3 -u scripts/run_luna_vivit_fast.py > luna_vivit.log 2>&1 &

echo 'Lanzando Extractor CLS...'
nohup python3 -u scripts/03_extract_cls_tokens.py > cls_extract.log 2>&1 &

echo '¡Procesos lanzados en background (nohup) exitosamente!'
