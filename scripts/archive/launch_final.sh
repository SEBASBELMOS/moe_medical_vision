#!/bin/bash
cd /workspace/moe_medical_vision
nohup python3 -u scripts/run_luna_final.py > luna_final.log 2>&1 &
LUNA=$!
nohup python3 -u scripts/run_panc_final.py > panc_final.log 2>&1 &
PANC=$!
echo "luna=$LUNA panc=$PANC" > active_pids.txt
echo "Launched: luna=$LUNA panc=$PANC"
