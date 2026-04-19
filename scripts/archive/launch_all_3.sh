#!/bin/bash
cd /workspace/moe_medical_vision

nohup python3 -u scripts/run_train_luna_v7_freeze.py > luna_v7.log 2>&1 &
V7=$!

nohup python3 -u scripts/run_train_luna_v8_candidate3d.py > luna_v8.log 2>&1 &
V8=$!

nohup python3 -u scripts/run_train_pancreatic_v5_finetune.py > panc_v5.log 2>&1 &
P5=$!

echo "v7=$V7 v8=$V8 p5=$P5" > active_pids.txt
echo "Launched: v7=$V7 v8=$V8 p5=$P5"
