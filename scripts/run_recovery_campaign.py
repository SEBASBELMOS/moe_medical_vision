from __future__ import annotations

import itertools
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/workspace/moe_medical_vision")
CHECKPOINT_DIR = ROOT / "checkpoints"
METRICS_DIR = CHECKPOINT_DIR / "metrics"
PY = "/venv/main/bin/python"

TARGETS = {
    "luna": 0.65,
    "nih": 0.72,
}
MAX_EXPERIMENTS = 50


def run(cmd: list[str]) -> int:
    print("RUN", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def wait_for_idle():
    while True:
        out = subprocess.check_output(
            "ps -eo pid=,args= | grep -E 'train_luna_mc3_v3.py|train_expert1_nih_improved.py|train_luna_mc3_param.py|train_expert1_nih_param.py' | grep -v grep | grep -v run_recovery_campaign || true",
            shell=True,
            text=True,
        )
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            print(
                "No conflicting training processes found. Starting campaign.",
                flush=True,
            )
            return
        print("Waiting for existing trainings to finish...")
        for l in lines:
            print("  ", l)
        time.sleep(60)


def read_ckpt_metric(name: str, key: str = "best_f1"):
    p = CHECKPOINT_DIR / name
    if not p.exists():
        return None
    import torch

    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck.get(key) or ck.get("best_val_f1")


def luna_campaign():
    current = read_ckpt_metric("expert4_luna_fixed.pth", "best_val_f1")
    if current is not None:
        print(f"Current LUNA best_val_f1={current:.4f}")
    if current is not None and current >= TARGETS["luna"]:
        print("LUNA target already met. Skipping sweep.")
        return

    grid = list(
        itertools.product(
            [0.2, 0.1, 0.3],
            [1e-4, 7e-5, 1.5e-4],
            [1e-3, 5e-4],
            [0.05, 0.0, 0.1],
        )
    )
    tried = 0
    for dropout, lr, wd, ls in grid:
        if tried >= MAX_EXPERIMENTS:
            break
        tag = f"expert4_luna_sweep_{tried + 1:02d}"
        code = run(
            [
                PY,
                str(ROOT / "scripts" / "train_luna_mc3_param.py"),
                "--tag",
                tag,
                "--epochs",
                "40",
                "--batch-size",
                "8",
                "--dropout",
                str(dropout),
                "--lr",
                str(lr),
                "--weight-decay",
                str(wd),
                "--label-smoothing",
                str(ls),
            ]
        )
        tried += 1
        if code != 0:
            continue
        best = read_ckpt_metric(f"{tag}.pth", "best_val_f1")
        print(f"LUNA_RESULT tag={tag} best={best}")
        if best is not None and best >= TARGETS["luna"]:
            print("LUNA target met. Stopping LUNA sweep.")
            return


def nih_campaign():
    current = read_ckpt_metric("expert1_nih_improved_best.pth", "best_f1")
    if current is not None:
        print(f"Current NIH improved best_f1={current:.4f}")
    if current is not None and current >= TARGETS["nih"]:
        print("NIH target already met. Skipping sweep.")
        return

    configs = [
        {
            "tag": "expert1_nih_bce_nosampler_a",
            "loss": "bce",
            "dropout": 0.2,
            "lr_head": 1e-3,
            "lr_backbone": 3e-5,
            "clahe": 2.0,
            "gamma": (0.95, 1.05),
            "epochs": 24,
            "freeze_epochs": 1,
            "warm_ckpt": "expert1_nih_improved_best.pth",
            "no_sampler": True,
            "thr": (0.05, 0.50, 0.025),
        },
        {
            "tag": "expert1_nih_bce_nosampler_b",
            "loss": "bce",
            "dropout": 0.1,
            "lr_head": 1e-3,
            "lr_backbone": 1e-5,
            "clahe": 2.0,
            "gamma": (1.0, 1.0),
            "epochs": 24,
            "freeze_epochs": 0,
            "warm_ckpt": "expert1_nih_enriched_best.pth",
            "no_sampler": True,
            "thr": (0.03, 0.40, 0.02),
        },
        {
            "tag": "expert1_nih_bce_sampler_c",
            "loss": "bce",
            "dropout": 0.2,
            "lr_head": 3e-4,
            "lr_backbone": 3e-5,
            "clahe": 3.0,
            "gamma": (0.9, 1.1),
            "epochs": 24,
            "freeze_epochs": 1,
            "warm_ckpt": "expert1_nih_enriched_best.pth",
            "no_sampler": False,
            "thr": (0.05, 0.50, 0.025),
        },
        {
            "tag": "expert1_nih_asl_nosampler_d",
            "loss": "asl",
            "dropout": 0.2,
            "lr_head": 1e-3,
            "lr_backbone": 3e-5,
            "clahe": 2.0,
            "gamma": (1.0, 1.0),
            "epochs": 24,
            "freeze_epochs": 0,
            "warm_ckpt": "expert1_nih_improved_best.pth",
            "no_sampler": True,
            "thr": (0.03, 0.40, 0.02),
        },
        {
            "tag": "expert1_nih_bce_nosampler_e",
            "loss": "bce",
            "dropout": 0.3,
            "lr_head": 5e-4,
            "lr_backbone": 1e-5,
            "clahe": 1.5,
            "gamma": (1.0, 1.0),
            "epochs": 28,
            "freeze_epochs": 2,
            "warm_ckpt": "expert1_nih_improved_best.pth",
            "no_sampler": True,
            "thr": (0.02, 0.35, 0.015),
        },
    ]

    tried = 0
    for cfg in configs:
        if tried >= MAX_EXPERIMENTS:
            break
        tag = cfg["tag"]
        if (CHECKPOINT_DIR / f"{tag}.pth").exists():
            best = read_ckpt_metric(f"{tag}.pth", "best_f1")
            print(f"NIH_RESULT existing tag={tag} best={best}")
            if best is not None and best >= TARGETS["nih"]:
                print("NIH target met. Stopping NIH sweep.")
                return
            tried += 1
            continue
        cmd = [
            PY,
            str(ROOT / "scripts" / "train_expert1_nih_param.py"),
            "--tag",
            tag,
            "--epochs",
            str(cfg["epochs"]),
            "--freeze-epochs",
            str(cfg["freeze_epochs"]),
            "--batch-size",
            "24",
            "--accum-steps",
            "2",
            "--dropout",
            str(cfg["dropout"]),
            "--lr-head",
            str(cfg["lr_head"]),
            "--lr-backbone",
            str(cfg["lr_backbone"]),
            "--clahe-clip",
            str(cfg["clahe"]),
            "--gamma-min",
            str(cfg["gamma"][0]),
            "--gamma-max",
            str(cfg["gamma"][1]),
            "--loss",
            cfg["loss"],
            "--thr-start",
            str(cfg["thr"][0]),
            "--thr-end",
            str(cfg["thr"][1]),
            "--thr-step",
            str(cfg["thr"][2]),
            "--warm-ckpt",
            cfg["warm_ckpt"],
        ]
        if cfg["no_sampler"]:
            cmd.append("--no-sampler")
        code = run(cmd)
        tried += 1
        if code != 0:
            continue
        best = read_ckpt_metric(f"{tag}.pth", "best_f1")
        print(f"NIH_RESULT tag={tag} best={best}")
        if best is not None and best >= TARGETS["nih"]:
            print("NIH target met. Stopping NIH sweep.")
            return


def main():
    wait_for_idle()
    luna_campaign()
    nih_campaign()
    print("RECOVERY_CAMPAIGN_DONE")


if __name__ == "__main__":
    main()
