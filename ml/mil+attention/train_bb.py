# train_bb.py
from __future__ import annotations

import os
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.unet_diffusion import UNetDiffusion
from models.scheduler import BrownianBridgeScheduler
from models.bb_diffusion import BrownianBridgeDiffusion
from losses.diffusion import BridgeDiffusionLoss
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint


def _list_files(root: str, exts: List[str]) -> List[str]:
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(root, f"*{e}")))
        out.extend(glob.glob(os.path.join(root, f"*{e.upper()}")))
    return sorted(set(out))


class PairedPatchDataset(Dataset):
    """
    Assumes flat paired patch folders:
      raw_dir/<name>.png
      he_dir/<name>.png
    If images are larger than patch_size, a random crop is taken.
    """
    def __init__(self, raw_dir: str, he_dir: str, exts: List[str], patch_size: int):
        self.raw_dir = raw_dir
        self.he_dir = he_dir
        self.raw_paths = _list_files(raw_dir, exts)
        self.patch = int(patch_size)

    def __len__(self):
        return len(self.raw_paths)

    def _read_rgb01(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    def __getitem__(self, idx: int):
        raw_path = self.raw_paths[idx]
        name = os.path.basename(raw_path)
        he_path = os.path.join(self.he_dir, name)

        if not os.path.exists(he_path):
            raise FileNotFoundError(f"Missing target for {name}: expected {he_path}")

        x = self._read_rgb01(raw_path)
        y = self._read_rgb01(he_path)

        H, W = x.shape[:2]
        p = self.patch

        if H < p or W < p:
            raise ValueError(f"Image smaller than patch size: {raw_path} shape=({H},{W}) patch={p}")

        if H != p or W != p:
            y0 = np.random.randint(0, H - p + 1)
            x0 = np.random.randint(0, W - p + 1)
            x = x[y0:y0 + p, x0:x0 + p, :]
            y = y[y0:y0 + p, x0:x0 + p, :]

        x = torch.from_numpy(x).permute(2, 0, 1)  # [3,H,W]
        y = torch.from_numpy(y).permute(2, 0, 1)  # [3,H,W]
        return {"x": x, "y": y, "name": name}


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bb_diffusion.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = PairedPatchDataset(
        raw_dir=cfg["data"]["raw_dir"],
        he_dir=cfg["data"]["he_dir"],
        exts=cfg["data"]["ext"],
        patch_size=int(cfg["data"]["patch_size"]),
    )

    if len(ds) == 0:
        raise RuntimeError(
            f"No files found in raw_dir={cfg['data']['raw_dir']} "
            f"with ext={cfg['data']['ext']}"
        )

    print("dataset size:", len(ds))

    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    denoiser = UNetDiffusion(
        in_ch=3,
        cond_ch=3,
        out_ch=3,
        base_ch=int(cfg["model"]["base_channels"]),
        time_dim=int(cfg["model"]["time_dim"]),
    ).to(device)

    scheduler = BrownianBridgeScheduler(
        num_steps=int(cfg["diffusion"]["num_steps"]),
        sigma_min=float(cfg["diffusion"]["sigma_min"]),
        sigma_max=float(cfg["diffusion"]["sigma_max"]),
        device=str(device),
    ).to(device)

    model = BrownianBridgeDiffusion(denoiser=denoiser, scheduler=scheduler).to(device)

    loss_fn = BridgeDiffusionLoss(
        lambda_eps=float(cfg["loss"]["lambda_eps"]),
        lambda_x0=float(cfg["loss"]["lambda_x0"]),
        use_l1_for_x0=bool(cfg["loss"]["use_l1_for_x0"]),
    ).to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.999),
    )

    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best = 1e9

    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and bool(cfg["train"]["amp"]):
            scaler.load_state_dict(ckpt["scaler"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best" in ckpt:
            best = float(ckpt["best"])

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(pbar, 1):
            x = batch["x"].to(device, non_blocking=True)  # raw condition
            y = batch["y"].to(device, non_blocking=True)  # target H&E

            B = x.shape[0]
            t = scheduler.sample_timesteps(B, device)

            with autocast(enabled=bool(cfg["train"]["amp"])):
                out = model.forward_train(x, y, t)
                loss_dict = loss_fn(
                    eps_pred=out["eps_pred"],
                    eps_true=out["noise"],
                    y_pred=out["y_pred"],
                    y_true=y,
                )
                loss = loss_dict["loss"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if float(cfg["train"]["grad_clip"]) > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))

            scaler.step(opt)
            scaler.update()

            running += float(loss.detach())

            if step % int(cfg["train"]["log_every"]) == 0:
                print(
                    f"epoch {epoch:03d} step {step:05d}/{len(dl)} "
                    f"loss={running/step:.4f} "
                    f"eps={float(loss_dict['loss_eps'].detach()):.4f} "
                    f"x0={float(loss_dict['loss_x0'].detach()):.4f}"
                )

            pbar.set_postfix({
                "loss": float(loss.detach()),
                "eps": float(loss_dict["loss_eps"].detach()),
                "x0": float(loss_dict["loss_x0"].detach()),
            })

        avg_loss = running / max(1, len(dl))

        last_path = out_dir / "last.pt"
        save_checkpoint(
            str(last_path),
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "config": cfg,
            },
        )
        print("saved:", last_path)

        if bool(cfg["output"]["save_best"]) and avg_loss < best:
            best = avg_loss
            best_path = out_dir / "best.pt"
            save_checkpoint(
                str(best_path),
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "config": cfg,
                },
            )
            print("saved BEST:", best_path, "loss=", best)


if __name__ == "__main__":
    main()