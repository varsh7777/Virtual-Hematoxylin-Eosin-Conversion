# train.py
# handles model training
# loads config, builds dataloaders, model, losses, optimizer, trains/validates, saves checkpoints

import os
import glob
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from losses import PatchL1Loss, GANLoss
from models import UNetGenerator, MILAttention, PatchDiscriminator
from utils import set_seed, save_checkpoint


def _list_files(root: str, exts: List[str]) -> List[str]:
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(root, f"*{e}")))
    return sorted(out)


class PairedBagDataset(Dataset):
    """
    Bag-of-patches dataset from paired full images.

    Expects:
      raw_dir/<name>.<ext>
      he_dir/<name>.<ext>  (same filename)

    Returns:
      x:    [N,3,P,P] float in [0,1]
      y_he: [N,3,P,P] float in [0,1]
      name: str
    """
    def __init__(self, raw_dir: str, he_dir: str, exts: List[str], patch_size: int, bag_size: int):
        self.raw_paths = _list_files(raw_dir, exts)
        self.he_dir = he_dir
        self.patch = patch_size
        self.N = bag_size

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

        raw = self._read_rgb01(raw_path)
        he = self._read_rgb01(he_path)

        H, W = raw.shape[:2]
        p = self.patch

        # sample N random patches
        xs = []
        ys = []
        for _ in range(self.N):
            y0 = np.random.randint(0, max(1, H - p + 1))
            x0 = np.random.randint(0, max(1, W - p + 1))
            xs.append(raw[y0:y0+p, x0:x0+p, :])
            ys.append(he[y0:y0+p, x0:x0+p, :])

        x = torch.from_numpy(np.stack(xs, 0)).permute(0, 3, 1, 2)     # [N,3,P,P]
        y = torch.from_numpy(np.stack(ys, 0)).permute(0, 3, 1, 2)     # [N,3,P,P]
        return {"x": x, "y_he": y, "name": name}


def collate_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch size B of dicts with x:[N,3,P,P]
    x = torch.stack([b["x"] for b in batch], dim=0)      # [B,N,3,P,P]
    y = torch.stack([b["y_he"] for b in batch], dim=0)   # [B,N,3,P,P]
    names = [b["name"] for b in batch]
    return {"x": x, "y_he": y, "names": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PairedBagDataset(
        raw_dir=cfg["data"]["raw_dir"],
        he_dir=cfg["data"]["he_dir"],
        exts=cfg["data"]["ext"],
        patch_size=int(cfg["data"]["patch_size"]),
        bag_size=int(cfg["data"]["bag_size"]),
    )
    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_bags,
    )

    G = UNetGenerator(base_channels=int(cfg["model"]["generator"]["base_channels"])).to(device)
    MIL = MILAttention(
        embed_dim=int(cfg["model"]["encoder"]["embed_dim"]),
        attn_type=str(cfg["model"]["attention"]["type"]),
    ).to(device)

    use_gan = float(cfg["loss"]["lambda_gan"]) > 0
    if use_gan:
        D = PatchDiscriminator(base_channels=int(cfg["gan"]["d_base_channels"])).to(device)
        gan_loss = GANLoss(cfg["gan"]["gan_mode"]).to(device)
        opt_d = optim.AdamW(D.parameters(), lr=float(cfg["train"]["lr"]), betas=(0.5, 0.999))
    else:
        D = None
        gan_loss = None
        opt_d = None

    recon_loss = PatchL1Loss().to(device)

    params = list(G.parameters()) + list(MIL.parameters())
    opt = optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.5, 0.999),
    )

    scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

    lambda_recon = float(cfg["loss"]["lambda_recon"])
    lambda_gan = float(cfg["loss"]["lambda_gan"])
    attn_weighted = bool(cfg["loss"]["attention_weighted_recon"])

    best = 1e9

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        G.train()
        MIL.train()
        if D is not None:
            D.train()

        running = 0.0

        for step, batch in enumerate(dl, 1):
            x = batch["x"].to(device, non_blocking=True)       # [B,N,3,P,P]
            y = batch["y_he"].to(device, non_blocking=True)    # [B,N,3,P,P]
            B, N, C, P, _ = x.shape

            # ---- forward generator per patch ----
            x_flat = x.view(B * N, C, P, P)
            with autocast(enabled=bool(cfg["train"]["amp"])):
                yhat_flat = G(x_flat)                          # [B*N,3,P,P]
                yhat = yhat_flat.view(B, N, 3, P, P)

                mil_out = MIL(x)                               # attention over raw patches
                attn = mil_out["attn"]                         # [B,N]

                patch_l1_map, patch_l1_mean = recon_loss(yhat, y)
                if attn_weighted:
                    loss_recon = (patch_l1_map * attn).sum(dim=1).mean()
                else:
                    loss_recon = patch_l1_mean

                loss_g = lambda_recon * loss_recon

                # ---- optional GAN: generator loss ----
                if use_gan:
                    # D expects [B,3,P,P], so use patch batch [B*N,3,P,P]
                    pred_fake = D(yhat_flat)
                    loss_g_gan = gan_loss(pred_fake, True)
                    loss_g = loss_g + lambda_gan * loss_g_gan

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()

            if float(cfg["train"]["grad_clip"]) > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, float(cfg["train"]["grad_clip"]))

            scaler.step(opt)
            scaler.update()

            # ---- optional GAN: discriminator step ----
            if use_gan:
                with torch.no_grad():
                    y_flat = y.view(B * N, 3, P, P)

                with autocast(enabled=bool(cfg["train"]["amp"])):
                    pred_real = D(y_flat)
                    pred_fake = D(yhat_flat.detach())
                    loss_d = 0.5 * (gan_loss(pred_real, True) + gan_loss(pred_fake, False))

                opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                scaler.update()

            running += float(loss_recon.detach())

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"epoch {epoch:03d} step {step:05d}/{len(dl)} recon={running/step:.4f}")

        avg_recon = running / max(1, len(dl))

        # save last
        last_path = out_dir / "last.pt"
        save_checkpoint(str(last_path), {"G": G.state_dict(), "MIL": MIL.state_dict()})
        print("saved:", last_path)

        # save best
        if bool(cfg["output"]["save_best"]) and avg_recon < best:
            best = avg_recon
            best_path = out_dir / "best.pt"
            save_checkpoint(str(best_path), {"G": G.state_dict(), "MIL": MIL.state_dict()})
            print("saved BEST:", best_path, "recon=", best)


if __name__ == "__main__":
    main()
