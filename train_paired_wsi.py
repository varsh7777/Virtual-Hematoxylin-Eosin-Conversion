from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Optional
import math
import random

import cv2
import numpy as np
from PIL import Image
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class SlidePair:
    raw_path: Path
    target_path: Path
    name: str
    raw_img: np.ndarray
    target_img: np.ndarray
    width: int
    height: int

    @classmethod
    def open(cls, raw_path: str, target_path: str) -> "SlidePair":
        raw_p = Path(raw_path)
        tgt_p = Path(target_path)

        raw_img = _read_image_rgb(raw_p)
        tgt_img = _read_image_rgb(tgt_p)

        raw_h, raw_w = raw_img.shape[:2]
        tgt_h, tgt_w = tgt_img.shape[:2]

        width = min(raw_w, tgt_w)
        height = min(raw_h, tgt_h)

        raw_img = raw_img[:height, :width, :]
        tgt_img = tgt_img[:height, :width, :]

        return cls(
            raw_path=raw_p,
            target_path=tgt_p,
            name=raw_p.stem,
            raw_img=raw_img,
            target_img=tgt_img,
            width=width,
            height=height,
        )

    def read_pair(self, x: int, y: int, tile_size: int) -> Tuple[np.ndarray, np.ndarray]:
        raw_rgb = self.raw_img[y:y + tile_size, x:x + tile_size, :]
        tgt_rgb = self.target_img[y:y + tile_size, x:x + tile_size, :]
        return raw_rgb, tgt_rgb


def _read_image_rgb(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            else:
                raise ValueError(f"Unsupported TIFF channel layout for {path}: {arr.shape}")
        else:
            raise ValueError(f"Unsupported TIFF shape for {path}: {arr.shape}")

        if arr.dtype == np.uint16:
            arr = (arr / 257.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(arr)

    # fallback for png/jpg/etc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)


def _tile_is_valid(
    rgb: np.ndarray,
    white_mean_thresh: float,
    low_std_thresh: float,
) -> bool:
    if rgb.size == 0:
        return False
    if float(rgb.mean()) >= white_mean_thresh:
        return False
    if float(rgb.std()) <= low_std_thresh:
        return False
    return True


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    arr = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _apply_basic_aug(
    raw_rgb: np.ndarray,
    tgt_rgb: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    k = rng.randint(0, 3)
    if k:
        raw_rgb = np.rot90(raw_rgb, k).copy()
        tgt_rgb = np.rot90(tgt_rgb, k).copy()

    if rng.random() < 0.5:
        raw_rgb = np.fliplr(raw_rgb).copy()
        tgt_rgb = np.fliplr(tgt_rgb).copy()

    if rng.random() < 0.5:
        raw_rgb = np.flipud(raw_rgb).copy()
        tgt_rgb = np.flipud(tgt_rgb).copy()

    return raw_rgb, tgt_rgb


class RandomPairedWSIDataset(Dataset):
    def __init__(
        self,
        slide_pairs: Sequence[SlidePair],
        tile_size: int,
        tiles_per_epoch: int,
        white_mean_thresh: float = 245.0,
        low_std_thresh: float = 5.0,
        max_retries: int = 50,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.slide_pairs = list(slide_pairs)
        self.tile_size = int(tile_size)
        self.tiles_per_epoch = int(tiles_per_epoch)
        self.white_mean_thresh = float(white_mean_thresh)
        self.low_std_thresh = float(low_std_thresh)
        self.max_retries = int(max_retries)
        self.augment = bool(augment)
        self.seed = int(seed)

        for p in self.slide_pairs:
            if p.width < self.tile_size or p.height < self.tile_size:
                raise ValueError(
                    f"Slide pair {p.name} is smaller than tile size {self.tile_size}: "
                    f"{p.width}x{p.height}"
                )

    def __len__(self) -> int:
        return self.tiles_per_epoch

    def __getitem__(self, idx: int):
        worker_seed = torch.initial_seed() % (2**32)
        rng = random.Random(self.seed + idx + worker_seed)

        for _ in range(self.max_retries):
            pair = rng.choice(self.slide_pairs)
            max_x = pair.width - self.tile_size
            max_y = pair.height - self.tile_size
            x = rng.randint(0, max_x)
            y = rng.randint(0, max_y)

            raw_rgb, tgt_rgb = pair.read_pair(x, y, self.tile_size)

            if not _tile_is_valid(raw_rgb, self.white_mean_thresh, self.low_std_thresh):
                continue
            if not _tile_is_valid(tgt_rgb, self.white_mean_thresh, self.low_std_thresh):
                continue

            if self.augment:
                raw_rgb, tgt_rgb = _apply_basic_aug(raw_rgb, tgt_rgb, rng)

            return {
                "input": _to_tensor(raw_rgb),
                "target": _to_tensor(tgt_rgb),
                "slide_name": pair.name,
                "xy": (x, y),
            }

        pair = rng.choice(self.slide_pairs)
        x = max(0, (pair.width - self.tile_size) // 2)
        y = max(0, (pair.height - self.tile_size) // 2)
        raw_rgb, tgt_rgb = pair.read_pair(x, y, self.tile_size)
        return {
            "input": _to_tensor(raw_rgb),
            "target": _to_tensor(tgt_rgb),
            "slide_name": pair.name,
            "xy": (x, y),
        }


class FixedValPairedWSIDataset(Dataset):
    def __init__(
        self,
        slide_pairs: Sequence[SlidePair],
        tile_size: int,
        tiles_per_slide: int,
        white_mean_thresh: float = 245.0,
        low_std_thresh: float = 5.0,
        max_retries_per_tile: int = 200,
        seed: int = 1337,
    ) -> None:
        self.samples: List[Tuple[SlidePair, int, int]] = []
        self.tile_size = int(tile_size)
        rng = random.Random(seed)

        for pair in slide_pairs:
            max_x = pair.width - self.tile_size
            max_y = pair.height - self.tile_size
            found = 0
            tries = 0
            while found < tiles_per_slide and tries < tiles_per_slide * max_retries_per_tile:
                tries += 1
                x = rng.randint(0, max_x)
                y = rng.randint(0, max_y)
                raw_rgb, tgt_rgb = pair.read_pair(x, y, self.tile_size)
                if not _tile_is_valid(raw_rgb, white_mean_thresh, low_std_thresh):
                    continue
                if not _tile_is_valid(tgt_rgb, white_mean_thresh, low_std_thresh):
                    continue
                self.samples.append((pair, x, y))
                found += 1

            if found == 0:
                raise RuntimeError(f"Could not sample any validation tiles from slide {pair.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pair, x, y = self.samples[idx]
        raw_rgb, tgt_rgb = pair.read_pair(x, y, self.tile_size)
        return {
            "input": _to_tensor(raw_rgb),
            "target": _to_tensor(tgt_rgb),
            "slide_name": pair.name,
            "xy": (x, y),
        }


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv((c_in // 2) + c_skip, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.bot = DoubleConv(base * 8, base * 16)

        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bot(x4)

        x = self.up1(xb, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            pred = model(x)
            loss = F.l1_loss(pred, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def train_paired_wsi(
    pairs: Sequence[Tuple[str, str]],
    out_dir: str,
    epochs: int = 20,
    tiles_per_epoch: int = 1000,
    val_tiles_per_slide: int = 128,
    batch_size: int = 8,
    tile_size: int = 512,
    lr: float = 1e-4,
    num_workers: int = 0,
    seed: int = 123,
    white_mean_thresh: float = 245.0,
    low_std_thresh: float = 5.0,
    resume_checkpoint: Optional[str] = None,
) -> None:
    if len(pairs) == 0:
        raise ValueError("At least one --pair RAW_TIF TARGET_TIF is required.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    slide_pairs = [SlidePair.open(raw, tgt) for raw, tgt in pairs]

    train_ds = RandomPairedWSIDataset(
        slide_pairs=slide_pairs,
        tile_size=tile_size,
        tiles_per_epoch=tiles_per_epoch,
        white_mean_thresh=white_mean_thresh,
        low_std_thresh=low_std_thresh,
        augment=True,
        seed=seed,
    )
    val_ds = FixedValPairedWSIDataset(
        slide_pairs=slide_pairs,
        tile_size=tile_size,
        tiles_per_slide=val_tiles_per_slide,
        white_mean_thresh=white_mean_thresh,
        low_std_thresh=low_std_thresh,
        seed=seed + 999,
    )

    train_loader = _make_loader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 1
    best_val = math.inf

    if resume_checkpoint:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", math.inf))

    print(f"Training on {len(slide_pairs)} paired slides.")
    print(f"Device: {device}")
    print(f"Tile size: {tile_size}")
    print(f"Tiles per epoch: {tiles_per_epoch}")
    print(f"Validation tiles per slide: {val_tiles_per_slide}")

    for epoch in range(start_epoch, epochs + 1):
        train_loss = _epoch_pass(model, train_loader, device, optimizer=optimizer)
        val_loss = _epoch_pass(model, val_loader, device, optimizer=None)

        print(f"Epoch {epoch:03d}/{epochs:03d}  train_l1={train_loss:.6f}  val_l1={val_loss:.6f}")

        last_ckpt = out_path / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "pairs": [(str(a), str(b)) for a, b in pairs],
                "tile_size": tile_size,
            },
            last_ckpt,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = out_path / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "pairs": [(str(a), str(b)) for a, b in pairs],
                    "tile_size": tile_size,
                },
                best_ckpt,
            )
            print(f"  saved best -> {best_ckpt}")