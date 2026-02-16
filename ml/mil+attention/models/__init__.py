import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    pred/target: [B,3,H,W] in [0,1]
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    import math
    return 10.0 * math.log10((max_val * max_val) / mse)


@torch.no_grad()
def l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()