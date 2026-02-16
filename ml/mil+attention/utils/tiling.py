from typing import List, Tuple


def tile_coords(H: int, W: int, tile: int, overlap: int) -> Tuple[List[int], List[int]]:
    stride = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    return ys, xs