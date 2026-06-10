"""Track B: deep-vision prototype -- per-frame whisker segmentation (GPU).

Trains the repo U-Net to segment whiskers (one class per whisker) directly from the image,
to test whether a 'modern CV' model beats the learned add-on (Track A) on identity and can
recover whiskers whisk never traced (coverage at the source). Masks are dilated from the
1-px GT polylines so the thin targets are learnable. Trained on a contiguous train window;
the held-out window is reserved for the head-to-head benchmark.

Eval adapter: relabel the existing combined detections by majority-vote of the predicted
mask along each whisker's skeleton -> a vision-identity parquet scored by the SAME
eval_linking harness, head-to-head with Track A.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .train_unet import UNet, VideoParquetDataset, build_mask

_E = r"E:/Thigmotaxis"
WID2CLS = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}   # background = 0
CLS2WID = {v: k for k, v in WID2CLS.items()}
RESIZE = (256, 256)
VIDEO = f"{_E}/whisker_active/sc013_active_enhanced.mp4"
GT = f"{_E}/whisker_active/sc013_active_updated_edited - backup.parquet"
MODEL = f"{_E}/_autotune/models/unet_sc013.pt"


class _Dilated(Dataset):
    """Wrap VideoParquetDataset to dilate the 1-px whisker masks (learnable targets)."""
    def __init__(self, base, k=3):
        self.base = base
        self.kernel = np.ones((k, k), np.uint8)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, mask = self.base[i]
        m = mask.numpy().astype(np.uint8) if torch.is_tensor(mask) else np.asarray(mask, np.uint8)
        out = np.zeros_like(m)
        for c in range(1, 7):           # dilate each whisker class separately
            d = cv2.dilate((m == c).astype(np.uint8), self.kernel, iterations=1)
            out[d > 0] = c
        return img, torch.as_tensor(out, dtype=torch.long)


def train(epochs=30, train_hi=3000, batch=8):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = VideoParquetDataset(VIDEO, GT, frame_indices=list(range(0, train_hi)),
                               resize_to=RESIZE, use_wid_only=True, wid_to_class=WID2CLS)
    ds = _Dilated(base, k=3)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = UNet(3, 7).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # heavy weight on whisker classes (background dominates)
    w = torch.tensor([0.02] + [1.0] * 6, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    print(f"[trackB] device={dev} train_frames={len(ds)} epochs={epochs}")
    for ep in range(epochs):
        model.train(); tot = 0.0
        for img, mask in dl:
            img, mask = img.to(dev), mask.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                out = model(img)
                loss = crit(out, mask)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[trackB] epoch {ep}: loss={tot/len(dl):.4f}", flush=True)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    torch.save(model.state_dict(), MODEL)
    print(f"[trackB] saved model to {MODEL}")
    return MODEL


@torch.no_grad()
def relabel_window(model_path, combined_parquet, lo, hi):
    """Predict masks for frames [lo,hi); relabel combined detections by majority-vote mask
    class along each whisker's skeleton. Returns a parquet-shaped df with vision `wid`."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(3, 7).to(dev); model.load_state_dict(torch.load(model_path, map_location=dev)); model.eval()
    comb = pd.read_parquet(combined_parquet)
    comb = comb[(comb.fid >= lo) & (comb.fid < hi)].copy()
    cap = cv2.VideoCapture(VIDEO)
    H0, W0 = int(cap.get(4)), int(cap.get(3))
    sy, sx = RESIZE[0] / H0, RESIZE[1] / W0
    rows = []
    for fid, g in comb.groupby("fid"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid)); ret, fr = cap.read()
        if not ret:
            continue
        x = torch.as_tensor(cv2.cvtColor(cv2.resize(fr, (RESIZE[1], RESIZE[0])), cv2.COLOR_BGR2RGB)
                            ).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        pred = model(x).argmax(1)[0].cpu().numpy()    # [256,256] class map
        for _, row in g.iterrows():
            xs = (np.asarray(row.pixels_x) * sx).clip(0, RESIZE[1] - 1).astype(int)
            ys = (np.asarray(row.pixels_y) * sy).clip(0, RESIZE[0] - 1).astype(int)
            votes = pred[ys, xs]
            votes = votes[votes > 0]
            if len(votes) == 0:
                continue
            cls = int(np.bincount(votes).argmax())
            r = row.copy(); r["wid"] = CLS2WID.get(cls, -1)
            rows.append(r)
    cap.release()
    return pd.DataFrame([r for r in rows if r["wid"] >= 0])


if __name__ == "__main__":
    import argparse, warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--eval", action="store_true")
    a = p.parse_args()
    if a.eval:
        from . import eval_linking as ev
        df = relabel_window(MODEL, f"{_E}/whisker_active/sc013_active.parquet", 3000, 4000)
        gt = pd.read_parquet(GT); gt = gt[(gt.fid >= 3000) & (gt.fid < 4000)]
        o = ev.compute_metrics(df, gt)["overall"]
        print(f"[trackB EVAL] vision identity: ida={o['identity_accuracy']:.4f} "
              f"idsw={o['total_id_switches']} miss={o['miss']} fp={o['false_positive']} "
              f"idf1={o['idf1']:.4f}  (n_pred={len(df)})")
    else:
        train(epochs=a.epochs)
