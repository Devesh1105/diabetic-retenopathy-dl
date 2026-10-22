"""Shared PyTorch pipeline for the CNN and ViT runs.

Same data split, class order, class weights, focal loss and metrics as the
Attention U-Net notebook, so the three models are compared on equal terms.
Images are preloaded once as uint8 (~2.6 GB RAM for all splits) and augmented
per-sample on the GPU, so no DataLoader workers are needed (Windows-safe).
"""
import json, math, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]  # same order as Keras flow_from_directory
EXTS = (".png", ".jpg", ".jpeg")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def seed_all(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _load(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_split(root, split, size=224, threads=12):
    """Return (uint8 tensor N,3,H,W, int64 labels) for one split folder."""
    paths, labels = [], []
    for ci, c in enumerate(CLASS_NAMES):
        d = os.path.join(root, split, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(d, f)); labels.append(ci)
    with ThreadPoolExecutor(threads) as ex:
        imgs = list(ex.map(lambda p: _load(p, size), paths))
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    print(f"  {split:<10}: {len(y):>6} images  {np.bincount(labels, minlength=5).tolist()}")
    return x, y


def gpu_augment(x, rot=30, shift=0.1, zoom=0.15, bc=0.2):
    """Per-sample random rotation/shift/zoom/flips + brightness/contrast. x: float 0..1 on GPU."""
    n, dev = x.shape[0], x.device
    a = (torch.rand(n, device=dev) * 2 - 1) * math.radians(rot)
    s = 1 + (torch.rand(n, device=dev) * 2 - 1) * zoom
    fx = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    t = (torch.rand(n, 2, device=dev) * 2 - 1) * shift * 2
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos * fx, -sin, t[:, 0]], 1),
                         torch.stack([sin, cos * fy, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * bc
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * bc
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c + m * b).clamp(0, 1)


def to_input(xb, device, augment=False):
    x = xb.to(device, non_blocking=True).float().div_(255)
    if augment:
        x = gpu_augment(x)
    return ((x - MEAN.to(device)) / STD.to(device)).contiguous(memory_format=torch.channels_last)


class FocalLoss(nn.Module):
    """Same formula as the U-Net notebook: sum_c alpha*(1-p)^gamma * -y*log p, scaled by class weight."""
    def __init__(self, class_weights, alpha=0.25, gamma=2.0):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None]).squeeze(1)
        p = logp.exp()
        return (self.alpha * (1 - p) ** self.gamma * -logp * self.w[y]).mean()


@torch.no_grad()
def predict(model, x, device, bs=128):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(F.softmax(model(to_input(x[i:i + bs], device)).float(), 1).cpu())
    return torch.cat(out).numpy()


def metrics(y, prob):
    pred = prob.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "quadratic_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_auc": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
    }


def class_weights(y):
    return compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=y.numpy())


def train(model, data, cfg, device, ckpt):
    """Adam + ReduceLROnPlateau + early stopping on val loss (same policy as the U-Net run)."""
    (xtr, ytr), (xva, yva) = data["train"], data["validation"]
    cw = class_weights(ytr)
    crit = FocalLoss(cw, cfg["focal_alpha"], cfg["focal_gamma"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg["reduce_lr_patience"], min_lr=1e-7)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hist = {k: [] for k in ["loss", "val_loss", "accuracy", "val_accuracy", "val_balanced_accuracy", "lr"]}
    best, bad, t0 = float("inf"), 0, time.time()
    bs = cfg["batch_size"]
    for ep in range(cfg["epochs"]):
        model.train(); perm = torch.randperm(len(ytr)); tl = tc = 0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x, y = to_input(xtr[idx], device, augment=True), ytr[idx].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item()
        n = (len(perm) // bs) * bs
        pv = predict(model, xva, device)
        vl = crit(torch.log(torch.from_numpy(pv).clamp_min(1e-7)).to(device), yva.to(device)).item()
        vm = metrics(yva.numpy(), pv)
        sched.step(vl)
        for k, v in zip(hist, [tl / n, vl, tc / n, vm["accuracy"], vm["balanced_accuracy"], opt.param_groups[0]["lr"]]):
            hist[k].append(v)
        flag = ""
        if vl < best:
            best, bad, flag = vl, 0, "  * saved"
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"Epoch {ep + 1:>2}/{cfg['epochs']}  loss {tl / n:.4f}  acc {tc / n:.4f} | val_loss {vl:.4f}  "
              f"val_acc {vm['accuracy']:.4f}  val_bal_acc {vm['balanced_accuracy']:.4f}  lr {hist['lr'][-1]:.1e}  "
              f"[{time.time() - t0:.0f}s]{flag}", flush=True)
        if bad >= cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {ep + 1}"); break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return hist, time.time() - t0, cw


def inference_ms(model, device, n=256, bs=64):
    x = torch.randint(0, 255, (bs, 3, 224, 224), dtype=torch.uint8)
    predict(model, x, device)  # warm-up
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(n // bs): predict(model, x, device, bs)
    if device.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def save_results(path, name, y, prob, extra):
    res = {"model": name, "class_names": CLASS_NAMES, "test": metrics(y, prob),
           "y_true": y.tolist(), "y_prob": np.round(prob, 6).tolist(), **extra}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f)
    print(classification_report(y, prob.argmax(1), target_names=CLASS_NAMES, digits=4))
    print({k: round(v, 4) for k, v in res["test"].items()})
    print("saved ->", path)
    return res


if __name__ == "__main__":  # smoke check of the aug + loss shapes
    x = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
    xi = to_input(x, torch.device("cpu"), augment=True)
    assert xi.shape == (4, 3, 224, 224)
    l = FocalLoss(np.ones(5))(torch.randn(4, 5), torch.tensor([0, 1, 2, 4]))
    assert l.item() > 0
    print("ok")
