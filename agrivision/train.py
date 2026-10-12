"""Entraînement de la tête de segmentation sur le sous-ensemble Agriculture-Vision.

    python -m agrivision.train --data /content/agv_subset --out /content/drive/MyDrive/agrivision/runs/vits16
    python -m agrivision.train --data /content/agv_subset --out <run> --eval-only --split test
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agrivision import CLASSES
from agrivision.data import AgriVisionSubset, decode_labels
from agrivision.losses import any_label_ce, soft_dice
from agrivision.metrics import AgriVisionIoU
from agrivision.model import DEFAULT_BACKBONE, DinoSegmenter


def autocast_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    metric = AgriVisionIoU(num_classes, device)
    for img, lbl in tqdm(loader, desc="eval", leave=False):
        img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        target, valid = decode_labels(lbl, num_classes)
        with torch.autocast("cuda", dtype=autocast_dtype(), enabled=device.type == "cuda"):
            logits = model(img)
        metric.update(logits.argmax(1), target, valid)
    return metric.compute()


def build_model(args, num_classes):
    return DinoSegmenter(num_classes, backbone=args.backbone, layers=args.layers, dim=args.dim,
                         backbone_res=args.backbone_res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="dossier du run (mettre sur Drive)")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--backbone-res", type=int, default=None, help="ex. 768 : patchs plus fins, ~2.5x plus lent")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="val", help="split évalué avec --eval-only")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    num_classes = len(CLASSES)
    args.out.mkdir(parents=True, exist_ok=True)

    model = build_model(args, num_classes).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"backbone {args.backbone} | paramètres entraînables : {n_train / 1e6:.2f} M")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0)

    if args.eval_only:
        ckpt = torch.load(args.out / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        loader = DataLoader(AgriVisionSubset(args.data, args.split), shuffle=False, **loader_kw)
        miou, ious = evaluate(model, loader, device, num_classes)
        report = {"split": args.split, "miou": miou, "iou": dict(zip(CLASSES, ious))}
        print(json.dumps(report, indent=2))
        (args.out / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2))
        return

    train_loader = DataLoader(AgriVisionSubset(args.data, "train", augment=True), shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(AgriVisionSubset(args.data, "val"), shuffle=False, **loader_kw)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warmup = int(args.warmup_epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))
    dtype = autocast_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    start_epoch, best = 0, -1.0
    last = args.out / "last.pt"
    if last.exists():  # reprise automatique après une coupure Colab
        ckpt = torch.load(last, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"reprise à l'epoch {start_epoch} (meilleur mIoU {best:.4f})")

    log_path = args.out / "log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, (img, lbl) in enumerate(pbar):
            img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            target, valid = decode_labels(lbl, num_classes)
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                logits = model(img)
            loss = any_label_ce(logits, target, valid) + args.dice_weight * soft_dice(logits, target, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{running / (step + 1):.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        miou, ious = evaluate(model, val_loader, device, num_classes)
        entry = {"epoch": epoch, "loss": running / len(train_loader), "val_miou": miou,
                 "val_iou": dict(zip(CLASSES, ious)), "time_s": time.time() - t0}
        print(f"epoch {epoch} | loss {entry['loss']:.4f} | val mIoU {miou:.4f} | "
              + " ".join(f"{c[:6]}={v:.2f}" for c, v in zip(CLASSES, ious)))
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        state = {"model": model.head_state_dict(), "args": {k: str(v) for k, v in vars(args).items()},
                 "epoch": epoch, "val_miou": miou}
        if miou > best:
            best = miou
            torch.save(state, args.out / "best.pt")
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best}, last)

    print(f"terminé, meilleur val mIoU : {best:.4f}")


if __name__ == "__main__":
    main()
