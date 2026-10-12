"""Entraînement de la segmentation Agriculture-Vision.

    # DINOv3 ViT-S, 4 derniers blocs ajustés, toutes les données
    python -m agrivision.train --data /content/agv_full --out <run> --unfreeze-last 4
    # Référence CNN
    python -m agrivision.train --data /content/agv_full --out <run> --arch smp --smp-encoder efficientnet-b3
    # Évaluation finale (l'architecture est relue dans le checkpoint), avec 8 rotations/symétries
    python -m agrivision.train --data /content/agv_full --out <run> --eval-only --split val --tta 8
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from agrivision import CLASSES
from agrivision.data import (AgriVisionSubset, class_pixel_freq, decode_labels, rare_class_weights,
                             tile_counts)
from agrivision.losses import any_label_ce, class_weights_from_freq, soft_dice
from agrivision.metrics import AgriVisionIoU
from agrivision.model import DEFAULT_BACKBONE, DinoSegmenter, SmpSegmenter

# Arguments qui définissent l'architecture : sauvegardés dans le checkpoint et relus en --eval-only.
MODEL_ARGS = ["arch", "backbone", "layers", "dim", "backbone_res", "unfreeze_last", "smp_arch", "smp_encoder"]


def autocast_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def predict(model, img, tta):
    """Logits, ou moyenne des probabilités sur `tta` versions tournées/retournées de l'image."""
    if tta <= 1:
        return model(img)
    probs = 0
    for k in range(4 if tta >= 4 else 2):
        for flip in ([False, True] if tta >= 8 else [False]):
            x = torch.flip(img, (-1,)) if flip else img
            p = model(torch.rot90(x, k, (-2, -1))).float().softmax(1)
            p = torch.rot90(p, -k, (-2, -1))
            probs = probs + (torch.flip(p, (-1,)) if flip else p)
    return probs


@torch.no_grad()
def evaluate(model, loader, device, num_classes, tta=1):
    model.eval()
    metric = AgriVisionIoU(num_classes, device)
    for img, lbl in tqdm(loader, desc="eval", leave=False):
        img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
        target, valid = decode_labels(lbl, num_classes)
        with torch.autocast("cuda", dtype=autocast_dtype(), enabled=device.type == "cuda"):
            out = predict(model, img, tta)
        metric.update(out.argmax(1), target, valid)
    return metric.compute()


def build_model(args, num_classes):
    if args.arch == "smp":
        return SmpSegmenter(num_classes, arch=args.smp_arch, encoder=args.smp_encoder)
    return DinoSegmenter(num_classes, backbone=args.backbone, layers=args.layers, dim=args.dim,
                         backbone_res=args.backbone_res, unfreeze_last=args.unfreeze_last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="dossier du run (mettre sur Drive)")
    # modèle
    ap.add_argument("--arch", choices=["dinov3", "smp"], default="dinov3")
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--backbone-res", type=int, default=None, help="ex. 768 : patchs plus fins, ~2.5x plus lent")
    ap.add_argument("--unfreeze-last", type=int, default=0, help="nb de derniers blocs du ViT à entraîner")
    ap.add_argument("--smp-arch", default="DeepLabV3Plus")
    ap.add_argument("--smp-encoder", default="efficientnet-b3")
    # optimisation
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="tête / décodeur")
    ap.add_argument("--backbone-lr", type=float, default=3e-5, help="blocs du ViT dégelés / encodeur CNN")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.5)
    ap.add_argument("--dice-weight", type=float, default=0.5)
    ap.add_argument("--class-weight-power", type=float, default=0.5, help="0 = pas de pondération des classes")
    # données
    ap.add_argument("--rcs-mix", type=float, default=0.5, help="part des tirages centrés sur une classe rare (0 = off)")
    ap.add_argument("--mosaic-p", type=float, default=0.25)
    ap.add_argument("--max-zoom", type=float, default=1.5)
    ap.add_argument("--val-max", type=int, default=4000, help="tuiles de val évaluées à chaque epoch (0 = toutes)")
    ap.add_argument("--workers", type=int, default=8)
    # évaluation seule
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="val", help="split évalué avec --eval-only")
    ap.add_argument("--tta", type=int, default=1, choices=[1, 2, 4, 8])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    num_classes = len(CLASSES)
    args.out.mkdir(parents=True, exist_ok=True)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                     persistent_workers=args.workers > 0)

    if args.eval_only:
        if not (args.out / "best.pt").exists():
            raise SystemExit(f"pas de best.pt dans {args.out} : l'entraînement n'a pas terminé au moins une epoch")
        ckpt = torch.load(args.out / "best.pt", map_location="cpu")
        for k, v in ckpt.get("model_args", {}).items():
            setattr(args, k, v)
        model = build_model(args, num_classes).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        loader = DataLoader(AgriVisionSubset(args.data, args.split), shuffle=False, **loader_kw)
        miou, ious = evaluate(model, loader, device, num_classes, args.tta)
        report = {"split": args.split, "n_tiles": len(loader.dataset), "tta": args.tta, "epoch": ckpt["epoch"],
                  "miou": miou, "iou": dict(zip(CLASSES, ious))}
        print(json.dumps(report, indent=2))
        (args.out / f"eval_{args.split}_tta{args.tta}.json").write_text(json.dumps(report, indent=2))
        return

    model = build_model(args, num_classes).to(device)
    bb = model.backbone_parameters()
    bb_ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in bb_ids]
    print(f"{args.arch} | entraînés : tête {sum(p.numel() for p in head) / 1e6:.2f} M, "
          f"backbone {sum(p.numel() for p in bb) / 1e6:.2f} M")

    train_ds = AgriVisionSubset(args.data, "train", augment=True, max_zoom=args.max_zoom, mosaic_p=args.mosaic_p)
    counts = tile_counts(train_ds, args.workers)
    freq = class_pixel_freq(counts, num_classes)
    class_w = class_weights_from_freq(freq, args.class_weight_power) if args.class_weight_power > 0 else None
    print("fréquence pixels :", " ".join(f"{c[:6]}={f:.4f}" for c, f in zip(CLASSES, freq)))
    if class_w is not None:
        print("poids classes    :", " ".join(f"{c[:6]}={w:.2f}" for c, w in zip(CLASSES, class_w.tolist())))
    if args.rcs_mix > 0:
        p_tile = rare_class_weights(counts, num_classes, mix=args.rcs_mix)
        train_ds.partner_p = p_tile
        sampler = WeightedRandomSampler(torch.as_tensor(p_tile, dtype=torch.double), len(train_ds))
        train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)

    val_ds = AgriVisionSubset(args.data, "val")
    if args.val_max and len(val_ds) > args.val_max:  # sous-ensemble fixe, pour aller vite à chaque epoch
        val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, args.val_max).astype(int).tolist())
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"train {len(train_ds)} tuiles | val par epoch {len(val_ds)} tuiles")

    groups = [{"params": head, "lr": args.lr}]
    if bb:
        groups.append({"params": bb, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warmup = int(args.warmup_epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))
    dtype = autocast_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    params = head + bb

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

    model_args = {k: getattr(args, k) for k in MODEL_ARGS}
    log_path = args.out / "log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running, t_data = time.time(), 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        t_wait = time.time()
        for step, (img, lbl) in enumerate(pbar):
            t_data += time.time() - t_wait
            img, lbl = img.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
            target, valid = decode_labels(lbl, num_classes)
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                logits = model(img)
            loss = (any_label_ce(logits, target, valid, class_w)
                    + args.dice_weight * soft_dice(logits, target, valid))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{running / (step + 1):.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
            t_wait = time.time()

        t_train = time.time() - t0
        miou, ious = evaluate(model, val_loader, device, num_classes)
        entry = {"epoch": epoch, "loss": running / len(train_loader), "val_miou": miou,
                 "val_iou": dict(zip(CLASSES, ious)), "time_s": time.time() - t0,
                 "train_s": t_train, "data_wait_frac": t_data / t_train}
        print(f"epoch {epoch} | loss {entry['loss']:.4f} | val mIoU {miou:.4f} | "
              f"{t_train / 60:.1f} min (attente données {100 * t_data / t_train:.0f} %) | "
              + " ".join(f"{c[:6]}={v:.2f}" for c, v in zip(CLASSES, ious)))
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        state = {"model": model.head_state_dict(), "model_args": model_args,
                 "args": {k: str(v) for k, v in vars(args).items()}, "epoch": epoch, "val_miou": miou}
        if miou > best:
            best = miou
            torch.save(state, args.out / "best.pt")
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best}, last)

    print(f"terminé, meilleur val mIoU (sous-ensemble par epoch) : {best:.4f}")


if __name__ == "__main__":
    main()
