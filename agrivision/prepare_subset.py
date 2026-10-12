"""Construit un sous-ensemble équilibré d'Agriculture-Vision 2021, prêt à mettre sur Drive.

Entrée : le dossier extrait de Agriculture-Vision-2021.tar.gz (train/ et val/).
Sortie :
    out/{train,val,test}/rgb/<id>.jpg   (copie brute du JPG d'origine)
    out/{train,val,test}/lbl/<id>.png   (uint16, bits décrits dans agrivision.LABEL_BITS)
    out/meta.json                       (classes, tailles, fréquences de pixels)

Le "test" est tiré du val officiel (les labels du test officiel ne sont pas publics),
avec des parcelles (field ids) disjointes de celles du val qu'on garde pour la sélection de modèle.

    python -m agrivision.prepare_subset --src /content/raw/Agriculture-Vision-2021 --out /content/agv_subset
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from agrivision import CLASSES, LABEL_BITS


def read_bin(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img > 0


def encode_tile(split_dir, tile_id):
    """Retourne le label uint16 encodé d'une tuile."""
    valid = read_bin(split_dir / "masks" / f"{tile_id}.png")
    boundary = split_dir / "boundaries" / f"{tile_id}.png"
    if boundary.exists():
        valid &= read_bin(boundary)
    lbl = valid.astype(np.uint16)
    for name, bit in LABEL_BITS.items():
        path = split_dir / "labels" / name / f"{tile_id}.png"
        if path.exists():
            lbl |= ((read_bin(path) & valid).astype(np.uint16) << bit)
    return lbl


def tile_stats(args):
    """Nombre de pixels par bit (0 = valide) pour une tuile."""
    split_dir, tile_id = args
    lbl = encode_tile(split_dir, tile_id)
    return tile_id, [int(((lbl >> b) & 1).sum()) for b in range(max(LABEL_BITS.values()) + 1)]


def scan_split(split_dir, workers):
    ids = sorted(p.stem for p in (split_dir / "images" / "rgb").glob("*.jpg"))
    found = sorted(p.name for p in (split_dir / "labels").iterdir() if p.is_dir())
    print(f"[{split_dir.name}] {len(ids)} tuiles, dossiers de labels : {found}")
    missing = set(LABEL_BITS) - set(found)
    if missing:
        print(f"  attention, labels absents : {sorted(missing)}")
    with Pool(workers) as pool:
        stats = dict(tqdm(pool.imap_unordered(tile_stats, [(split_dir, i) for i in ids], chunksize=64),
                          total=len(ids), desc=f"scan {split_dir.name}"))
    return stats


def balanced_pick(stats, n, rng, min_pixels):
    """Prend d'abord des tuiles contenant les classes rares, puis complète au hasard."""
    n_cls = len(CLASSES)
    ids = [i for i, s in stats.items() if s[0] > 0]  # au moins un pixel valide
    if n >= len(ids):
        return sorted(ids)
    has = {c: [i for i in ids if stats[i][c] >= min_pixels] for c in range(1, n_cls)}
    quota = n // n_cls  # environ 1/9 du budget par classe, le reste en tirage libre
    picked, counts = set(), defaultdict(int)
    for c in sorted(has, key=lambda c: len(has[c])):  # de la plus rare à la plus fréquente
        cand = has[c][:]
        rng.shuffle(cand)
        for i in cand:
            if counts[c] >= quota or len(picked) >= n:
                break
            if i in picked:
                continue
            picked.add(i)
            for k in range(1, n_cls):
                if stats[i][k] >= min_pixels:
                    counts[k] += 1
    rest = [i for i in ids if i not in picked]
    rng.shuffle(rest)
    picked.update(rest[: n - len(picked)])
    return sorted(picked)


def split_by_field(ids, frac_b, rng):
    """Sépare des ids en deux groupes de parcelles disjointes (~frac_b des tuiles dans b)."""
    by_field = defaultdict(list)
    for i in ids:
        by_field[i.split("_")[0]].append(i)
    fields = sorted(by_field)
    rng.shuffle(fields)
    a, b = [], []
    for f in fields:
        (b if len(b) < frac_b * len(ids) else a).extend(by_field[f])
    return a, b


def write_tile(args):
    split_dir, out_dir, tile_id, with_nir = args
    shutil.copyfile(split_dir / "images" / "rgb" / f"{tile_id}.jpg", out_dir / "rgb" / f"{tile_id}.jpg")
    if with_nir:
        shutil.copyfile(split_dir / "images" / "nir" / f"{tile_id}.jpg", out_dir / "nir" / f"{tile_id}.jpg")
    cv2.imwrite(str(out_dir / "lbl" / f"{tile_id}.png"), encode_tile(split_dir, tile_id))


def export(split_dir, out_dir, ids, stats, with_nir, workers):
    for sub in ["rgb", "lbl"] + (["nir"] if with_nir else []):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        jobs = [(split_dir, out_dir, i, with_nir) for i in ids]
        list(tqdm(pool.imap_unordered(write_tile, jobs, chunksize=32), total=len(ids), desc=f"write {out_dir.name}"))
    pix = np.array([stats[i] for i in ids]).sum(0)
    tiles = np.array([[s >= 1 for s in stats[i]] for i in ids]).sum(0)
    return {
        "n_tiles": len(ids),
        "valid_pixels": int(pix[0]),
        "pixels": {n: int(pix[b]) for n, b in LABEL_BITS.items()},
        "tiles_with": {n: int(tiles[b]) for n, b in LABEL_BITS.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dossier Agriculture-Vision-2021 extrait")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--min-pixels", type=int, default=500, help="seuil pour compter une classe comme présente")
    ap.add_argument("--with-nir", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = {"classes": CLASSES, "label_bits": LABEL_BITS, "seed": args.seed, "splits": {}}

    train_stats = scan_split(args.src / "train", args.workers)
    train_ids = balanced_pick(train_stats, args.n_train, rng, args.min_pixels)
    meta["splits"]["train"] = export(args.src / "train", args.out / "train", train_ids, train_stats,
                                     args.with_nir, args.workers)

    val_stats = scan_split(args.src / "val", args.workers)
    val_pool, test_pool = split_by_field(list(val_stats), args.n_test / (args.n_val + args.n_test), rng)
    for name, pool_ids, n in [("val", val_pool, args.n_val), ("test", test_pool, args.n_test)]:
        ids = balanced_pick({i: val_stats[i] for i in pool_ids}, n, rng, args.min_pixels)
        meta["splits"][name] = export(args.src / "val", args.out / name, ids, val_stats,
                                      args.with_nir, args.workers)

    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["splits"], indent=2))


if __name__ == "__main__":
    main()
