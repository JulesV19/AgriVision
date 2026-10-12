from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from agrivision import LABEL_BITS, MEAN, STD

cv2.setNumThreads(0)  # évite la sur-souscription avec les workers du DataLoader


class AgriVisionSubset(Dataset):
    """Lit les données produites par prepare_subset.

    Renvoie (image float32 normalisée [3,H,W], label int32 [H,W] encodé en bits).
    Le décodage bits -> cibles multi-label se fait sur GPU (voir decode_labels).

    Augmentations d'entraînement (augment=True) :
      - rotations de 90° et symétries (vue aérienne, pas de "haut") + légère variation d'éclairage ;
      - zoom aléatoire x1 à x`max_zoom` puis recadrage 512 (motifs fins vus plus gros) ;
      - mosaïque avec probabilité `mosaic_p` : 2x2 ou 3x3 tuiles assemblées puis réduites à 512
        (vue plus large / plus basse résolution, comme la 2e place du challenge 2022).
    """

    def __init__(self, root, split, augment=False, max_zoom=1.5, mosaic_p=0.0):
        self.dir = Path(root) / split
        self.ids = sorted(p.stem for p in (self.dir / "rgb").glob("*.jpg"))
        if not self.ids:
            raise FileNotFoundError(f"aucune image dans {self.dir / 'rgb'}")
        self.augment = augment
        self.max_zoom = max_zoom
        self.mosaic_p = mosaic_p
        self.partner_p = None  # probas de tirage des tuiles d'une mosaïque (classes rares)
        self.mean = np.array(MEAN, np.float32) * 255
        self.std = np.array(STD, np.float32) * 255

    def __len__(self):
        return len(self.ids)

    def read(self, idx):
        tile_id = self.ids[idx]
        img = cv2.cvtColor(cv2.imread(str(self.dir / "rgb" / f"{tile_id}.jpg")), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(self.dir / "lbl" / f"{tile_id}.png"), cv2.IMREAD_UNCHANGED)
        return img, lbl

    def mosaic(self, idx):
        k = np.random.choice([2, 3])
        others = np.random.choice(len(self), k * k - 1, p=self.partner_p)
        tiles = [self.read(i) for i in [idx, *others]]
        np.random.shuffle(tiles)
        img = np.concatenate([np.concatenate([t[0] for t in tiles[r * k:(r + 1) * k]], 1) for r in range(k)], 0)
        lbl = np.concatenate([np.concatenate([t[1] for t in tiles[r * k:(r + 1) * k]], 1) for r in range(k)], 0)
        size = tiles[0][0].shape[0]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        lbl = cv2.resize(lbl, (size, size), interpolation=cv2.INTER_NEAREST)  # le plus proche garde les bits
        return img, lbl

    def zoom(self, img, lbl):
        size = img.shape[0]
        s = int(size * np.random.uniform(1.0, self.max_zoom))
        if s <= size:
            return img, lbl
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (s, s), interpolation=cv2.INTER_NEAREST)
        y, x = np.random.randint(0, s - size + 1, 2)
        return img[y:y + size, x:x + size], lbl[y:y + size, x:x + size]

    def __getitem__(self, idx):
        if self.augment and np.random.rand() < self.mosaic_p:
            img, lbl = self.mosaic(idx)
        else:
            img, lbl = self.read(idx)
            if self.augment and self.max_zoom > 1:
                img, lbl = self.zoom(img, lbl)

        img = img.astype(np.float32)
        if self.augment:
            k = np.random.randint(4)
            img, lbl = np.rot90(img, k), np.rot90(lbl, k)
            if np.random.rand() < 0.5:
                img, lbl = img[:, ::-1], lbl[:, ::-1]
            img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
            img = np.clip(img, 0, 255)

        img = (img - self.mean) / self.std
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        lbl = torch.from_numpy(np.ascontiguousarray(lbl).astype(np.int32))
        return img, lbl


def tile_counts(ds, workers=8):
    """Pixels par bit pour chaque tuile du dataset (lit tiles.npz, sinon le calcule et le met en cache)."""
    cache = ds.dir / "tiles.npz"
    if cache.exists():
        z = np.load(cache)
        by_id = dict(zip(z["ids"].tolist(), z["counts"]))
        if all(i in by_id for i in ds.ids):
            return np.stack([by_id[i] for i in ds.ids])
    from multiprocessing import Pool
    with Pool(workers) as pool:
        counts = list(tqdm(pool.imap(_count_file, [ds.dir / "lbl" / f"{i}.png" for i in ds.ids], chunksize=64),
                           total=len(ds), desc=f"stats {ds.dir.name}"))
    counts = np.array(counts, np.int64)
    np.savez_compressed(cache, ids=np.array(ds.ids), counts=counts)
    return counts


def _count_file(path):
    lbl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return [int(((lbl >> b) & 1).sum()) for b in range(max(LABEL_BITS.values()) + 1)]


def class_pixel_freq(counts, num_classes):
    """Fréquence de pixels par classe (0 = background = valide sans annotation)."""
    valid = counts[:, 0].sum()
    fg = counts[:, 1:num_classes].sum(0)
    bg = max(valid - fg.sum(), 1)  # approx. : les chevauchements sont rares
    return np.concatenate([[bg], fg]) / valid


def rare_class_weights(counts, num_classes, mix=0.5, min_pixels=500):
    """Probabilité de tirage par tuile, pour voir plus souvent les classes rares.

    Mélange de deux tirages :
      - (1 - mix) : tirage uniforme (garde la vraie proportion de tuiles sans anomalie) ;
      - mix : on choisit une des 8 anomalies au hasard (même chance pour chacune), puis une tuile qui la contient.
    """
    n = len(counts)
    p = np.full(n, (1 - mix) / n)
    present = counts[:, 1:num_classes] >= min_pixels
    n_cls = present.any(0).sum()
    for c in range(present.shape[1]):
        k = present[:, c].sum()
        if k:
            p[present[:, c]] += mix / n_cls / k
    return p / p.sum()


def decode_labels(lbl, num_classes):
    """bits [B,H,W] -> (target multi-hot bool [B,C,H,W], valid bool [B,H,W]).

    Le canal 0 (background) vaut 1 sur les pixels valides sans aucune annotation.
    Un pixel peut avoir plusieurs classes (annotations qui se chevauchent).
    """
    valid = (lbl & 1).bool()
    fg = torch.stack([((lbl >> k) & 1).bool() for k in range(1, num_classes)], 1) & valid[:, None]
    bg = valid & ~fg.any(1)
    return torch.cat([bg[:, None], fg], 1), valid
