from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from agrivision import MEAN, STD

cv2.setNumThreads(0)  # évite la sur-souscription avec les workers du DataLoader


class AgriVisionSubset(Dataset):
    """Lit le sous-ensemble produit par prepare_subset.

    Renvoie (image float32 normalisée [3,H,W], label int32 [H,W] encodé en bits).
    Le décodage bits -> cibles multi-label se fait sur GPU (voir decode_labels).
    """

    def __init__(self, root, split, augment=False):
        self.dir = Path(root) / split
        self.ids = sorted(p.stem for p in (self.dir / "rgb").glob("*.jpg"))
        if not self.ids:
            raise FileNotFoundError(f"aucune image dans {self.dir / 'rgb'}")
        self.augment = augment
        self.mean = np.array(MEAN, np.float32) * 255
        self.std = np.array(STD, np.float32) * 255

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tile_id = self.ids[idx]
        img = cv2.cvtColor(cv2.imread(str(self.dir / "rgb" / f"{tile_id}.jpg")), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(self.dir / "lbl" / f"{tile_id}.png"), cv2.IMREAD_UNCHANGED)

        img = img.astype(np.float32)
        if self.augment:
            # Vue aérienne : invariance aux rotations de 90° et aux symétries.
            k = np.random.randint(4)
            img, lbl = np.rot90(img, k), np.rot90(lbl, k)
            if np.random.rand() < 0.5:
                img, lbl = img[:, ::-1], lbl[:, ::-1]
            # Légère variation photométrique (éclairage, capteur).
            img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
            img = np.clip(img, 0, 255)

        img = (img - self.mean) / self.std
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        lbl = torch.from_numpy(np.ascontiguousarray(lbl).astype(np.int32))
        return img, lbl


def decode_labels(lbl, num_classes):
    """bits [B,H,W] -> (target multi-hot bool [B,C,H,W], valid bool [B,H,W]).

    Le canal 0 (background) vaut 1 sur les pixels valides sans aucune annotation.
    Un pixel peut avoir plusieurs classes (annotations qui se chevauchent).
    """
    valid = (lbl & 1).bool()
    fg = torch.stack([((lbl >> k) & 1).bool() for k in range(1, num_classes)], 1) & valid[:, None]
    bg = valid & ~fg.any(1)
    return torch.cat([bg[:, None], fg], 1), valid
