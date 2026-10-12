import torch


class AgriVisionIoU:
    """mIoU officiel d'Agriculture-Vision avec gestion des labels qui se chevauchent.

    - Si la classe prédite fait partie des classes vraies du pixel, c'est un vrai positif pour elle.
    - Sinon, le pixel est compté comme erreur (vraie s -> prédite p) pour chacune de ses classes vraies s.
    Les pixels invalides (hors masque ou hors parcelle) sont ignorés.
    """

    def __init__(self, num_classes, device="cpu"):
        self.n = num_classes
        self.conf = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, pred, target, valid):
        # pred [B,H,W] long, target [B,C,H,W] bool, valid [B,H,W] bool
        n = self.n
        hit = target.gather(1, pred[:, None]).squeeze(1) & valid
        self.conf += torch.diag(torch.bincount(pred[hit], minlength=n).double())
        miss = valid & ~hit
        p = pred[miss]
        t = target.permute(0, 2, 3, 1)[miss]  # [N, C]
        rows, cols = t.nonzero(as_tuple=True)
        idx = cols * n + p[rows]
        self.conf += torch.bincount(idx, minlength=n * n).view(n, n).double()

    def compute(self):
        tp = self.conf.diag()
        iou = tp / (self.conf.sum(0) + self.conf.sum(1) - tp).clamp(min=1)
        return iou.mean().item(), iou.tolist()
