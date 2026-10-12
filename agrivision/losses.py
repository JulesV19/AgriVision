import numpy as np
import torch


def class_weights_from_freq(freq, power=0.5, clip=(0.1, 10.0)):
    """Poids par classe (fréquence médiane / fréquence)^power : les classes rares comptent plus.

    power=0 -> tous les poids à 1 (désactivé), power=1 -> rééquilibrage complet.
    """
    w = (np.median(freq) / np.maximum(freq, 1e-8)) ** power
    return torch.tensor(np.clip(w, *clip), dtype=torch.float32)


def any_label_ce(logits, target, valid, class_weights=None):
    """Cross-entropy "au moins une des classes vraies".

    Pour un pixel annoté avec l'ensemble S (souvent 1 classe, parfois plusieurs qui se chevauchent) :
        loss = -log(sum_{k in S} p_k)
    C'est cohérent avec la métrique officielle, où prédire n'importe quelle classe de S est correct.
    Avec class_weights, chaque pixel est pondéré par le poids max de ses classes vraies.
    """
    logits = logits.float()
    log_z = torch.logsumexp(logits, 1)
    log_s = torch.logsumexp(logits.masked_fill(~target, float("-inf")), 1)
    loss = log_z - log_s
    if class_weights is None:
        loss = loss[valid]
        return loss.mean() if loss.numel() else logits.sum() * 0
    w = (target.float() * class_weights.to(logits.device)[None, :, None, None]).amax(1) * valid
    loss = torch.where(valid, loss, 0.0)  # pixels invalides : loss = inf, et inf * 0 donnerait NaN
    return (loss * w).sum() / w.sum().clamp(min=1e-6)


def soft_dice(logits, target, valid, eps=1.0):
    """Dice moyen sur les classes présentes dans le batch, pour mieux traiter les classes rares."""
    probs = logits.float().softmax(1) * valid[:, None]
    tgt = target.float() * valid[:, None]
    inter = (probs * tgt).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + tgt.sum((0, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    present = tgt.sum((0, 2, 3)) > 0
    return dice[present].mean() if present.any() else dice.sum() * 0
