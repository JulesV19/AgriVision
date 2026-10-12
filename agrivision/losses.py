import torch


def any_label_ce(logits, target, valid):
    """Cross-entropy "au moins une des classes vraies".

    Pour un pixel annoté avec l'ensemble S (souvent 1 classe, parfois plusieurs qui se chevauchent) :
        loss = -log(sum_{k in S} p_k)
    C'est cohérent avec la métrique officielle, où prédire n'importe quelle classe de S est correct.
    """
    logits = logits.float()
    log_z = torch.logsumexp(logits, 1)
    log_s = torch.logsumexp(logits.masked_fill(~target, float("-inf")), 1)
    loss = (log_z - log_s)[valid]
    return loss.mean() if loss.numel() else logits.sum() * 0


def soft_dice(logits, target, valid, eps=1.0):
    """Dice moyen sur les classes présentes dans le batch, pour mieux traiter les classes rares."""
    probs = logits.float().softmax(1) * valid[:, None]
    tgt = target.float() * valid[:, None]
    inter = (probs * tgt).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + tgt.sum((0, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    present = tgt.sum((0, 2, 3)) > 0
    return dice[present].mean() if present.any() else dice.sum() * 0
