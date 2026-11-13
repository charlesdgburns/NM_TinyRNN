"""Prediction metrics shared by scientific studies."""
import math
import torch
from torch.utils.data import DataLoader, Subset


def free_choice_nll(model, dataset, indices, batch_size):
    """Trial-weighted held-out cross-entropy, excluding forced-choice targets."""
    total = 0.
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y, mask in DataLoader(Subset(dataset, indices), batch_size=batch_size):
            logits, _ = model(x)
            free = mask == 0
            n = int(free.sum())
            if n:
                total += float(torch.nn.functional.cross_entropy(logits[free], y[free], reduction='sum'))
                count += n
    if not count or not math.isfinite(total):
        raise ValueError('No free-choice targets or nonfinite NLL')
    return {'held_out_nll':total/count, 'nll_sum':total, 'n_free':count}
