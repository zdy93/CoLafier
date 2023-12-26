import torch.nn.functional as F


def loss_lid(logits1, logits2, logits2_new, label, weight, criterion=None):
    if criterion is None:
        loss_1 = F.cross_entropy(logits1, label, reduction='none')
        loss_2 = F.cross_entropy(logits2, label, reduction='none')
        loss_2_new = F.cross_entropy(logits2_new, label, reduction='none')
    else:
        loss_1 = criterion(logits1, label)
        loss_2 = criterion(logits2, label)
        loss_2_new = criterion(logits2_new, label)
    loss_1 = (weight * loss_1).mean()
    loss_2 = (weight * loss_2).mean()
    loss_2_new = (weight * loss_2_new).mean()
    # loss_1 = loss_1.mean()
    # loss_2 = loss_2.mean()
    # loss_2_new = loss_2_new.mean()
    return loss_1, loss_2, loss_2_new


def loss_lid_cos(logits1, logits2, logits2_new, label, weight):
    loss_1 = F.cross_entropy(logits1, label, reduction='none')
    loss_2 = F.cross_entropy(logits2, label, reduction='none')
    loss_2_new = F.cross_entropy(logits2_new, label, reduction='none')
    loss_2_cos = 1.0 - F.cosine_similarity(logits2, logits2_new)
    loss_1 = (weight * loss_1).mean()
    loss_2 = (weight * loss_2).mean()
    loss_2_new = (weight * loss_2_new).mean()
    loss_2_cos = 10.0 * ((1.0 - weight) * loss_2_cos).mean()
    # loss_1 = loss_1.mean()
    # loss_2 = loss_2.mean()
    # loss_2_new = loss_2_new.mean()
    # loss_2_cos = loss_2_cos.mean()
    return loss_1, loss_2, loss_2_cos + loss_2_new


def loss_consistency(logits, logits_new, weight):
    w_logits, s_logits = logits[:logits.shape[0] // 2], logits[logits.shape[0] // 2:]
    w_logits_new, s_logits_new = logits_new[:logits_new.shape[0] // 2], logits_new[logits_new.shape[0] // 2:]
    w_loss_cos = 1.0 - F.cosine_similarity(w_logits, w_logits_new)
    s_loss_cos = 1.0 - F.cosine_similarity(s_logits, s_logits_new)
    loss_cos = (weight * w_loss_cos).mean() + (weight * s_loss_cos).mean()
    return loss_cos

def loss_consistency_ws(logits, weight):
    w_logits, s_logits = logits[:logits.shape[0] // 2], logits[logits.shape[0] // 2:]
    loss_cos = 1.0 - F.cosine_similarity(w_logits, s_logits)
    loss_cos = (weight * loss_cos).mean()
    return loss_cos


def loss_lid_general(logits1, logits2, logits2_new, label, weight, criterion):
    w_logits_1, s_logits_1 = logits1[:logits1.shape[0] // 2], logits1[logits1.shape[0] // 2:]
    w_logits_2, s_logits_2 = logits2[:logits2.shape[0] // 2], logits2[logits2.shape[0] // 2:]
    w_logits_2_new, s_logits_2_new = logits2_new[:logits2_new.shape[0] // 2], logits2_new[logits2_new.shape[0] // 2:]
    w_loss_1 = criterion(w_logits_1, label)
    s_loss_1 = criterion(s_logits_1, label)
    w_loss_2 = criterion(w_logits_2, label)
    s_loss_2 = criterion(s_logits_2, label)
    w_loss_2_new = criterion(w_logits_2_new, label)
    s_loss_2_new = criterion(s_logits_2_new, label)
    loss_1 = (weight * w_loss_1).mean() + (weight * s_loss_1).mean()
    loss_2 = (weight * w_loss_2).mean() + (weight * s_loss_2).mean()
    loss_2_new = (weight * w_loss_2_new).mean() + (weight * s_loss_2_new).mean()
    return loss_1, loss_2, loss_2_new