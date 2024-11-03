import torch as th
from torch.nn.functional import relu


def gci0_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    neg=False,
):
    if neg:
        return gci1_bot_loss(
            data,
            class_embed,
            class_rad,
            class_reg,
            margin,
            neg=False,
        )

    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        loss = relu(dist - margin)

        if class_reg is None:
            return loss
        else:
            return loss + class_reg(c) + class_reg(d)


def gci0_bot_loss(data, class_rad, epsilon, neg=False):
    if neg:
        return gci0_bot_loss_neg(data, class_rad, epsilon)
    else:
        rc = th.abs(class_rad(data[:, 0]))
        return rc
    

def gci0_bot_loss_neg(data, class_rad, epsilon):
    rc = th.abs(class_rad(data[:, 0]))
    loss = relu(epsilon - rc)
    return loss


def gci1_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    neg=False,
):
    if neg:
        return gci1_loss_neg(
            data,
            class_embed,
            class_rad,
            class_reg,
            margin,
        )

    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        e = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))

        sr = rc + rd

        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = (
            relu(dst - sr - margin)
            + relu(dst2 - rc - margin)
            + relu(dst3 - rd - margin)
        )

        if class_reg is None:
            return loss
        else:
            return loss + class_reg(c) + class_reg(d) + class_reg(e)


def gci1_loss_neg(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd

    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
    dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
    loss = (
        relu(dst - sr - margin)
        + relu(-dst2 + rc + margin)
        + relu(-dst3 + rd + margin)
    )

    if class_reg is None:
        return loss
    else:
        return loss + class_reg(c) + class_reg(d) + class_reg(e)


def gci1_bot_loss(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
    neg=False,
):
    if neg:
        return gci1_bot_loss_neg(data, class_embed, class_rad, class_reg, margin)
    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 1]))

        sr = rc + rd
        dst = th.linalg.norm(d - c, dim=1, keepdim=True)

        if class_reg is None:
            return relu(sr - dst + margin)
        else:
            return relu(sr - dst + margin) + class_reg(c) + class_reg(d)
        

def gci1_bot_loss_neg(
    data,
    class_embed,
    class_rad,
    class_reg,
    margin,
):
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 1]))

    sr = rc + rd

    dst = th.linalg.norm(d - c, dim=1, keepdim=True)
    loss = relu(dst - sr - margin)

    if class_reg is None:
        return loss
    else:
        return loss + class_reg(c) + class_reg(d)

def gci2_loss(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    neg=False,
):
    if neg:
        return gci2_loss_neg(
            data,
            class_embed,
            class_rad,
            rel_embed,
            class_reg,
            margin,
        )

    else:
        c = class_embed(data[:, 0])
        rE = rel_embed(data[:, 1])
        d = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 0]))
        rd = th.abs(class_rad(data[:, 2]))

        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = relu(dst + rc - rd - margin)
        if class_reg is not None:
            return loss + class_reg(c) + class_reg(d)
        else:
            return loss


def gci2_loss_neg(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
):
    c = class_embed(data[:, 0])
    rE = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 0]))
    rd = th.abs(class_rad(data[:, 2]))

    dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
    loss = relu(rc + rd - dst + margin)
    if class_reg is None:
        return loss
    else:
        return loss + class_reg(c) + class_reg(d)

def gci3_loss(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
    neg=False,
):
    if neg:
        return gci3_loss_neg(
            data,
            class_embed,
            class_rad,
            rel_embed,
            class_reg,
            margin,
        )

    else:
        rE = rel_embed(data[:, 0])
        c = class_embed(data[:, 1])
        d = class_embed(data[:, 2])
        rc = th.abs(class_rad(data[:, 1]))
        rd = th.abs(class_rad(data[:, 2]))

        euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
        loss = relu(euc + rc - rd - margin)

        if class_reg is None:
            return loss
        else:
            return loss + class_reg(c) + class_reg(d)


def gci3_loss_neg(
    data,
    class_embed,
    class_rad,
    rel_embed,
    class_reg,
    margin,
):
    rE = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])
    rc = th.abs(class_rad(data[:, 1]))
    rd = th.abs(class_rad(data[:, 2]))

    euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
    loss = relu(-euc + rc + rd + margin)

    if class_reg is None:
        return loss
    else:
        return loss + class_reg(c) + class_reg(d)


def gci3_bot_loss(data, class_rad, epsilon, neg=False):
    if neg:
        return gci3_bot_loss_neg(data, class_rad, epsilon)
    else:
        rc = th.abs(class_rad(data[:, 1]))
        return rc
    
def gci3_bot_loss_neg(data, class_rad, epsilon):
    rc = th.abs(class_rad(data[:, 1]))
    loss = relu(epsilon - rc)
    return loss