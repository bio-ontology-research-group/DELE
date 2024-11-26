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
    """
    Compute GCI0 (`C \sqsubseteq D`) loss

    :param data: GCI0 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI0_BOT (`C \sqsubseteq \bot`) loss

    :param data: GCI0_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci0_bot_loss_neg(data, class_rad, epsilon)
    else:
        rc = th.abs(class_rad(data[:, 0]))
        return rc
    

def gci0_bot_loss_neg(data, class_rad, epsilon):
    """
    Compute GCI0_BOT (`C \sqsubseteq \bot`) negative loss

    :param data: GCI0_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) loss

    :param data: GCI1 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) negative loss

    :param data: GCI1 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) loss

    :param data: GCI1_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) negative loss

    :param data: GCI1_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) loss

    :param data: GCI2 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) negative loss

    :param data: GCI2 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) loss

    :param data: GCI3 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) negative loss

    :param data: GCI3 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param class_reg: class center regularization function
    :type class_reg: method
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
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
    """
    Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) loss

    :param data: GCI3_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci3_bot_loss_neg(data, class_rad, epsilon)
    else:
        rc = th.abs(class_rad(data[:, 1]))
        return rc
    
def gci3_bot_loss_neg(data, class_rad, epsilon):
    """
    Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) negative loss

    :param data: GCI3_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_rad: class radii embeddings
    :type class_rad: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    rc = th.abs(class_rad(data[:, 1]))
    loss = relu(epsilon - rc)
    return loss