import torch as th
from torch.nn.functional import relu


def gci0_loss(
    data,
    class_embed,
    class_offset,
    margin,
    epsilon,
    neg=False,
):
    """
    Compute GCI0 (`C \sqsubseteq D`) loss

    :param data: GCI0 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci1_bot_loss(
            data,
            class_embed,
            class_offset,
            margin,
            epsilon,
            neg=False,
        )

    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])

        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))

        euc = th.abs(c - d)
        dst = th.reshape(
            th.linalg.norm(relu(euc + off_c - off_d - margin), axis=1), [-1, 1]
        )

        return dst


def gci0_bot_loss(
    data, class_offset, epsilon, neg=False
):
    """
    Compute GCI0_BOT (`C \sqsubseteq \bot`) loss

    :param data: GCI0_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci0_bot_loss_neg(
            data, class_offset, epsilon
        )
    else:
        off_c = th.abs(class_offset(data[:, 0]))
        loss = th.reshape(th.linalg.norm(off_c, axis=1), [-1, 1])
        return loss


def gci0_bot_loss_neg(
    data,
    class_offset,
    epsilon,
):
    """
    Compute GCI0_BOT (`C \sqsubseteq \bot`) negative loss

    :param data: GCI0_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    off_c = th.abs(class_offset(data[:, 0]))
    loss = th.reshape(relu(epsilon - th.linalg.norm(off_c, axis=1)), [-1, 1])
    return loss


def gci1_loss(
    data,
    class_embed,
    class_offset,
    margin,
    neg=False,
):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) loss

    :param data: GCI1 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
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
            class_offset,
            margin,
        )

    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])
        e = class_embed(data[:, 2])

        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))
        off_e = th.abs(class_offset(data[:, 2]))

        startAll = th.maximum(c - off_c, d - off_d)
        endAll = th.minimum(c + off_c, d + off_d)

        new_offset = th.abs(startAll - endAll) / 2

        cen1 = (startAll + endAll) / 2
        euc = th.abs(cen1 - e)

        dst = th.reshape(
            th.linalg.norm(relu(euc + new_offset - off_e - margin), axis=1),
            [-1, 1],
        ) + th.reshape(th.linalg.norm(relu(startAll - endAll), axis=1), [-1, 1])
        return dst


def gci1_loss_neg(
    data,
    class_embed,
    class_offset,
    margin,
):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) negative loss

    :param data: GCI1 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])
    e = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))

    startAll = th.maximum(c - off_c, d - off_d)
    endAll = th.minimum(c + off_c, d + off_d)

    new_offset = th.abs(startAll - endAll) / 2

    cen1 = (startAll + endAll) / 2
    euc = th.abs(cen1 - e)

    dst = th.reshape(
        th.linalg.norm(relu(-euc + new_offset + off_e + margin), axis=1),
        [-1, 1],
    ) + th.reshape(th.linalg.norm(relu(startAll - endAll), axis=1), [-1, 1])

    return dst


def gci1_bot_loss(
    data,
    class_embed,
    class_offset,
    margin,
    epsilon,
    neg=False,
):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) loss

    :param data: GCI1_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci1_bot_loss_neg(
            data,
            class_embed,
            class_offset,
            margin,
            epsilon,
        )
    else:
        c = class_embed(data[:, 0])
        d = class_embed(data[:, 1])

        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))

        euc = th.abs(c - d)
        dst = th.reshape(
            th.linalg.norm(relu(-euc + off_c + off_d + margin), axis=1), [-1, 1]
        )
        return dst


def gci1_bot_loss_neg(
    data,
    class_embed,
    class_offset,
    margin,
    epsilon,
):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) negative loss

    :param data: GCI1_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    c = class_embed(data[:, 0])
    d = class_embed(data[:, 1])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))

    startAll = th.maximum(c - off_c, d - off_d)
    endAll = th.minimum(c + off_c, d + off_d)

    new_offset = th.abs(startAll - endAll) / 2

    dst = th.reshape(relu(epsilon - th.linalg.norm(new_offset, axis=1)), [-1, 1])
    return dst


def gci2_loss(
    data,
    class_embed,
    class_offset,
    rel_embed,
    margin,
    neg=False,
):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) loss

    :param data: GCI2 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
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
            class_offset,
            rel_embed,
            margin,
        )

    else:
        c = class_embed(data[:, 0])
        r = rel_embed(data[:, 1])
        d = class_embed(data[:, 2])

        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 2]))

        euc = th.abs(c + r - d)
        dst = th.reshape(
            th.linalg.norm(relu(euc + off_c - off_d - margin), axis=1), [-1, 1]
        )
        return dst


def gci2_loss_neg(
    data,
    class_embed,
    class_offset,
    rel_embed,
    margin,
):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) negative loss

    :param data: GCI2 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    c = class_embed(data[:, 0])
    r = rel_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 2]))

    euc = th.abs(c + r - d)
    dst = th.reshape(
        th.linalg.norm(relu(-euc + off_c + off_d + margin), axis=1), [-1, 1]
    )
    return dst


def gci3_loss(
    data,
    class_embed,
    class_offset,
    rel_embed,
    margin,
    neg=False,
):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) loss

    :param data: GCI3 data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
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
            class_offset,
            rel_embed,
            margin,
        )

    else:
        r = rel_embed(data[:, 0])
        c = class_embed(data[:, 1])
        d = class_embed(data[:, 2])

        off_c = th.abs(class_offset(data[:, 1]))
        off_d = th.abs(class_offset(data[:, 2]))

        euc = th.abs(c - r - d)
        dst = th.reshape(
            th.linalg.norm(relu(euc + off_c - off_d - margin), axis=1), [-1, 1]
        )
        return dst


def gci3_loss_neg(
    data,
    class_embed,
    class_offset,
    rel_embed,
    margin,
):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) negative loss

    :param data: GCI3 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_embed: class centers' embeddings
    :type class_embed: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param rel_embed: relations' embeddings
    :type rel_embed: torch.nn.modules.sparse.Embedding
    :param margin: margin parameter \gamma
    :type margin: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    r = rel_embed(data[:, 0])
    c = class_embed(data[:, 1])
    d = class_embed(data[:, 2])

    off_c = th.abs(class_offset(data[:, 1]))
    off_d = th.abs(class_offset(data[:, 2]))

    euc = th.abs(c - r - d)
    dst = th.reshape(
        th.linalg.norm(relu(-euc + off_c + off_d + margin), axis=1), [-1, 1]
    )
    return dst


def gci3_bot_loss(
    data, class_offset, epsilon, neg=False
):
    """
    Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) loss

    :param data: GCI3_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci3_bot_loss_neg(
            data, class_offset, epsilon
        )
    else:
        off_c = th.abs(class_offset(data[:, 1]))
        loss = th.reshape(th.linalg.norm(off_c, axis=1), [-1, 1])
        return loss


def gci3_bot_loss_neg(data, class_offset, epsilon):
    """
    Compute GCI3_BOT (`\exists R.C \sqsubseteq \bot`) loss

    :param data: GCI3_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    off_c = th.abs(class_offset(data[:, 1]))
    loss = th.reshape(relu(epsilon - th.linalg.norm(off_c, axis=1)), [-1, 1])
    return loss