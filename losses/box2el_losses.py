import torch as th
from torch.nn.functional import relu

def box_distance(box_a, box_b):
    """
    Compute the distance between two boxes

    :param box_a: the first box
    :type box_a: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param box_b: the second box
    :type box_b: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :return: distance between given two boxes
    :return type: torch.Tensor(torch.float64)
    """
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    dist = th.abs(center_a - center_b) - offset_a - offset_b
    return dist


def box_intersection(box_a, box_b):
    """
    Find a box -- intersection of two given boxes

    :param box_a: the first box
    :type box_a: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param box_b: the second box
    :type box_b: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :return: box corresponding to intersection between given two boxes
    :return type: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :return: left lower corner of the intersecting box
    :return type: torch.Tensor(torch.float64)
    :return: right upper corner of the intersecting box
    :return type: torch.Tensor(torch.float64)
    """
    center_a, offset_a = box_a
    center_b, offset_b = box_b

    lower = th.maximum(center_a - offset_a, center_b - offset_b)
    upper = th.minimum(center_a + offset_a, center_b + offset_b)
    centers = (lower + upper) / 2
    offsets = th.abs(upper - lower) / 2
    intersection = (centers, offsets)
    return intersection, lower, upper


def inclusion_score(box_a, box_b, gamma):
    """
    Compute the inclusion score

    :param box_a: the first box
    :type box_a: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param box_b: the second box
    :type box_b: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param gamma: margin parameter
    :type gamma: float
    :return: inclusion score
    :return type: torch.Tensor(torch.float64)
    """
    dist_a_b = box_distance(box_a, box_b)
    _, offset_a = box_a
    score = th.linalg.norm(th.relu(dist_a_b + 2 * offset_a - gamma), dim=1)
    return score


def minimal_distance(box_a, box_b, gamma):
    """
    Compute the minimal distance between two boxes

    :param box_a: the first box
    :type box_a: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param box_b: the second box
    :type box_b: tuple(torch.Tensor(torch.float64), torch.Tensor(torch.float64))
    :param gamma: margin parameter
    :type gamma: float
    :return: minimal distance between given two boxes
    :return type: torch.Tensor(torch.float64)
    """
    dist = box_distance(box_a, box_b)
    min_dist = th.linalg.norm(th.relu(dist + gamma), dim=1)
    return min_dist


def gci0_loss(
    data,
    class_center,
    class_offset,
    gamma,
    epsilon,
    neg=False,
):
    """
    Compute GCI0 (`C \sqsubseteq D`) loss

    :param data: GCI0 data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
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
            class_center,
            class_offset,
            gamma,
            epsilon,
            neg=False,
        )
    else:
        center_c = class_center(data[:, 0])
        center_d = class_center(data[:, 1])
        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))
        box_c = (center_c, off_c)
        box_d = (center_d, off_d)
        score = inclusion_score(box_c, box_d, gamma)
        return score


def gci0_bot_loss(data, class_offset, epsilon, neg=False):
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
        return gci0_bot_loss_neg(data, class_offset, epsilon)
    else:
        off_c = th.abs(class_offset(data[:, 0]))
        score = th.linalg.norm(off_c, dim=1)
        return score


def gci0_bot_loss_neg(data, class_offset, epsilon):
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
    loss = relu(epsilon - th.linalg.norm(off_c, axis=1))
    return loss


def gci1_loss(
    data,
    class_center,
    class_offset,
    gamma,
    neg=False,
):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) loss

    :param data: GCI1 data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci1_loss_neg(data, class_center, class_offset, gamma)
    else:
        center_c = class_center(data[:, 0])
        center_d = class_center(data[:, 1])
        center_e = class_center(data[:, 2])
        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))
        off_e = th.abs(class_offset(data[:, 2]))

        box_c = (center_c, off_c)
        box_d = (center_d, off_d)
        box_e = (center_e, off_e)

        intersection, lower, upper = box_intersection(box_c, box_d)
        box_incl_score = inclusion_score(intersection, box_e, gamma)

        additional_score = th.linalg.norm(th.relu(lower - upper), dim=1)
        score = box_incl_score + additional_score
        return score


def gci1_loss_neg(data, class_center, class_offset, gamma):
    """
    Compute GCI1 (`C \sqcap D \sqsubseteq E`) negative loss

    :param data: GCI1 data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    center_c = class_center(data[:, 0])
    center_d = class_center(data[:, 1])
    center_e = class_center(data[:, 2])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))
    off_e = th.abs(class_offset(data[:, 2]))

    box_c = (center_c, off_c)
    box_d = (center_d, off_d)
    box_e = (center_e, off_e)

    intersection, lower, upper = box_intersection(box_c, box_d)
    box_dist = box_distance(intersection, box_e)
    score1 = th.linalg.norm(th.relu(-box_dist - gamma), dim=1)

    additional_score = th.linalg.norm(th.relu(lower - upper), dim=1)
    score = score1 + additional_score
    return score


def gci1_bot_loss(
    data,
    class_center,
    class_offset,
    gamma,
    epsilon,
    neg=False,
):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) loss

    :param data: GCI1_BOT data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci1_bot_loss_neg(data, class_center, class_offset, gamma, epsilon)
    else:
        center_c = class_center(data[:, 0])
        center_d = class_center(data[:, 1])
        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 1]))

        box_c = (center_c, off_c)
        box_d = (center_d, off_d)

        box_dist = box_distance(box_c, box_d)
        score = th.linalg.norm(th.relu(-box_dist - gamma), dim=1)
        return score


def gci1_bot_loss_neg(data, class_center, class_offset, gamma, epsilon):
    """
    Compute GCI1_BOT (`C \sqcap D \sqsubseteq \bot`) negative loss

    :param data: GCI1_BOT negative data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param epsilon: $\varepsilon$ parameter for negative loss computation
    :type epsilon: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    center_c = class_center(data[:, 0])
    center_d = class_center(data[:, 1])
    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 1]))

    box_c = (center_c, off_c)
    box_d = (center_d, off_d)

    intersection, lower, upper = box_intersection(box_c, box_d)
    center_intersection, off_intersection = intersection
    loss = relu(epsilon - th.linalg.norm(off_intersection, axis=1))
    return loss


def gci2_loss(
    data,
    class_center,
    class_offset,
    head_center,
    head_offset,
    tail_center,
    tail_offset,
    bump,
    gamma,
    delta,
    neg=False,
):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) loss

    :param data: GCI2 data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param head_center: relations' head centers' embeddings
    :type head_center: torch.nn.modules.sparse.Embedding
    :param head_offset: relations' head offsets' embeddings
    :type head_offset: torch.nn.modules.sparse.Embedding
    :param tail_center: relations' tail centers' embeddings
    :type tail_center: torch.nn.modules.sparse.Embedding
    :param tail_offset: relations' tail offsets' embeddings
    :type tail_offset: torch.nn.modules.sparse.Embedding
    :param bump: bump vectors' embeddings
    :type bump: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param delta: $\delta$ parameter for negative loss computation
    :type delta: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci2_loss_neg(
            data,
            class_center,
            class_offset,
            head_center,
            head_offset,
            tail_center,
            tail_offset,
            bump,
            gamma,
            delta,
        )
    else:
        center_c = class_center(data[:, 0])

        center_head = head_center(data[:, 1])
        offset_head = th.abs(head_offset(data[:, 1]))

        center_tail = tail_center(data[:, 1])
        offset_tail = th.abs(tail_offset(data[:, 1]))

        center_d = class_center(data[:, 2])

        off_c = th.abs(class_offset(data[:, 0]))
        off_d = th.abs(class_offset(data[:, 2]))

        bump_c = bump(data[:, 0])
        bump_d = bump(data[:, 2])

        box_head = (center_head, offset_head)
        box_tail = (center_tail, offset_tail)

        bumped_c = (center_c + bump_d, off_c)
        bumped_d = (center_d + bump_c, off_d)

        inclussion_1 = inclusion_score(bumped_c, box_head, gamma)
        inclussion_2 = inclusion_score(bumped_d, box_tail, gamma)

        score = (inclussion_1 + inclussion_2) / 2
        return score


def gci2_loss_neg(
    data,
    class_center,
    class_offset,
    head_center,
    head_offset,
    tail_center,
    tail_offset,
    bump,
    gamma,
    delta,
):
    """
    Compute GCI2 (`C \sqsubseteq \exists R.D`) negative loss

    :param data: GCI2 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param head_center: relations' head centers' embeddings
    :type head_center: torch.nn.modules.sparse.Embedding
    :param head_offset: relations' head offsets' embeddings
    :type head_offset: torch.nn.modules.sparse.Embedding
    :param tail_center: relations' tail centers' embeddings
    :type tail_center: torch.nn.modules.sparse.Embedding
    :param tail_offset: relations' tail offsets' embeddings
    :type tail_offset: torch.nn.modules.sparse.Embedding
    :param bump: bump vectors' embeddings
    :type bump: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param delta: $\delta$ parameter for negative loss computation
    :type delta: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    center_c = class_center(data[:, 0])
    center_head = head_center(data[:, 1])
    offset_head = th.abs(head_offset(data[:, 1]))
    center_tail = tail_center(data[:, 1])
    offset_tail = th.abs(tail_offset(data[:, 1]))
    center_d = class_center(data[:, 2])
    bump_c = bump(data[:, 0])
    bump_d = bump(data[:, 2])

    off_c = th.abs(class_offset(data[:, 0]))
    off_d = th.abs(class_offset(data[:, 2]))

    box_head = (center_head, offset_head)
    box_tail = (center_tail, offset_tail)

    bumped_c = (center_c + bump_d, off_c)
    bumped_d = (center_d + bump_c, off_d)

    first_part = (delta - minimal_distance(bumped_c, box_head, gamma)).square().mean()
    second_part = (delta - minimal_distance(bumped_d, box_tail, gamma)).square().mean()

    loss = first_part + second_part
    reg_loss = 0  
    return loss + reg_loss


def gci3_loss(
    data,
    class_center,
    class_offset,
    head_center,
    head_offset,
    bump,
    gamma,
    delta,
    neg=False,
):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) loss

    :param data: GCI3 data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param head_center: relations' head centers' embeddings
    :type head_center: torch.nn.modules.sparse.Embedding
    :param head_offset: relations' head offsets' embeddings
    :type head_offset: torch.nn.modules.sparse.Embedding
    :param bump: bump vectors' embeddings
    :type bump: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param delta: $\delta$ parameter for negative loss computation
    :type delta: float
    :param neg: whether to compute negative or positive loss
    :type neg: bool
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    if neg:
        return gci3_loss_neg(
            data,
            class_center,
            class_offset,
            head_center,
            head_offset,
            bump,
            gamma,
            delta,
        )
    else:
        center_d = class_center(data[:, 2])
        off_d = th.abs(class_offset(data[:, 2]))

        center_head = head_center(data[:, 0])
        offset_head = th.abs(head_offset(data[:, 0]))

        bump_c = bump(data[:, 1])

        bumped_head = (center_head - bump_c, offset_head)
        box_d = (center_d, off_d)
        score = inclusion_score(bumped_head, box_d, gamma)
        return score


def gci3_loss_neg(
    data,
    class_center,
    class_offset,
    head_center,
    head_offset,
    bump,
    gamma,
    delta,
):
    """
    Compute GCI3 (`\exists R.C \sqsubseteq D`) negative loss

    :param data: GCI3 negative data
    :type data: torch.Tensor(torch.int64)
    :param class_center: class centers' embeddings
    :type class_center: torch.nn.modules.sparse.Embedding
    :param class_offset: class offsets' embeddings
    :type class_offset: torch.nn.modules.sparse.Embedding
    :param head_center: relations' head centers' embeddings
    :type head_center: torch.nn.modules.sparse.Embedding
    :param head_offset: relations' head offsets' embeddings
    :type head_offset: torch.nn.modules.sparse.Embedding
    :param bump: bump vectors' embeddings
    :type bump: torch.nn.modules.sparse.Embedding
    :param gamma: margin parameter 
    :type gamma: float/int
    :param delta: $\delta$ parameter for negative loss computation
    :type delta: float
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    center_d = class_center(data[:, 2])
    off_d = th.abs(class_offset(data[:, 2]))

    center_head = head_center(data[:, 0])
    offset_head = th.abs(head_offset(data[:, 0]))

    bump_c = bump(data[:, 1])
    bumped_head = (center_head - bump_c, offset_head)

    box_d = (center_d, off_d)

    loss = (delta - minimal_distance(bumped_head, box_d, gamma)).square().mean()
    reg_loss = 0  
    return loss + reg_loss


def gci3_bot_loss(data, class_offset, epsilon, neg=False):
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
        return gci3_bot_loss_neg(data, class_offset, epsilon)
    else:
        off_c = th.abs(class_offset(data[:, 1]))
        score = th.linalg.norm(off_c, dim=1)
        return score


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
    loss = relu(epsilon - th.linalg.norm(off_c, axis=1))
    return loss


def reg_loss(bump, reg_factor):
    """
    Regularization loss

    :param bump: bump vectors' embeddings
    :type bump: torch.nn.modules.sparse.Embedding
    :param reg_factor: regularization factor
    :type reg_factor: float/int
    :return: loss value for each data sample
    :return type: torch.Tensor(torch.float64)
    """
    reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).mean()
    return reg_loss