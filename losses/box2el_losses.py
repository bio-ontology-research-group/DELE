import torch as th
from torch.nn.functional import relu

def box_distance(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b
    dist = th.abs(center_a - center_b) - offset_a - offset_b
    return dist


def box_intersection(box_a, box_b):
    center_a, offset_a = box_a
    center_b, offset_b = box_b

    lower = th.maximum(center_a - offset_a, center_b - offset_b)
    upper = th.minimum(center_a + offset_a, center_b + offset_b)
    centers = (lower + upper) / 2
    offsets = th.abs(upper - lower) / 2
    intersection = (centers, offsets)
    return intersection, lower, upper


def inclusion_score(box_a, box_b, gamma):
    dist_a_b = box_distance(box_a, box_b)
    _, offset_a = box_a
    score = th.linalg.norm(th.relu(dist_a_b + 2 * offset_a - gamma), dim=1)
    return score


def minimal_distance(box_a, box_b, gamma):
    dist = box_distance(box_a, box_b)
    min_dist = th.linalg.norm(th.relu(dist + gamma), dim=1)
    return min_dist


def gci0_loss(
    data,
    class_center,
    class_offset,
    gamma,
    embed_dim,
    epsilon,
    neg=False,
):
    if neg:
        return gci1_bot_loss(
            data,
            class_center,
            class_offset,
            gamma,
            embed_dim,
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


def gci0_bot_loss(data, embed_dim, class_offset, epsilon, neg=False):
    if neg:
        return gci0_bot_loss_neg(data, embed_dim, class_offset, epsilon)
    else:
        off_c = th.abs(class_offset(data[:, 0]))
        score = th.linalg.norm(off_c, dim=1)
        return score


def gci0_bot_loss_neg(data, embed_dim, class_offset, epsilon):
    off_c = th.abs(class_offset(data[:, 0]))
    loss = relu(epsilon - th.linalg.norm(off_c, axis=1))
    return loss


def gci1_loss(
    data,
    class_center,
    class_offset,
    gamma,
    embed_dim,
    neg=False,
):
    if neg:
        return gci1_loss_neg(data, class_center, class_offset, gamma, embed_dim)
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


def gci1_loss_neg(data, class_center, class_offset, gamma, embed_dim):
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
    embed_dim,
    epsilon,
    neg=False,
):
    if neg:
        return gci1_bot_loss_neg(data, class_center, class_offset, gamma, embed_dim, epsilon)
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


def gci1_bot_loss_neg(data, class_center, class_offset, gamma, embed_dim, epsilon):
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
    embed_dim,
    neg=False,
):
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
            embed_dim,
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
    embed_dim,
):
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
    embed_dim,
    neg=False,
):
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
            embed_dim,
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
    embed_dim,
):
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


def gci3_bot_loss(data, embed_dim, class_offset, epsilon, neg=False):
    if neg:
        return gci3_bot_loss_neg(data, embed_dim, class_offset, epsilon)
    else:
        off_c = th.abs(class_offset(data[:, 1]))
        score = th.linalg.norm(off_c, dim=1)
        return score


def gci3_bot_loss_neg(data, embed_dim, class_offset, epsilon):
    off_c = th.abs(class_offset(data[:, 1]))
    loss = relu(epsilon - th.linalg.norm(off_c, axis=1))
    return loss


def reg_loss(bump, reg_factor):
    reg_loss = reg_factor * th.linalg.norm(bump.weight, dim=1).mean()
    return reg_loss