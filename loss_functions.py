import typing
import warnings
import torch.nn.functional as F
import torch


class TransformLoss(torch.nn.Module):
    def __init__(self, full_transform: bool):
        super(TransformLoss, self).__init__()

        self.mse = torch.nn.MSELoss()

        if full_transform:
            target = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float).view(1, 9)
            self.register_buffer('target', target)
        else:
            self.register_buffer('target', torch.zeros(1, 1))

    def forward(self, code: torch.Tensor):
        target = self.target.expand_as(code)
        return self.mse(code, target)


def calc_l2_reg(tensor: torch.Tensor):
    """
    Calculate l2 norm
    :param tensor:
    :return:
    """
    reg = (tensor ** 2).mean()
    return reg


def calc_l1_reg(tensor: torch.Tensor):
    """
    Calculate l1 norm
    :param tensor:
    :return:
    """
    reg = tensor.abs().mean()
    return reg


def weight_norm_l2(tensor: torch.Tensor):
    """
    Return l2 norm of given tensor
    :param tensor:
    :return:
    """
    return torch.norm(tensor, p='fro')


def calc_weights_reg(weights: torch.Tensor, fg_probability: torch.Tensor, eps: float = 0.01):
    """
    Constrain the sum of weights to be 1 in foreground and 0 in background
    :param weights: (n, i) [0, 1)
    :param fg_probability: (n, i) (0, 1)
    :param eps:
    :return:
    """
    assert eps >= 0.0, 'eps must be non-negative.'

    # mask to split fore- and back- ground
    fg_mask = (fg_probability > 0.5).float()

    # classification confidence
    conf = torch.clamp(((fg_probability - 0.5).abs() + eps) / 0.5, min=0.0, max=1.0)

    # L1-norm for regularization, L2-norm can cause blur along edges
    reg = fg_mask * (1.0 - eps - weights).clamp(min=0.0) + (1.0 - fg_mask) * weights

    # weighted sum
    loss = (reg * conf).sum() / (conf.sum() + 1.0e-6)

    return loss


def calc_mse_prob(x: torch.Tensor, y: torch.Tensor, fg_probability: torch.Tensor):
    """
    Calculates the mean square error between tensors `x` and `y`.
    :param x: (n, i, 3)
    :param y: (n, i, 3)
    :param fg_probability: (n, i, 1)
    """
    error = (x - y) ** 2
    assert fg_probability.dim() == 3
    weight = fg_probability.expand_as(error)
    loss = (weight * error).sum() / (weight.sum() + 1.0e-6)
    return loss


def calc_weights_reg_conf(weights: torch.Tensor, fg_probability: torch.Tensor, beta: torch.Tensor):
    """
    Compute weights_reg with uncertainty
    :param weights: (n, i) [0, 1)
    :param fg_probability: (n, i) (0, 1)
    :param beta: uncertainty, (n, i)
    :return:
    """
    # use l1 norm to get a clearer edge
    reg = fg_probability * (1.0 - weights) + (1.0 - fg_probability) * weights

    # aleatoric uncertainty
    reg = reg / (2.0 * beta) + 0.5 * torch.log(beta)

    return reg.mean()


def calc_mask_reg_conf(opacity: torch.Tensor, fg_probability: torch.Tensor, uncertainty: torch.Tensor):
    """
    An adapted version of calc_weights_reg_conf that use hard label of fg_probability to identify the fore- and back-
    ground
    :param opacity:
    :param fg_probability:
    :param uncertainty:
    :return:
    """
    # 1 - fg, 0 - bg
    mask = (fg_probability >= 0.5).float()

    # use l2 norm to measure the distance
    reg = (opacity - mask).square()

    if uncertainty is not None:
        reg = reg / (2.0 * uncertainty) + 0.5 * torch.log(uncertainty)

    return reg.mean()


def calc_mse_beta(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor):
    error = (x - y).square() / (2.0 * beta.square()) + torch.log(beta)
    return error.mean()


def calc_weighted_mse(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    """
    Calc weighted mse. The weight is computed according to the min difference of degree between target and
    source cameras
    :param x: (n, i, 3)
    :param y: (n, i, 3)
    :param w: (n,)
    :return:
    """
    assert w.dim() == 1

    # compute normed weight, (n,)
    norm_w = w / (w.sum() + 1.0e-6)

    # compute weighted error, (n,)
    error = (x - y).square().mean(dim=(1, 2))
    loss = (error * norm_w).sum()

    return loss


def calc_mse_conf():
    pass


def calc_deformation_loss(deformations: torch.Tensor, rays_points: torch.Tensor, pairs_per_image: int,
                          epsilon: float):
    """
    Constrain the deformation between paris
    :param deformations: (n, i, p, 3)
    :param rays_points: (n, i, p, 3)
    :param pairs_per_image:
    :param epsilon:
    :return:
    """
    n, i, p, _ = deformations.shape #batch,chunk,sample

    deformations = deformations.view(n, i * p, 3)
    rays_points = rays_points.detach().view(n, i * p, 3)  # rays points require no grad

    rd1 = []
    rp1 = []

    rd2 = []
    rp2 = []

    for b in range(n):
        ids1 = torch.randint(0, i * p, (pairs_per_image,))
        ids2 = torch.randint(0, i * p, (pairs_per_image,))

        rd1.append(deformations[b, ids1, :])
        rp1.append(rays_points[b, ids1, :])

        rd2.append(deformations[b, ids2, :])
        rp2.append(rays_points[b, ids2, :])

    rd1 = torch.stack(rd1, dim=0)
    rp1 = torch.stack(rp1, dim=0)

    rd2 = torch.stack(rd2, dim=0)
    rp2 = torch.stack(rp2, dim=0)

    loss = torch.relu(torch.norm(rd1 - rd2, dim=-1) / (torch.norm(rp1 - rp2, dim=-1) + 1.0e-6) - epsilon).mean()

    return loss


class KeyPointMatchLoss(torch.nn.Module):
    """
    Key point loss that constraint the match of front and back
    """
    def __init__(self, target_points: torch.Tensor, tol: float):
        """
        Initialize
        :param target_points: key points to align with, (p, 3)
        :param tol:
        """
        super(KeyPointMatchLoss, self).__init__()

        p, _3 = target_points.shape
        assert _3 == 3
        assert p >= 2, f'Require at least two target points.'
        assert 0.0 < tol <= 1.0
        if tol >= 0.5:
            warnings.warn(f'{tol} is bigger than the suggested maximum 0.5.')

        self.register_buffer('_target_points', target_points)

        # compute the min distance of each point to others
        with torch.no_grad():
            distances = (target_points[None, :, :] - target_points[:, None, :]).norm(dim=-1)  # (p, p)
            top2, _ = torch.topk(distances, k=2, dim=0, largest=False, sorted=True)  # (2, p)
            min_distance = top2[1, :]  # (p,)
        self.register_buffer('_min_distances', min_distance)

        self._tol = tol

        # # smooth L1 loss
        # self._smooth_l1 = torch.nn.SmoothL1Loss()
        # self.register_buffer('_loss_target', torch.zeros(1, dtype=torch.float), persistent=False)

    def forward(self, input_points: torch.Tensor, input_ids: typing.List[int] = None):
        """
        Randomly select one pair of points and compute the match loss
        :param input_points: points stacked in order, (n, p, 3)
        :param input_ids: the indexes of input points
        :return:
        """
        # select
        if input_ids is not None:
            target_points = self._target_points[input_ids, :]
            min_distances = self._min_distances[input_ids]
        else:
            target_points = self._target_points
            min_distances = self._min_distances

        # compute distance to target point
        distances = (input_points - target_points[None, :, :]).norm(dim=-1)  # (n, p)
        # norm
        dist_norm = distances / (min_distances[None, :] + 1.0e-6)
        dist_norm = torch.relu(dist_norm - self._tol)  # (n, p)
        # # smooth L1 loss
        # loss = self._smooth_l1(dist_norm, self._loss_target.expand_as(dist_norm))

        # l1 loss
        loss = dist_norm.mean()

        return loss
