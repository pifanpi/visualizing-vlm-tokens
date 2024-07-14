import torch
import torch.nn.functional as F


def all_to_all_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = (a**2).sum(1).view(-1, 1)
    b_norm = (b**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.mm(a, b.transpose(0, 1))
    return torch.sqrt(torch.clamp(dist, min=0.0))


def all_to_all_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())