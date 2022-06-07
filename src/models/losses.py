"""
Accurate Node Feature Estimation with Structured Variational Graph Autoencoder
(KDD 2022)

Authors:
- Jaemin Yoo (jaeminyoo@cmu.edu), Carnegie Mellon University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- Jinhong Jung (jinhongjung@jbnu.ac.kr), Jeonbuk National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
from torch import nn
import torch
from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor


def to_laplacian(edge_index, num_nodes):
    """
    Make a graph Laplacian term for the GMRF loss.
    """
    if isinstance(edge_index, SparseTensor):
        row = edge_index.storage.row()
        col = edge_index.storage.col()
        edge_index = torch.stack([row, col])
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    size = num_nodes, num_nodes
    return torch.sparse_coo_tensor(edge_index, edge_weight, size=size, device=edge_index.device)


def to_mean_loss(features, laplacian):
    """
    Compute the loss term that compares features of adjacent nodes.
    """
    return torch.bmm(features.t().unsqueeze(1), laplacian.matmul(features).t().unsqueeze(2)).view(-1)


class BernoulliLoss(nn.Module):
    """
    Loss term for the binary features.
    """

    def __init__(self, version='base'):
        """
        Class initializer.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.version = version

    def forward(self, input, target):
        """
        Run forward propagation.
        """
        assert (target < 0).sum() == 0
        if self.version == 'base':
            loss = self.loss(input, target)
        elif self.version == 'balanced':
            pos_ratio = (target > 0).float().mean()
            weight = torch.ones_like(target)
            weight[target > 0] = 1 / (2 * pos_ratio)
            weight[target == 0] = 1 / (2 * (1 - pos_ratio))
            loss = self.loss(input, target) * weight
        else:
            raise ValueError(self.version)
        return loss.mean()


class GMRFLoss(nn.Module):
    """
    Implementation of the GMRF loss.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()
        self.cached_adj = None
        self.beta = beta

    def forward(self, features, edge_index):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = to_laplacian(edge_index, features.size(0))

        num_nodes = features.size(0)
        hidden_dim = features.size(1)
        eye = torch.eye(hidden_dim, device=features.device)
        l1 = (eye + features.t().matmul(features) / self.beta).logdet()
        l2 = to_mean_loss(features, self.cached_adj).sum()
        return (l2 - l1 / 2) / num_nodes


class GMRFSamplingLoss(nn.Module):
    """
    Implementation of the GMRF loss without deterministic modeling.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()
        self.cached_adj = None
        self.beta = beta

    def forward(self, z_mean, z_std, edge_index):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = to_laplacian(edge_index, z_mean.size(0))

        device = edge_index.device
        num_nodes = z_mean.size(0)
        rank = z_std.size(1)
        var_mat = self.beta * torch.eye(num_nodes, device=device) + z_std.matmul(z_std.t())

        eye = torch.eye(rank, device=device)
        l1 = (eye + z_std.t().matmul(z_std) / self.beta).logdet()
        l2 = self.cached_adj.matmul(var_mat).diagonal().sum()
        l3 = to_mean_loss(z_mean, self.cached_adj).sum()

        return (l3 / z_mean.size(1) + l2 - l1) / (2 * num_nodes)
