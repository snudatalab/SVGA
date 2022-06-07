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
import torch
from torch import nn
from torch.nn import MSELoss
from torch_sparse import SparseTensor

from .gnn import SGC, GNN
from .losses import BernoulliLoss, GMRFLoss, GMRFSamplingLoss
import numpy as np


def to_x_loss(x_loss):
    """
    Make a loss term for estimating node features.
    """
    if x_loss in ['base', 'balanced']:
        return BernoulliLoss(x_loss)
    elif x_loss == 'gaussian':
        return MSELoss()
    else:
        raise ValueError(x_loss)


class Features(nn.Module):
    """
    Class that supports various types of node features.
    """

    def __init__(self, edge_index, num_nodes, version, obs_nodes=None, obs_features=None,
                 dropout=0):
        """
        Class initializer.
        """
        super().__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.indices = None
        self.values = None
        self.shape = None

        if version == 'diag':
            indices, values, shape = self.to_diag_features()
        elif version == 'degree':
            indices, values, shape = self.to_degree_features(edge_index)
        elif version == 'diag-degree':
            indices, values, shape = self.to_diag_degree_features(edge_index)
        elif version == 'obs-diag':
            indices, values, shape = self.to_obs_diag_features(obs_nodes, obs_features)
        else:
            raise ValueError(version)

        self.indices = nn.Parameter(indices, requires_grad=False)
        self.values = nn.Parameter(values, requires_grad=False)
        self.shape = shape

    def forward(self):
        """
        Make a feature Tensor from the current information.
        """
        return torch.sparse_coo_tensor(self.indices, self.values, size=self.shape,
                                       device=self.indices.device)

    def to_diag_features(self):
        """
        Make a diagonal feature matrix.
        """
        nodes = torch.arange(self.num_nodes)
        if self.training and self.dropout > 0:
            nodes = nodes[torch.rand(self.num_nodes) > self.dropout]
        indices = nodes.view(1, -1).expand(2, -1).contiguous()
        values = torch.ones(self.num_nodes)
        shape = self.num_nodes, self.num_nodes
        return indices, values, shape

    def to_degree_features(self, edge_index):
        """
        Make a degree-based feature matrix.
        """
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                             sparse_sizes=(self.num_nodes, self.num_nodes))
        degree = adj_t.sum(dim=0).long()
        degree_list = torch.unique(degree)
        degree_map = torch.zeros_like(degree)
        degree_map[degree_list] = torch.arange(len(degree_list))
        indices = torch.stack([torch.arange(self.num_nodes), degree_map[degree]], dim=0)
        values = torch.ones(indices.size(1))
        shape = self.num_nodes, indices[1, :].max() + 1
        return indices, values, shape

    def to_diag_degree_features(self, edge_index):
        """
        Combine the diagonal and degree-based feature matrices.
        """
        indices1, values1, shape1 = self.to_diag_features()
        indices2, values2, shape2 = self.to_degree_features(edge_index)
        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2])
        shape = shape1[0], shape1[1] + shape2[1]
        return indices, values, shape

    def to_obs_diag_features(self, obs_nodes, obs_features):
        """
        Combine the observed features and diagonal ones.
        """
        num_features = obs_features.size(1) + self.num_nodes - len(obs_nodes)
        row, col = torch.nonzero(obs_features, as_tuple=True)
        indices1 = torch.stack([obs_nodes[row], col])
        values1 = obs_features[row, col]

        nodes2 = torch.arange(self.num_nodes)
        nodes2[obs_nodes] = False
        nodes2 = torch.nonzero(nodes2).view(-1)
        indices2 = torch.stack([nodes2, torch.arange(len(nodes2))])
        indices2[1, :] += obs_features.size(1)
        values2 = torch.ones(indices2.size(1))

        indices = torch.cat([indices1, indices2], dim=1)
        values = torch.cat([values1, values2], dim=0)
        shape = self.num_nodes, num_features
        return indices, values, shape


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, dropout, conv):
        super().__init__()
        if conv == 'sgc':
            self.model = SGC(num_features, hidden_size, num_layers)
        elif conv == 'lin':
            self.model = nn.Linear(num_features, hidden_size)
        elif conv in ['gcn', 'sage', 'gat']:
            self.model = GNN(num_features, hidden_size, num_layers, hidden_size, dropout, conv=conv)
        else:
            raise ValueError()

    def forward(self, features, edge_index):
        if isinstance(self.model, nn.Linear):
            return self.model(features)
        else:
            return self.model(features, edge_index)


class Decoder(nn.Module):
    """
    Encoder network in the proposed framework.
    """

    def __init__(self, input_size, output_size, hidden_size=16, num_layers=2, dropout=0.5):
        """
        Class initializer.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            if i > 0:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(in_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Run forward propagation.
        """
        return self.layers(x)


class Identity(nn.Module):
    """
    PyTorch module that implements the identity function.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, x):
        """
        Run forward propagation.
        """
        return x


class UnitNorm(nn.Module):
    """
    Unit normalization of latent variables.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        valid_index = (vectors != 0).sum(1, keepdims=True) > 0
        vectors = torch.where(valid_index, vectors, torch.randn_like(vectors))
        return vectors / (vectors ** 2).sum(1, keepdims=True).sqrt()


class EmbNorm(nn.Module):
    """
    The normalization of node representations.
    """

    def __init__(self, hidden_size, function='unit', affine=True):
        """
        Class initializer.
        """
        super().__init__()
        if function == 'none':
            self.norm = Identity()
        elif function == 'unit':
            self.norm = UnitNorm()
        elif function == 'batchnorm':
            self.norm = nn.BatchNorm1d(hidden_size, affine=affine)
        elif function == 'layernorm':
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=affine)
        else:
            raise ValueError(function)

    def forward(self, vectors):
        """
        Run forward propagation.
        """
        return self.norm(vectors)


class SVGA(nn.Module):
    """
    Class of our proposed method.
    """

    def __init__(self, edge_index, num_nodes, num_features, num_classes, hidden_size=256, lamda=1,
                 beta=0.1, num_layers=2, conv='gcn', dropout=0.5, x_type='diag', x_loss='balanced',
                 emb_norm='unit', obs_nodes=None, obs_features=None, dec_bias=False):
        """
        Class initializer.
        """
        super().__init__()
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

        self.features = Features(edge_index, num_nodes, x_type, obs_nodes, obs_features)
        self.encoder = Encoder(self.features.shape[1], hidden_size, num_layers, dropout, conv)
        self.emb_norm = EmbNorm(hidden_size, emb_norm)

        self.x_decoder = nn.Linear(hidden_size, num_features, bias=dec_bias)
        self.y_decoder = nn.Linear(hidden_size, num_classes, bias=dec_bias)

        self.x_loss = to_x_loss(x_loss)
        self.y_loss = nn.CrossEntropyLoss()
        self.kld_loss = GMRFLoss(beta)

    def forward(self, edge_index, for_loss=False):
        """
        Run forward propagation.
        """
        z = self.emb_norm(self.encoder(self.features(), edge_index))
        z_dropped = self.dropout(z)
        x_hat = self.x_decoder(z_dropped)
        y_hat = self.y_decoder(z_dropped)
        if for_loss:
            return z, x_hat, y_hat
        return x_hat, y_hat

    def to_y_loss(self, y_hat, y_nodes, y_labels):
        """
        Make a loss term for observed labels.
        """
        if y_nodes is not None and y_labels is not None:
            return self.y_loss(y_hat[y_nodes], y_labels)
        else:
            return torch.zeros(1, device=y_hat.device)

    def to_kld_loss(self, z, edge_index):
        """
        Make a KL divergence regularizer.
        """
        if self.lamda > 0:
            return self.lamda * self.kld_loss(z, edge_index)
        else:
            return torch.zeros(1, device=z.device)

    def to_losses(self, edge_index, x_nodes, x_features, y_nodes=None, y_labels=None):
        """
        Make three loss terms for the training.
        """
        z, x_hat, y_hat = self.forward(edge_index, for_loss=True)
        l1 = self.x_loss(x_hat[x_nodes], x_features)
        l2 = self.to_y_loss(y_hat, y_nodes, y_labels)
        l3 = self.to_kld_loss(z, edge_index)
        return l1, l2, l3


class StochasticSVGA(nn.Module):
    """
    Stochastic version of our SVGA, without deterministic modeling.
    """
    def __init__(self, edge_index, num_nodes, num_features, num_classes, hidden_size=256, lamda=1,
                 beta=0.1, num_layers=2, conv='gcn', dropout=0.5, x_type='diag', x_loss='balanced',
                 emb_norm='unit', obs_nodes=None, obs_features=None, dec_bias=False):
        """
        Class initializer.
        """
        super().__init__()
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

        self.features = Features(edge_index, num_nodes, x_type, obs_nodes, obs_features)
        self.encoder_avg = Encoder(self.features.shape[1], hidden_size, num_layers, dropout, conv)
        self.encoder_std = Encoder(self.features.shape[1], hidden_size, num_layers, dropout, conv)
        self.emb_norm = EmbNorm(hidden_size, emb_norm)

        self.x_decoder = nn.Linear(hidden_size, num_features, bias=dec_bias)
        self.y_decoder = nn.Linear(hidden_size, num_classes, bias=dec_bias)

        self.x_loss = to_x_loss(x_loss)
        self.y_loss = nn.CrossEntropyLoss()
        self.kld_loss = GMRFSamplingLoss(beta)

    def forward(self, edge_index, for_loss=False):
        """
        Run forward propagation.
        """
        z_avg = self.encoder_avg(self.features(), edge_index)
        z_std = self.encoder_std(self.features(), edge_index)

        eps1 = torch.randn_like(z_avg)
        eps2 = torch.randn(z_std.size(1), z_avg.size(1), device=edge_index.device)
        z = z_avg + np.sqrt(self.kld_loss.beta) * eps1 + z_std.matmul(eps2)

        z = self.emb_norm(z)
        z_dropped = self.dropout(z)
        x_hat = self.x_decoder(z_dropped)
        y_hat = self.y_decoder(z_dropped)
        if for_loss:
            return z_avg, z_std, x_hat, y_hat
        return x_hat, y_hat

    def to_y_loss(self, y_hat, y_nodes, y_labels):
        """
        Make a loss term for observed labels.
        """
        if y_nodes is not None and y_labels is not None:
            return self.y_loss(y_hat[y_nodes], y_labels)
        else:
            return torch.zeros(1, device=y_hat.device)

    def to_kld_loss(self, z_avg, z_std, edge_index):
        """
        Make a KL divergence regularizer.
        """
        if self.lamda > 0:
            return self.lamda * self.kld_loss(z_avg, z_std, edge_index)
        else:
            return torch.zeros(1, device=z_avg.device)

    def to_losses(self, edge_index, x_nodes, x_features, y_nodes=None, y_labels=None):
        """
        Make three loss terms for the training.
        """
        z_avg, z_std, x_hat, y_hat = self.forward(edge_index, for_loss=True)
        l1 = self.x_loss(x_hat[x_nodes], x_features)
        l2 = self.to_y_loss(y_hat, y_nodes, y_labels)
        l3 = self.to_kld_loss(z_avg, z_std, edge_index)
        return l1, l2, l3
