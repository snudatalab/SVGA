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
from torch_geometric.nn import GATConv
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor, nn
from torch.nn import Linear
from torch.nn import functional as func
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv


# noinspection PyMethodOverriding
class SGConv(MessagePassing):
    """
    Convolution layer for a simplified graph convolutional network.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        """
        Class initializer.
        """
        kwargs.setdefault('aggr', 'add')
        super(SGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters in the linear layer.
        """
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Run forward propagation.
        """
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)

        x = self.lin(x)

        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Message function for the PyG implementation.
        """
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Message and aggregate function for the PyG implementation.
        """
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        """
        Make a representation of this layer.
        """
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


# noinspection PyMethodOverriding
class SAGEConv(MessagePassing):
    """
    Convolution layer for GraphSAGE.
    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True, **kwargs):
        """
        Class initializer.
        """
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters in the layers.
        """
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """
        Run forward propagation.
        """
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.lin_l(x[0])
        out = self.propagate(edge_index, x=out, size=size)
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)
        if self.normalize:
            out = func.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j):
        """
        Message function for the PyG implementation.
        """
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """
        Message and aggregate function for the PyG implementation.
        """
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        """
        Make a representation of this layer.
        """
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SGC(nn.Module):
    """
    Implementation of a simplified graph convolutional network.
    """

    def __init__(self, num_features, num_classes, num_layers=2, bias=True):
        """
        Class initializer.
        """
        super().__init__()
        self.layer = SGConv(num_features, num_classes, K=num_layers, bias=bias)

    def forward(self, x, edges):
        """
        Run forward propagation.
        """
        return self.layer(x, edges)


class GNN(nn.Module):
    """
    Class that supports general graph neural networks.
    """
    def __init__(self, num_features, num_classes, num_layers=2, hidden_size=16,
                 dropout=0.5, residual=True, conv='gcn'):
        """
        Class initializer.
        """
        super().__init__()
        self.residual = residual
        self.conv = conv
        self.relu = nn.ReLU()
        self.drop_prob = dropout
        self.dropout = nn.Dropout()

        layers = []
        for i in range(num_layers):
            in_size = num_features if i == 0 else hidden_size
            out_size = num_classes if i == num_layers - 1 else hidden_size
            layers.append(self.to_conv(in_size, out_size))
        self.layers = nn.ModuleList(layers)

    def to_conv(self, in_size, out_size, heads=8):
        """
        Make a convolution layer based on the current type.
        """
        if self.conv == 'gcn':
            return GCNConv(in_size, out_size)
        elif self.conv == 'sage':
            return SAGEConv(in_size, out_size)
        elif self.conv == 'gat':
            return GATConv(in_size, out_size // heads, heads=heads, dropout=self.drop_prob)
        else:
            raise ValueError(self.conv)

    def forward(self, x, edges):
        """
        Run forward propagation.
        """
        out = x
        for i, layer in enumerate(self.layers[:-1]):
            out2 = out
            if i > 0:
                out2 = self.dropout(out2)
            out2 = layer(out2, edges)
            out2 = self.relu(out2)
            if i > 0 and self.residual:
                out2 = out2 + out
            out = out2
        out = self.dropout(out)
        return self.layers[-1](out, edges)
