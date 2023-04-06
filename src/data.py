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
import os
from argparse import Namespace
from collections import defaultdict

import pickle as pkl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor


def is_large(data):
    """
    Return whether a dataset is large or not.
    """
    return data == 'arxiv'


def is_continuous(data):
    """
    Return whether a dataset has continuous features or not.
    """
    return data in ['pubmed', 'coauthor', 'arxiv']


def to_edge_tensor(edge_index):
    """
    Convert an edge index tensor to a SparseTensor.
    """
    row, col = edge_index
    value = torch.ones(edge_index.size(1))
    return SparseTensor(row=row, col=col, value=value)


def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError()

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError()
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError()


def load_steam(root):
    """
    Load the Steam dataset with manual preprocessing.
    """
    freq_item_mat = pkl.load(open(os.path.join(root, 'Steam', 'processed', 'freq_item_mat.pkl'), 'rb'))
    features = pkl.load(open(os.path.join(root, 'Steam', 'processed', 'sp_fts.pkl'), 'rb'))
    features = torch.from_numpy(features.todense()).float()
    labels = torch.zeros(features.size(0), dtype=int)

    adj = freq_item_mat.copy()
    adj[adj < 10.0] = 0.0
    adj[adj >= 10.0] = 1.0
    indices = np.where(adj != 0.0)
    rows = indices[0]
    cols = indices[1]
    edge_index = torch.from_numpy(np.stack([rows, cols], axis=0))
    return Namespace(data=Namespace(x=features, y=labels, edge_index=edge_index))


def load_arxiv(root):
    """
    Load the Arxiv dataset, which is not included in PyG.
    """
    features = torch.from_numpy(np.loadtxt(f'{root}/ArXiv/raw/node-feat.csv.gz', delimiter=',', dtype=np.float32))
    labels = torch.from_numpy(np.loadtxt(f'{root}/ArXiv/raw/node-label.csv.gz', delimiter=',', dtype=np.int64))
    edge_index = torch.from_numpy(np.loadtxt(f'{root}/ArXiv/raw/edge.csv.gz', delimiter=',', dtype=np.int64))
    edge_index = to_undirected(edge_index.t())
    return Namespace(data=Namespace(x=features, y=labels, edge_index=edge_index))


def load_data(dataset, split=None, seed=None, verbose=False, normalize=False,
              validate=False):
    """
    Load a dataset from its name.
    """
    root = '../data'
    if dataset == 'cora':
        data = datasets.Planetoid(root, 'Cora')
    elif dataset == 'citeseer':
        data = datasets.Planetoid(root, 'CiteSeer')
    elif dataset == 'computers':
        data = datasets.Amazon(root, 'Computers')
    elif dataset == 'photo':
        data = datasets.Amazon(root, 'Photo')
    elif dataset == 'steam':
        data = load_steam(root)
    elif dataset == 'pubmed':
        data = datasets.Planetoid(root, 'PubMed')
    elif dataset == 'coauthor':
        data = datasets.Coauthor(root, 'CS')
    elif dataset == 'arxiv':
        data = load_arxiv(root)
    else:
        raise ValueError(dataset)

    node_x = data.data.x
    node_y = data.data.y
    edges = data.data.edge_index

    if validate:
        validate_edges(edges)

    if normalize:
        assert (node_x < 0).sum() == 0  # all positive features
        norm_x = node_x.clone()
        norm_x[norm_x.sum(dim=1) == 0] = 1
        norm_x = norm_x / norm_x.sum(dim=1, keepdim=True)
        node_x = norm_x

    if split is None:
        if hasattr(data.data, 'train_mask'):
            trn_mask = data.data.train_mask
            val_mask = data.data.val_mask
            trn_nodes = torch.nonzero(trn_mask).view(-1)
            val_nodes = torch.nonzero(val_mask).view(-1)
            test_nodes = torch.nonzero(~(trn_mask | val_mask)).view(-1)
        else:
            trn_nodes, val_nodes, test_nodes = None, None, None
    elif len(split) == 3 and sum(split) == 1:
        trn_size, val_size, test_size = split
        indices = np.arange(node_x.shape[0])
        trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed,
                                                 stratify=node_y)
        trn_nodes, val_nodes = train_test_split(trn_nodes, test_size=val_size / (trn_size + val_size),
                                                random_state=seed, stratify=node_y[trn_nodes])

        trn_nodes = torch.from_numpy(trn_nodes)
        val_nodes = torch.from_numpy(val_nodes)
        test_nodes = torch.from_numpy(test_nodes)
    else:
        raise ValueError(split)

    if verbose:
        print('Data:', dataset)
        print('Number of nodes:', node_x.size(0))
        print('Number of edges:', edges.size(1) // 2)
        print('Number of features:', node_x.size(1))
        print('Ratio of nonzero features:', (node_x > 0).float().mean().item())
        print('Number of classes:', node_y.max().item() + 1 if node_y is not None else 0)
        print()
    return edges, node_x, node_y, trn_nodes, val_nodes, test_nodes


def main():
    """
    Main function.
    """
    for data in ['cora', 'citeseer', 'photo', 'computers', 'steam', 'pubmed',
                 'coauthor']:
        load_data(data, split=(0.4, 0.1, 0.5), validate=True, verbose=True)


if __name__ == '__main__':
    main()
