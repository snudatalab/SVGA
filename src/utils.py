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
import argparse

import torch


def to_device(gpu):
    """
    Return a PyTorch device from a GPU index.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def str2bool(v):
    """
    Convert a string variable into a bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()
