# Structured Variational Graph Autoencoder

This project is a PyTorch implementation of *Accurate Node Feature Estimation
with Structured Variational Graph Autoencoder* (KDD 2022).

## Prerequisites

Our implementation is based on Python 3.8 and PyTorch Geometric. Please see the
full list of packages required to run  our codes in `requirements.txt`.

- Python 3.8
- PyTorch 1.4.0
- PyTorch Geometric 1.6.3

PyTorch Geometric requires a separate installation process from the other
packages. We included `install.sh` to guide the installation process of PyTorch
Geometric based on the OS and CUDA version. The code includes the cases for
`Linux + CUDA 10.0`, `Linux + CUDA 10.1`, and `MacOS + CPU`.

## Datasets

We use 8 datasets in our work: Cora, Citeseer, Amazon-Computers, Amazon-Photo,
Steam, Pubmed, Coauthor, and Arxiv. All datasets except Steam are downloaded
automatically by PyTorch Geometric when the code is run. The Steam dataset can
be found at https://github.com/xuChenSJTU/SAT-master-online, which includes the
raw dataset and a preprocessing code. Please run the preprocessing code and put
the result in `data/Steam/processed`.

## Training

We included `main.sh`, which reproduces the experiment of our paper for feature
estimation. The code automatically downloads the Cora dataset and trains SVGA.
The training process is printed as follows, where the last line corresponds to
the best epoch with the highest validation recall (or RMSE if the dataset has
continuous features).

```
-----------------------------------------------
epoch x_loss y_loss r_loss trn    val    test
    0 0.6938 0.0000 0.1568 0.0056 0.0063 0.0051
  100 0.3145 0.0000 0.0352 0.1912 0.1468 0.1579
  200 0.2248 0.0000 0.0384 0.2445 0.1614 0.1690
  300 0.1864 0.0000 0.0409 0.2975 0.1618 0.1753
  400 0.1637 0.0000 0.0398 0.3316 0.1632 0.1763
  500 0.1542 0.0000 0.0351 0.3642 0.1624 0.1772
  600 0.1424 0.0000 0.0346 0.3816 0.1645 0.1771
-----------------------------------------------
  420 0.1631 0.0000 0.0389 0.3385 0.1689 0.1767
-----------------------------------------------
```

Then, the performance of the best model is printed as follows, where the values
represent the recalls and nDCGs with k in 10, 20, and 50. The printed values
become R^2 and RMSE if the dataset has continuous features. 

```
{
    "epoch": 420,
    "test_res": [
        0.17666353285312653,
        0.2577683627605438,
        0.387383371591568,
        0.24040722306262607,
        0.2938389755350002,
        0.3626111867471361
    ],
    "val_res": [
        0.16893510520458221,
        0.24577094614505768,
        0.38291221857070923,
        0.23090830913822932,
        0.2824937344217453,
        0.3558690611104378
    ]
}
```

## Evaluation

The evaluation for feature estimation is done during the training process of a
model. On the other hand, the evaluation for node classification, based on the
generated features, is done by the scripts of a previous work, which are
included in https://github.com/xuChenSJTU/SAT-master-online. Specifically,
`eva_classification_X.py` and `eva_classfication_AX.py` in their repository
evaluate the generated features using an MLP and a GCN classifier, respectively.
The only change from their scripts is that we use an Adam optimizer with
learning rate `1e-3` and no weight decay for the datasets of discrete features,
which is used also in the training of generator models, while the original
scripts use different optimizers for the two classifiers.
