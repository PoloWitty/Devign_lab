# Devign

> HUST CS: GNN 2023 Fall

This repo is cloned from [this repo](https://github.com/gystar/devign_lab/tree/main) and modified.

## Setup
```bash
bash setup.sh
```
For more detail, please have a look at `setup.sh`.

## Result

Experimental environment is based on pytorch1.13.0 using one piece of RTX2080Ti GPU card. During the experiments, *dataset ratio* was set to 1 to use the full dataset for training and testing. The results are shown in the table below. Both GCN and GAT are built in the form of a layer of GNN networks followed by a layer of ReLU activation functions. The GGNN (Gated Graph Neural Networks) was built using only one layer of GatedGraphConv. Conv readout is the same as those used in [Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks]. The MLP sum is the initial features and the node embedding obtained from the GNN network follows the two layers plus the linear layer of the activation function, after which it is summed and activated.


Table1: The result of experiments. I tested 3 kinds of model (GCN, GAT, and GGNN) and 2 types of readout function (MLP sum and conv).

|              | F1   | Acc  |
|--------------|------|------|
| GCN+conv     | 0.21 | 0.45 |
| GCN+MLP sum  | 0.38 | 0.53 |
| GAT+conv     | 0.068| 0.50 |
| GAT+MLP sum  | 0.49 | 0.49 |
| GGNN+MLP sum | 0.44 | 0.55 |
| GGNN+conv    | 0.22 | 0.52 |
