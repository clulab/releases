# Parsing as Tagging

### Environment set-up
```
conda create -n pat python=3
conda activate pat
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch
pip install pytorch-pretrained-bert
pip install networkx
```

### Hyperparameters

| Parameter                              | Value     |
|----------------------------------------|-----------|
| Early stopping                         | 3         |
| Batch size                             | 64        |
| CNN kernel size                        | 3         |
| CNN embedding size                     | 50        |
| CNN output size                        | 50        |
| Learning rate                          | 0.002     |
| &beta;<sub>1</sub>, &beta;<sub>2</sub> | 0.7, 0.99 |
| Dropout                                | 0.6       |
| Weight decay                           | 1e-5      |
| BiLSTM layers                          | 3         |
| BiLSTM hidden-size                     | 600       |
| BiLSTM dropout                         | 0.3       |
| MLP Hidden Layers                      | 500, 150  |

Random seeds used:

UD
```
1 2449 9959019
```
SD
```
1 2449 4973 3499854 9959019
```
