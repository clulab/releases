# Parsing as Tagging

### Environment set-up
```
conda create -n pat python=3
conda activate pat
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch
pip install pytorch-pretrained-bert
pip install networkx
```

### Data
We used Universal Dependencies 2.2.

For each dataset, we filtered non-integer id and deleted the comments.

### Architecture
![Architecture](architecture.png)

For each word, we predict 
* relative position of the head
* label

We use a BiLSTM that operates over our token representation (BERT (without fine-tuning) + word Embedding + char-level embedding + part-of-speech embedding). The resulted hidden state is then passed into an MLP. The result is then used to predict the head. To predict the label, we concatenate the MLP outputs of the current token and its predicted head.

### Hyperparameters

The parameters used for our 'un-tuned' version. This set of parameters were selected out of 100 possible configurations.
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

The parameters used for our 'tuned' version. For each language, we selected the best performing parameters from a search space of 12 configurations. Same search space was used for all languages.

|         | Learning rate | Dropout | MLP Hidden Layers |
|---------|---------------|---------|-------------------|
| ar      | 0.0025        | 0.50    | 400, 150          |
| bu      | 0.0025        | 0.50    | 400, 150          |
| ca      | 0.0025        | 0.50    | 400, 150          |
| cs      | 0.0020        | 0.50    | 500, 150          |
| de      | 0.0020        | 0.55    | 500, 150          |
| en      | 0.0020        | 0.60    | 500, 150          |
| en (SD) | 0.0020        | 0.55    | 500, 150          |
| es      | 0.0020        | 0.50    | 500, 150          |
| et      | 0.0020        | 0.50    | 500, 150          |
| fr      | 0.0020        | 0.60    | 500, 150          |
| it      | 0.0020        | 0.55    | 500, 150          |
| ja      | 0.0025        | 0.50    | 400, 150          |
| nl      | 0.0025        | 0.50    | 400, 150          |
| no      | 0.0020        | 0.55    | 500, 150          |
| ro      | 0.0025        | 0.50    | 400, 150          |
| ru      | 0.0020        | 0.50    | 500, 150          |

The parameters not appearing in the table are kept the same as in 'un-tuned' version

Random seeds used:

UD
```
1 2449 9959019
```
SD
```
1 2449 4973 3499854 9959019
```
