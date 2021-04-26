# Interpretability Rules: Jointly Bootstrapping a Neural Relation Extractor with an Explanation Decoder: 

This is the accompanying code and data for our NAACL TrustNLP 2020 [paper](http://clulab.cs.arizona.edu/papers/trustNLP2021_edin.pdf) 

## Requirements

- Python 3 (tested on 3.6.5)
- PyTorch (tested on 0.4.0)

## Preparation

The code requires that you have access to the TACRED dataset (LDC license required). The TACRED dataset is currently scheduled for public release via LDC in December 2018. For possible early access to this data please contact us at `yuhao.zhang ~at~ stanford.edu`. Once you have the TACRED data, please put the JSON files under the directory `dataset/tacred`.

First, download and unzip GloVe vectors from the Stanford NLP group website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training

To train a graph convolutional neural network (GCN) model, run:
```
bash python train.py --id 0 --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003
```

Model checkpoints and logs will be saved to `./saved_models/00`.


For details on the use of other parameters, such as the pruning distance k, please refer to `train.py`.

## Evaluation

To run evaluation on the test set, run:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

## Retrain

Reload a pretrained model and finetune it, run:
```
python train.py --load --model_file saved_models/01/best_model.pt --optim sgd --lr 0.001
```

## Related Repo

The paper also includes comparisons to the GCN over pruned tree model for relation extraction. To reproduce the corresponding results, please refer to [this repo](https://github.com/qipeng/gcn-over-pruned-trees).

## Citation