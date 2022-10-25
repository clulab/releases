It Takes Two Flints to Make a Fire: Multitask Learning of Neural Relation and Explanation Classifiers
=========================

This repo contains the *PyTorch* code for paper [It Takes Two Flints to Make a Fire: Multitask Learning of Neural Relation and Explanation Classifiers](https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00463/113094/It-Takes-Two-Flints-to-Make-a-Fire-Multitask).

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).

## Requirements

- Python 3 (tested on 3.10.4)
- PyTorch (tested on 1.11.0)
- Huggingface Transformers (tested on 4.20.0)

## Preparation

Download the TACRED dataset and put the json files in `./dataset/tacred`.

## Training

Train model for TACRED:
```
python train.py --lr 1e-5 --pooling avg --batch_size 32 --num_epoch 20 --seed 43 --id 0 --device 0 --warmup_prop 0.3 --data_dir dataset/tacred --info "the training info" 
```

Model checkpoints and logs will be saved to `./saved_models/00`.

## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --device 0 --dataset train
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.
