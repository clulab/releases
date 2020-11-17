# How to train/test CapsNet and CNN models

This readme describes how to train and test the CapsNet and CNN models. To train/test the LSTM model, go up one directory and enter the `lstm/` directory.

## Hardware used:
- CPU: 1 CPU
- Memory: 16GB of RAM
- GPUS:  none

## To train:
Run `ipython train_caps_learn.py LANGUAGE TRAIN_AMOUNT RANDOM_SEED`

- Replace `LANGUAGE` with `es` or `gd`
- Replace `TRAIN_AMOUNT` with `100`, `50`, `10`, or `1`
- Replace `RANDOM_SEED` with the random seed you want

Do the same for `train_caps_nolearn.py`, `train_cnn_nolearn.py`, and `train_cnn_learn.py`.

To test:
Run `model_testing.ipynb` in jupyter notebook, change the `LANGUAGE` and `TRAIN AMOUNT` to the values you wish to test.

## notate bene

Much of the original code base refers to GloVe vectors. However, our project uses FastText vectors.

This repo does NOT include the FastText vectors! You will need to place them in the appropriate directory in `code/data/` (NOT `./data`).
