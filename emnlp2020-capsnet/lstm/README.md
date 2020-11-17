# How to train and test the lstm

## Hardware info:

- CPUs: 48 CPUs (24 physical cores with 2 threads per core)
- Memory: 264GB of RAM
- GPUs: 1 NVIDIA Tesla k20m GPU (5120MB of memory)

## To train:
Run `sbt 'runMain org.clulab.dynet.Metal -train LANGUAGE-TRAIN_AMOUNT-learn-seedX -conf org/clulab/mtl-LANGUAGE-pos.conf'`

- Replace `LANGUAGE` with `es` or `gd`
- Replace `TRAIN_AMOUNT` with `100`, `50`, `10`, or `1`
- Replace `X` with the random seed you want

## To test:

Run `sbt 'runMain org.clulab.dynet.Metal -test LANGUAGE-TRAIN_AMOUNT-learn-seedX-epochY -conf org/clulab/mtl-LANGUAGE-pos.conf'`

- Replace `LANGUAGE` with `es` or `gd`
- Replace `TRAIN_AMOUNT` with `100`, `50`, `10`, or `1`
- Replace `X` with the random seed you want
- Replace `Y` with the best epoch from training

## notate bene

The LSTM was trained using a specific version of the CLU Lab's [processors](https://github.com/clulab/processors) repo. The current public version is NOT the same!

A tag to the appropriate version will be posted here.

The `processors` code base refers to GloVe vectors. However, our project uses FastText vectors. You can safely disregard any references to GloVe.
