#!/bin/bash

python train_eval_openbook_squad_nq.py --dataset=squad --n_epoch 2 --seed 1
python train_eval_openbook_squad_nq.py --dataset=squad --n_epoch 2 --seed 2
python train_eval_openbook_squad_nq.py --dataset=squad --n_epoch 2 --seed 3
python train_eval_openbook_squad_nq.py --dataset=squad --n_epoch 2 --seed 4