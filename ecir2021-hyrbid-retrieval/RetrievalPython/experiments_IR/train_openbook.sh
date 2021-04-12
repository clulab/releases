#!/bin/bash

python train_eval_openbook_squad_nq.py --dataset=openbook --n_neg_sample=10 --seed 0
python train_eval_openbook_squad_nq.py --dataset=openbook --n_neg_sample=10 --seed 1
python train_eval_openbook_squad_nq.py --dataset=openbook --n_neg_sample=10 --seed 2
python train_eval_openbook_squad_nq.py --dataset=openbook --n_neg_sample=10 --seed 3
python train_eval_openbook_squad_nq.py --dataset=openbook --n_neg_sample=10 --seed 4