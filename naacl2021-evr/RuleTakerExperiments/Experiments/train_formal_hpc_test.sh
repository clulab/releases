#!/bin/bash

# This is for exp 1
python 2_TrainAndSave.py 3nn f 70k 5 3 &
python 2_TrainAndSave.py 3nn r 70k 5 3 &
python 2_TrainAndSave.py 3nn c 70k 5 3

# This the multitask setting
python 2_TrainAndSave.py 1nn s 70k 5 3 &

# This is the large buffer size setting.
python 2_TrainAndSave.py 3nn f 70k 20 10
python 2_TrainAndSave.py 3nn r 70k 20 10 &
python 2_TrainAndSave.py 3nn c 70k 20 10

# This is the 10k training setting
python 2_TrainAndSave.py 3nn f 10k 5 3 &
python 2_TrainAndSave.py 3nn r 10k 5 3 &
python 2_TrainAndSave.py 3nn c 10k 5 3

# This is the 30k training setting
python 2_TrainAndSave.py 3nn f 30k 5 3 &
python 2_TrainAndSave.py 3nn r 30k 5 3 &
python 2_TrainAndSave.py 3nn c 30k 5 3
