#!/bin/bash

model_name=$1
save_best_in_path=$2

python train.py --config-file configs/config1.json --model-name $model_name --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in $save_best_in_path'/last1.ckpt'
python train.py --config-file configs/config1.json --model-name $model_name --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in $save_best_in_path'/last2.ckpt' --load-from-checkpoint '/data/nlp/corpora/odinsynth/models/stagewise/last1.ckpt'
python train.py --config-file configs/config1.json --model-name $model_name --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in $save_best_in_path'/last3.ckpt' --load-from-checkpoint '/data/nlp/corpora/odinsynth/models/stagewise/last2.ckpt'
