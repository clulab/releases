#!/bin/bash

save_best_in_path=$1

# mkdir ${save_best_in_path}/2_128
# mkdir ${save_best_in_path}/2_256 
# mkdir ${save_best_in_path}/4_256 # bert mini
# mkdir ${save_best_in_path}/4_512 # bert small
# mkdir ${save_best_in_path}/8_512 # bert medium
mkdir ${save_best_in_path}/bert_base_uncased # bert base

# model1="google/bert_uncased_L-2_H-128_A-2"
# model2="/data/nlp/corpora/bert-models/uncased_L-2_H-256_A-4"
# model3="google/bert_uncased_L-4_H-256_A-4"
# model4="google/bert_uncased_L-4_H-512_A-8"
# model5="google/bert_uncased_L-8_H-512_A-8"
model6="google/bert_uncased_L-12_H-768_A-12"


# python train.py --config-file configs/config1.json --model-name $model1 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in ${save_best_in_path}/2_128/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model1 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/2_128/last2.ckpt --load-from-checkpoint ${save_best_in_path}/2_128/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model1 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/2_128/last3.ckpt --load-from-checkpoint ${save_best_in_path}/2_128/last2.ckpt

# python train.py --config-file configs/config1.json --model-name $model2 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in ${save_best_in_path}/2_256/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model2 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/2_256/last2.ckpt --load-from-checkpoint ${save_best_in_path}/2_256/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model2 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/2_256/last3.ckpt --load-from-checkpoint ${save_best_in_path}/2_256/last2.ckpt
 
# python train.py --config-file configs/config1.json --model-name $model3 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in ${save_best_in_path}/4_256/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model3 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/4_256/last2.ckpt --load-from-checkpoint ${save_best_in_path}/4_256/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model3 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/4_256/last3.ckpt --load-from-checkpoint ${save_best_in_path}/4_256/last2.ckpt

# python train.py --config-file configs/config1.json --model-name $model4 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in ${save_best_in_path}/4_512/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model4 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/4_512/last2.ckpt --load-from-checkpoint ${save_best_in_path}/4_512/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model4 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/4_512/last3.ckpt --load-from-checkpoint ${save_best_in_path}/4_512/last2.ckpt

# python train.py --config-file configs/config1.json --model-name $model5 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 256 -agb 1 --val-batch-size 256 --save-best-in ${save_best_in_path}/8_512/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model5 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/8_512/last2.ckpt --load-from-checkpoint ${save_best_in_path}/8_512/last1.ckpt
# python train.py --config-file configs/config1.json --model-name $model5 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/8_512/last3.ckpt --load-from-checkpoint ${save_best_in_path}/8_512/last2.ckpt

python train.py --config-file configs/config1.json --model-name $model6 --max-epochs 3 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset1' -nsvs 1000 --checkpoint-prefix 's1_1' --train-batch-size 128 -agb 2 --val-batch-size 256 --save-best-in ${save_best_in_path}/bert_base_uncased/last1.ckpt
python train.py --config-file configs/config1.json --model-name $model6 --max-epochs 2 --load-from-arrow-dir '/data/nlp/corpora/odinsynth/data/rules100k_arrow/dataset2' -nsvs 1000 --checkpoint-prefix 's1_2' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/bert_base_uncased/last2.ckpt --load-from-checkpoint ${save_best_in_path}/bert_base_uncased/last1.ckpt
python train.py --config-file configs/config1.json --model-name $model6 --max-epochs 1 --load-from-arrow-dir '/home/rvacareanu/temp/huggingface_datasets_100k_complete'  -nsvs 1000 --checkpoint-prefix 's1_3' --train-batch-size 32  -agb 8 --val-batch-size 64  --save-best-in ${save_best_in_path}/bert_base_uncased/last3.ckpt --load-from-checkpoint ${save_best_in_path}/bert_base_uncased/last2.ckpt
