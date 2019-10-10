
# Fact Verification using Mean Teacher in PyTorch

Here we will take the output of the tagging process and feed it as input to a [decomposable attention](https://arxiv.org/pdf/1606.01933.pdf) based neural network model.

 

# Pre reqs:
 
 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create --name rte python=3 numpy scipy pandas nltk tqdm
source activate rte
pip install sklearn
pip install jsonlines
mkdir outputs
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119 *
conda install pytorch-cpu torchvision-cpu -c pytorch *

```

*: These are for installing pytorch. You should get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

To train on FEVER, run the following command in the folder `pytorch/` :


``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file fever_dev_lex_3labels_100_no_lists_evidence_not_sents.jsonl --train_input_file fever_train_lex_3labels_400_smartner_3labels_no_lists_evidence_not_sents.jsonl --arch da_RTE --log_level INFO --use_gpu false --pretrained_wordemb_file data-local/glove/glove.840B.300d.txt --use_double_optimizers true --run_student_only true --labels 20.0 --consistency 1

```

## Notes


- Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).
 
 - The glove file kept at `data-local/glove/glove.840B.300d.txt` is a very small version of the actual glove file. You might want to replace it with the actual 840B [glove file](https://nlp.stanford.edu/projects/glove/)

 - I personally like/trust `pip install ` instead of `conda install`  because the repos of pip are more comprehensive

 - The code expects to find the data in specific directories inside the data-local directory.  For example some sample training and dev is kept here: `pytorch/data-local/rte/fever/`. Also you will have to replace the sample data with the the actual [train](https://drive.google.com/open?id=1bA32_zRn8V2voPmb1sN5YbLcVFo6KBWf) and [dev](https://drive.google.com/open?id=1xb6QHfMQUI3Q44DQZNVL481rYyMGN-sR) files from google drive


 - The code for decomposable attention is forked from [here](https://github.com/libowen2121/SNLI-decomposable-attention)
