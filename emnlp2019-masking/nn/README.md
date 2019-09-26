
# Fact Verification using Mean Teacher in PyTorch

In this fork of the original mean teacher code, we replace the feed forward networks in a mean teacher setup with 
 a decomposable attention. Also the data input is that from FEVER 2018 shared task.
 
# Pre reqs:
 
 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create --name mean_teacher python=3 numpy scipy pandas nltk tqdm
source activate mean_teacher
pip install sklearn
pip install jsonlines
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119
conda install pytorch-cpu torchvision-cpu -c pytorch *
```


To train on FEVER, run the following command in the folder `pytorch/` :


``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file fever_dev_lex_3labels_100_no_lists_evidence_not_sents.jsonl --train_input_file fever_train_lex_3labels_400_smartner_3labels_no_lists_evidence_not_sents.jsonl --arch da_RTE --log_level INFO --use_gpu false --pretrained_wordemb_file data-local/glove/glove.840B.300d.txt --use_double_optimizers true --run_student_only true --labels 20.0 --consistency 1

```

## Notes
- for pytorch instinstallation get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

- Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).
 
 - The glove file kept at `data-local/glove/glove.840B.300d.txt` is a very small version of the actual glove file. You might want to replace it with the actual 840B [glove file](https://nlp.stanford.edu/projects/glove/)

 - I personally like/trust `pip install ` instead of `conda install`  because the repos of pip are more comprehensive

 - The code expects to find the data in specific directories inside the data-local directory.  For example some sample training and dev is kept here:

```
pytorch/data-local/rte/fever/
```
You will have to also get the actual [train](https://drive.google.com/open?id=1bA32_zRn8V2voPmb1sN5YbLcVFo6KBWf) and [dev](https://drive.google.com/open?id=1xb6QHfMQUI3Q44DQZNVL481rYyMGN-sR) files from google drive


# explanation of command line parameters

`--workers`: if you dont want multiprocessing make workers=0

`--run_student_only false` means : Run the code as both teacher and student. Difference here is 
that when running both, first the dataset is divided into labeled and unlabeled based on the percentage you mentioned in `--labels`.
now instead if you just want to work with labeled data: i.e supervised training. i.e you don't want to run mean teacher for some reason: then you turn this on/true.

 If you are doing `--run_student_only true` i.e to run as only a student (which might internally have say a simple feed forward supervised network)
  with all data points having labels, you shouldn't pass any of these argument parameters which are meant for mean teacher.

```
--labeled_batch_size
--labels
--consistency
```

#### reason: 


`--labels`: is the percentage or number of labels indicating the number of labeled data points amongst the entire training data. If its int, the code assumes that you are
passing the actual number of data points you want to be labeled. Else
if its float, the code assumes it is a percentage value.

`--labeled_batch_size` : say 20% (`--labels`) of the full data you are dropping the labels for/marking
as unlabeled. You still have a batch size (as specified in `----batch_size`). `--labeled_batch_size` is an int value
which says how many of the data points within a batch do you want to be marked as unlabeled.
Eg: Say there are 200 data points in your training batch and your command line looks like this:
`--labels 20.0 ----batch_size 32 --labeled_batch_size 10`. This means that, 20% of the labels (i.e 40 data points) overall will be marked as
unlabeled. Now all these 400 data points will be divided into approximately 12 batches with each batch containing 32 data points. Within this
32, atleast 10 of them will be marked as unlabeled, and the rest 22 will be left as is, i.e labeled.

Note: If you want to run both student and teacher, but without dropping any labels, you must not have any of these:
```
--labeled_batch_size
--labels
--consistency
```

i.e you should use only `----batch_size` from the 3 above . ALso you must make `--run_student_only false`

so the total command will look like
```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6 --run-name fever_transform --batch_size 10 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file dev_with_50_evi_sents.jsonl --train_input_file train_with_100_evi_sents.jsonl --arch da_RTE --log_level INFO --use_gpu false --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --run_student_only false --labeled_batch_size 10
```

Further details of other command line parameters can be found in `pytorch/mean_teacher/tests/cli.py`



- also, make sure the value of `--labels` is removed.

- also note that due to the below code whenever `run_student_only=true`, 
the sampler becomes a simple `BatchSampler`, and not a `TwoStreamBatchSampler` (which is
the name of the sampler used in mean teacher).

```

if args.run_student_only:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

```

# Testing
To do testing (on dev or test partition), you need to run the code again with `--evaluate` set to `true`. i.e training and testing uses same code but are mutually exclusive. You cannot run testing immediately after training.
You need to finish training and use the saved model to do testing.

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.

Note to anyone testing from clulab (including myself, mithun). Run on 
server:clara.
- cd meanteacher
- tmux
- git pull
- source activate meanteacher
- run one of the linux commands given [here](#commands_to_run)   

Note:
 
- the commands must be run at the folder level which has main.py eg:`mean-teacher/pytorch/main.py`
- from the same level you can keep track of the logs by doing `tail -f meanteacher.log`
- there will be a time delay of around 3 minutes while the vocabulary is created. The log file will have an entry just before that saying : `INFO:datasets:Done loading embeddings. going to create vocabulary ...`
 
