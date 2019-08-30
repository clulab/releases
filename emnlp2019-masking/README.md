
# UOFA- Fact Extraction and VERification


## Steps to install + run
- install docker
- start your docker daemon
- docker pull myedibleenso/processors-server:latest
- docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server
- conda create --name fever python=3.7
- conda activate fever
- conda install pytorch torchvision -c pytorch (might be different for your machine)
- pip install -r requirements.txt
- PYTHONPATH=src python run_fact_verif.py -p config/fever_nn_ora_sent_updateEmbeds.json

## Extra info
This code is built on top of sheffield's fever baseline. So we assume you have installed all the required documents they mention in their [readme file](https://github.com/sheffieldnlp/fever-baselines)

Apart from that you will need PyProcessors over Docker. After you have installed [Docker](https://www.docker.com/), do:


- `docker pull myedibleenso/processors-server:latest`

- `docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`

note: the docker run command is for the very first time you create this container. 

Second time onwards use: `docker start procserv`


## In our project we are experimenting with fact verification but unlexicalized.
## As of Nov 2018 there are two main lines of develpment
1. With decomposable attention + hand crafted features of NER replacement
2. With handcrafted features + SVM

# All code can be run with the below command (but with different config file values)
`PYTHONPATH=src python run_fact_verif.py -p config/fever_nn_ora_sent_updateEmbeds.json`

# details of config file entries
- The name/path of the config file file is: config/fever_nn_ora_sent_updateEmbeds.json
- Almost always/mostly you will have to change only two entries `datasets_to_work_on` and `list_of_runs`
- With `datasets_to_work_on` you tell the machine which all data sets you want to work on. 
- Combined with `list_of_runs` it is an indication of what kind of process to run on what type of data set.
Eg: 
`
"datasets_to_work_on": [
"fnc"
],
`

- other options for value of datasets_to_work_on include: fever. Can add more than one like this: "fnc,fever" 

-

 `
"list_of_runs": [
"dev"
],
`       
- here other options include: "test/train/dev". 
- Can add more than one of these options together comma separated like this: "dev,test". 
- Note that it means, corresponding data set and corresponding runs. i.e if you add "fever,fnc" in datasets and "train,dev" on runs, it means, it will run train on fever and dev on fnc.
- If you had a training run, the trained model is usually stored in the path mentioned in `serialization_dir` in config file. Usually its timestamped inside the `logs` folder Eg: `logs/fever_train_100/model.tar.gz`.
- Whenever you run dev or test (with or without train), make sure you copy the trained model to the `data/models/` folder. Also paste the name of the model file (`*.tar.gz`) to the variable `name_of_trained_model_to_use` .
- note to self: In server the trained models are stored at: `mithunpaul@jenny:/data1/home/mithun/fever_fnc_all_pkl_json_files/fever/training/pickles`

### Example:

So if you want to run just dev on fake news data set this is how your config will look like:
```
"datasets_to_work_on": [
       "fnc"
     ],
     "list_of_runs": [
       "dev"
     ],
```

### Example:

Instead if you want to first train the code on fever and then test on fnc,  your config will look like:
```
"datasets_to_work_on": [
       "fever",
       "fever"
     ],
     "list_of_runs": [
       "train",
       "dev"
     ],
```

## Features

everything above assumes that features are going to be generated on the fly Eg: create smartner tags.
These can be specified as a list inside:  

```
"features": [
       "fully_lexicalized",
       "merge_smartner_supersense_tagging"
     ],
``` 

## Load input files with features
 
 Instead if you have the neutered data written to disk and you just want to load it before training do:
 
`create_features =false`

## Logging
Logs are written into `general_log.txt` in the home directory.
A home directory is the one within fever-baselines folder Eg:`/work/mithunpaul/fever/dev_branch/fever-baselines/`
So to keep an eye on the run you can do `tail -f general_log.txt`

## Annotation 
means it will take a given data dump (eg:dev) and annotate it with `pyprocessors` to create lemmas, pos tags etc. This is inturn provided as input to the actual training code. Refer example below.
##### Example:
If you want to annotate the fever data set with pyprocessors your config will look like:

```
do_annotation=true

"datasets_to_work_on": [
       "fever"
     ],
     "list_of_runs": [
       "test"
     ],

```
Note: the input file to annotate must be in the fever-data folder
For example:
`/work/mithunpaul/fever/dev_branch/fever-baselines/data/fever-data/test.json` 
The annotated file will be written to the path specified by the config file variable `path_to_pyproc_annotated_data_folder`
### Example:

If you are developing code and don't want to train on the whole training dataset of fever
 which is 145k entries you can use small:
```
"datasets_to_work_on": [
       "small"
     ],
     "list_of_runs": [
       "annotation"
     ],

```

## version tracker is kept [here](https://github.com/mithunpaul08/fever-baselines/blob/master/versions.md)

