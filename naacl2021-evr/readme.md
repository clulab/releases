This repo contains the code of the NAACL 2021 paper: Explainable Multi-hop Verbal Reasoning Through Internal Monologue.

# Requirements:
`python = 3.7`

`numpy = 1.19.1`

`torch = 1.6.0`

`transformers = 3.0.2`

`editdistance`

The installation of `transformers` package can be found [here](https://huggingface.co/transformers/installation.html). The installation of `editdistance` can be found [here](https://github.com/roy-ht/editdistance).

# Before Running:
Please download the original RuleTaker dataset from [here](https://allenai.org/data/ruletaker) and put it in the Data folder of this project. 

# Run Experiments: 

## Train the Model
To train the neural module, run the following commands: 

`cd RuleTakerExperiments/Experiments/`

`python 2_TrainAndSave 3nn c 70k 5 3 1`

`python 2_TrainAndSave 3nn f 70k 5 3 1`

`python 2_TrainAndSave 3nn r 70k 5 3 1`

The commands above will train the EVR1 model as introduced in the paper. 
+ 3nn means to use 3 T5s to learn all patterns. 
+ "c" means "controller", it learns all patterns except pattern 6 and 10; 
+ "f" means facts handler, it learns pattern 6; 
+ "r" means rules handler, it learns pattern 10. 
+ "70k" means using all training data. 
+ "5" means the fact buffer size is 5; 
+ "3" means the rule buffer size is "3". 
+ "1" means to use DU1 data for training.

The random seed is hard coded to 0 in the script. 

After training, it will create a folder "saved_models" and create a subfolder to store the trained T5s. 

## Test the Model:
`cd RuleTakerExperiments/Experiments/`

`python 4_T5Small_Chaining_FormalResults 1 0 0 3nn 70k 5 3 c f r`

This script will load the trained T5s and evaluate on the RuleTaker test sets. 
+ "1" (the first argument) means the experiment ID, it can be any number. 
+ "0" (the second argument) means to use the DU5 dataset. If it is set to "1", it is to use the Birds-Electricity dataset for evaluation. 
+ "0" (the third argument) means to evaluate on depth 0. If DU5 data is used, this number can be [0,1,2,3,4,5]; If Birds-Electricity is used, this number can be [0,1,2,3,4]. 
+ "3nn" this is to match the trained model. 
+ "70k" training amount, this should match trained model.
+ "5" this is the fact buffer size (should match the trained model). 
+ "3" the rule buffer size (should matched the trained model). 
+ "c", "f", "r" should match the trained models. 

After testing, it should save the results in the subfolder in the "saved_mdoels" folder. 


