
# Cross Domain Fact Verification: Training and Testing

Here we will take the output of the tagging process and feed it as input to a [decomposable attention](https://arxiv.org/pdf/1606.01933.pdf) based neural network model.
This code is based on the baseline code used in FEVER1.0
shared task.

 

#### Testing:

To test using Allennlp based code:

```
conda create --name rte python=3 
source activate rte
brew install npm
git clone this_repo
cd eval/allennlp-simple-server-visualization/demo/
npm start
``` 
This should open browser and run the GUI in `localhost:3000`.

**Note**: All these commands are for OSX.
 Replace with corresponding packages based on your OS. Eg: replace `brew install` with `pip install`

Now open another terminal and do:
```
conda create --name rte_runner python=3.6
source activate rte_runner
cd allennlp-as-a-library-example/
pip install -r requirements.txt
mkdir -p tests/fixtures/trained_models/
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/FeverModels/Smart_NER/decomposable_attention.tar.gz -O tests/fixtures/trained_models/decomposable_attention.tar.gz
wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_test_delex_oaner_4labels.jsonl -O my_library/predictors/test_files/delexicalized_input_file.jsonl
python -m allennlp.service.server_simple \
  --archive-path tests/fixtures/trained_models/decomposable_attention.tar.gz \
  --predictor drwiki-te\
  --include-package my_library \
  --title "Academic Paper Classifier" \
  --field-name title \
  --field-name paperAbstract
```

**Note:** Python for this conda environment(rte_runner) has to be exactly `python 3.6` to be compatible with the right Allennlp versions.

If every thing runs fine you will see `Model loaded, serving demo on port 8000`. Now go back to the browser window
`localhost:3000` , refresh the page and just click `Run` (don't enter anything in the claim or evidence fields). 
If everything thing runs fine you will see a rotating circle on your right half of the screen. It means
it has started doing predictions on the 
given test file. Once the prediction cycle is completed you will see an output in the browser with attention weights and labels of the final
input data point (takes around 15 mins in a MacBookPro with OSX Mojave, 8GB RAM, Intel core i5). The predicted file is 
stored in the same location as:`predictions.jsonl`

Now enter:

```
python fnc_official_scorer.py 
```

That should give an output that looks like:

```-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    396    |    86     |    469    |    52     |
-------------------------------------------------------------
| disagree  |    101    |    35     |    183    |    33     |
-------------------------------------------------------------
|  discuss  |   1124    |    239    |    943    |    161    |
-------------------------------------------------------------
| unrelated |    681    |    345    |   3171    |   6037    |
-------------------------------------------------------------
Score: 3433.75 out of 6380.5	(53.81631533578873%)
```
### Testing with other models and masking strategies:
To test with a  model (that was trained on the **same** FEVER dataset) that was trained using a different masking strategy 
you have to change the corresponding  path in the wget command to any of the following.

**Trained  models:**
- FullyLexicalized
- NoNER
- SSTagged
- Smart_NER

**Masked files:**
- fn_test_split_fourlabels.jsonl
- fnc_test_smartner_sstags_merged.jsonl

For example to test with a **Trained FEVER model:** that was trained on FEVER dataset which was delexicalized
 using OANer+SSTagged masking, the `wget` will change to:

```
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/FeverModels/SSTagged/decomposable_attention.tar.gz -O tests/fixtures/trained_models/decomposable_attention.tar.gz
```
In this case the test file also has to be the correspondingly delexicalized version, in this case is FNC-OANer+SSTagged
which can be retrieved as:
```
wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_test_smartner_sstags_merged.jsonl -O my_library/predictors/test_files/delexicalized_input_file.jsonl
```


### To test without Allennlp
 
Here is a version of the same code which runs without AllenNLP. To install the dependencies and prepare the datasets 
do the following commands:

```
conda create --name rte python=3 numpy scipy pandas nltk tqdm
source activate rte
pip install sklearn
pip install jsonlines
```
`pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119` **refer note

`conda install pytorch-cpu torchvision-cpu -c pytorch` *refer note

**Note**: for pytorch installations get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.
```
git clone thisrepo.git
```

To download data run these command from the folder `eval_noalnlp/` :




To test using a model that was trained on FEVER lexicalized data, and test on FNC dataset:`. 

```
cd eval_noalnlp
./get_data_lex.sh
./get_glove_small.sh
./get_model_lex.sh
python main.py --run_type test --database_to_test_with fnc 
```

To test using a model trained on FEVER delexicalized data (mentioned as OANER in the paper), and test on FNC dataset, run the following commands from the folder `pytorch/`. 
```
./get_data_delex.sh
./get_glove_small.sh
./get_model_delex.sh
python main.py --run_type test --database_to_test_with fnc 
```


#### Training:

To train on FEVER lexicalized, run the following command in the folder `pytorch/` :

``` 
./get_glove.sh
./get_data_lex.sh
python main.py --run_type train --database_to_train_with fever_lex

```


To train on FEVER delexicalized data (mentioned as OANER in the paper), run the following command in the folder `pytorch/` :

``` 
./get_glove.sh
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl  -O data/rte/fever/train/fever_train_delex_oaner_4labels.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl  -O data/rte/fever/dev/fever_dev_delex_oaner_4labels.jsonl
python main.py --run_type train --database_to_train_with fever_delex

```

##### Notes:
- You can keep track of the training and dev accuracies by doing `tail -f mean_teacher.log` 
- The trained model will be stored under `/model_storage/best_model.pth ` 
- Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).
- The glove file kept at `data/glove/glove.840B.300d.txt` is a very small version of the actual glove file. You might want to replace it with the actual 840B [glove file](https://nlp.stanford.edu/projects/glove/)
- I personally like/trust `pip install ` instead of `conda install`  because the repos of pip are more comprehensive

 

##### Acknolwedgements/code adaptations based on:
- [allennlp](https://github.com/allenai/allennlp)
- [libowen](https://github.com/libowen2121/SNLI-decomposable-attention)
- [recognai](https://github.com/recognai/get_started_with_deep_learning_for_text_with_allennlp)
- [fever_baseline_code](https://github.com/sheffieldnlp/fever-naacl-2018)
- [fnc_baseline code](https://github.com/FakeNewsChallenge/fnc-1)

##### Disclaimer: 
Though we have tried our best, it is possible that there might be bugs or broken links. Please get in touch with **mithunpaul@email.arizona.edu** if you can't get something to run.
