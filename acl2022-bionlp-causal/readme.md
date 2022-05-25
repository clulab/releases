Code and data for paper causal event detection, BioNLP 2022. The current files are for the fine-tuning and distillation. The code and data for LSTM and BERT pre-training will be uploaded soon. 

# Requirements
`numpy==1.19.1` \
`torch==1.6.0` \
`sklearn==0.23.2` \
`transformers==3.0.2`

# Run experiments
## Run LSTM fine-tuning:
`python 1_TrainLSTM.py [random_seed] [train_dev_split_num] [embd_opt] [vocab_min_occur] [hidden_dim] [lr]` 

Example: 
`python 1_TrainLSTM.py 0 5 w2v_general 2 750 1e-4` 

Explanation:
 - `random_seed`: the random seed used in the training script, sets the random seed for python, numpy and pytorch.
 - `train_dev_split_num`: how many splits in the train. Note that the test split ratio is hard coded to the script. That is, the ratio of the test split is always 20% of all data (and the ratio of train&dev data is always 80% of all data). This number here only controls the train/dev ratio. For example, if this number is 5, it mean 1/5 of the train&dev data will be used as the dev set. That makes 16% (80% of all data as train&dev * 20% of train&dev as dev) of all data as dev data, 64% of all data as the train, and 20% of all data as test. 
 - `embd_opt`: the embedding option for the LSTM model. Accepted options:
    + `w2v_general`: use the w2v embeddings trained on over 1M pubmed articles.
    + `w2v_in_domain`: use the w2v embeddings trained on the 10K in-domain pubmed articles (PMC-10000, see the paper for details).
    + `bert_pubmed_10000`: use the embeddings trained by the LSTM language modeling. It uses the sub-word tokenizer (see the paper for details). The vocabulary size is 10K, and the LSTM language model is trained on the 10K in-domain pubmed articles (PMC-10000). 
 - `vocab_min_occur`: effective for the embedding options `w2v_general` and `w2v_in_domain`. It determines whether to train the `unk` token when training the LSTM. If `vocab_min_occur=1`, all tokens in the training set will be added to the vocabulary. So in the training, the `unk` token will not be tuned (so it will just be the default value of the w2v embedding). If it is set to 2, any words that appear less than 2 times in the training data will be considered `unk` so the `unk` is tuned during training. 
 - `hidden_dim`: the number of hidden units of LSTM. 
 - `lr`: learning rate. Set to 1e-4 in our experiments. 
