# Author: Fan luo
 
import dask
import dask.dataframe as dd 
import multiprocessing 
from bridge_phrases_identification import *
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from eval_metircs import *
from datasets import list_datasets, list_metrics, load_dataset, load_metric
dataset = load_dataset('hotpot_qa', 'distractor',split="validation")   
dataset = dataset.filter(lambda x: x["type"] == 'bridge') 
dataset = dataset.map(lambda row: bridge_phrases_extraction(row), load_from_cache_file=False)
dataset = dataset.map(lambda example: {'sentences': list(flatten(example['context']['sentences'])) }) 


# randomly sample 100 exmaples for maunally eval
sample_df = dataset.to_pandas().sample(n=100).loc[:, ['id', 'question', 'answer', 'bridge_phrases', 'steiner_points', 'supporting_facts_text']]
annotation_df = dataset.to_pandas() 

 
###### evidence retrieval ######
 

####### cross-encoder

model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder = CrossEncoder(model_name) 

def nn_retrieval_eval(example, phrase_type=''):
    # @phrase_type:  bridge_phrases, steiner_points
 
    question = example['question']
    query = question  
    
    if phrase_type != '': 
        phrases = example[phrase_type] 
        if len(phrases) > 0:
            query += ' '
            query += ' '.join(phrases) 
                 
    context = example['context']
    sentences = context['sentences']
    titles = list(context["title"])
    formatted_sentences = []
    for t, sents in zip(titles,  sentences):
        for sent in sents:
            formatted_sentences.append(TITLE_START + ' ' + t + ' ' + TITLE_END + ' ' + sent + ' ' + SENT_MARKER_END)
    
    cross_inp = [[query, sent] for sent in formatted_sentences]  
    
    cross_scores = cross_encoder.predict(cross_inp)
    
    rank_index = np.argsort(cross_scores)[::-1] 
    ranked_sents = [formatted_sentences[idx] for idx in rank_index]
    
    
    formatted_sp_sents = [] 
    sp_sent_ids = example['supporting_facts']['sent_id']
    for spi, sp_title in enumerate(example['supporting_facts']['title']): 
        pi = titles.index(sp_title)
        sent_id = sp_sent_ids[spi]
        if pi < len(sentences) and sent_id < len(sentences[pi]):   # in case edge case 
            formatted_sp_sents.append(sp_title + ' ' + sentences[pi][sent_id] )
     
    res = retriveal_eval(formatted_sp_sents, ranked_sents) 
    return res

 
ddf = dd.from_pandas(dataset.to_pandas(), npartitions=4*multiprocessing.cpu_count()) 

# baseline: retrieve with original question
nn_question_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: nn_retrieval_eval(row)), axis=1), meta=object)

###### with requery
nn_steiner_points_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: nn_retrieval_eval(row, phrase_type='steiner_points')), axis=1), meta=object)

nn_bridge_phrases_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: nn_retrieval_eval(row, phrase_type='bridge_phrases')), axis=1), meta=object)


####### BM25

from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords 
import string
from tqdm.autonotebook import tqdm
import numpy as np


# We lower case our text and remove stop-words from indexing
def simple_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in stopwords.words('english'):
            tokenized_doc.append(token)
    return tokenized_doc

def bm25_retrieval_eval(example, with_question=True, phrase_type='', top_k= -1): 
 
    question = example['question'] 
    context = example['context']
    sentences = context['sentences']
    titles = list(context["title"])
 
    query = question if with_question else ''

    if phrase_type != '':
        phrases = example[phrase_type] 
        query += ' '  
        query += ' '.join(phrases)
     
    formatted_sentences = []
    for t, sents in zip(titles,  sentences):
        for sent in sents:
            formatted_sentences.append(t + ' ' + sent)
    
        
    tokenized_corpus = []
    for formatted_sentence in formatted_sentences:
        tokenized_corpus.append(simple_tokenizer(formatted_sentence))
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_scores = bm25.get_scores(simple_tokenizer(query))
    if top_k== -1:
        bm25_hits = [{'corpus_id': idx, 'score': bm25_score} for idx, bm25_score in enumerate(bm25_scores)]
    else:
        idxs = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in idxs]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
      
    retrieved_sents = []  
    scores = []
    for hit in bm25_hits: 
        scores.append(hit['score']) 
        sent = formatted_sentences[hit['corpus_id']].replace("\n", " ")
        retrieved_sents.append(sent) 
        
    formatted_sp_sents = [] 
    sp_sent_ids = example['supporting_facts']['sent_id']
    for spi, sp_title in enumerate(example['supporting_facts']['title']): 
        pi = titles.index(sp_title)
        sent_id = sp_sent_ids[spi]
        if pi < len(sentences) and sent_id < len(sentences[pi]):   # in case edge case, such as in 5ae61bfd5542992663a4f261
            formatted_sp_sents.append(sp_title + ' ' + sentences[pi][sent_id] )
            
    res = retriveal_eval(formatted_sp_sents, retrieved_sents)        
    return res


bm25_question_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: bm25_retrieval_eval(row)), axis=1), meta=object)

###### with requery
bm25_steiner_points_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: bm25_retrieval_eval(row, phrase_type='steiner_points')), axis=1), meta=object)

bm25_bridge_phrases_eval_res = ddf.map_partitions(lambda df: df.apply((lambda row: bm25_retrieval_eval(row, phrase_type='bridge_phrases')), axis=1), meta=object)


####### baseline: random sample k sentences
dataset = dataset.map(lambda row: {'random_rerank_sents': random.sample(list(flatten([[TITLE_START + ' ' + t + ' ' + TITLE_END + ' ' + sent + ' ' + SENT_MARKER_END for t, sent in zip([title] * len(sents), sents)] for title, sents in zip(row['context']['title'], row['context']['sentences'])])), min(top_k, len(list(flatten(row['context']['sentences'])))))}, load_from_cache_file=False) 
