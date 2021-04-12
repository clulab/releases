'''
The following command needs to be types before running the python script.
!pip install tensorflow_text
!pip install nltk
'''

import json
import nltk
import os
import pprint
import random
import urllib
from IPython.display import HTML, display

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import numpy as np

from google.colab import output
from google.colab import files


import pickle
import time

def generate_query_embeddings_time(model, instances_list):
    batch_size = 1

    embeddings_list = list()
    time_list = []

    print('Computing embeddings for %s questions' % len(instances_list))
    slices = zip(*(iter(instances_list),) * batch_size)
    num_batches = int(len(instances_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        question_batch = list([query_string for query_string in s])
        encoding_start = time.time()
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        encoding_end = time.time()
        time_list.append(encoding_end-encoding_start)

        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if num_batches*batch_size<len(instances_list):
        question_batch = list([question_dict["question"] for question_dict in instances_list[num_batches*batch_size:]])

        encoding_start = time.time()
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        encoding_end = time.time()
        time_list.append(encoding_end - encoding_start)

        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    return time_list

def load_query(filename):
    with open(filename,"r") as handle:
        query_list = handle.read()

    if (filename=="openbook_test_query.txt"):
        query_list = query_list.split("\n")[:-1]
    else:
        query_list = query_list.split(" QUERY_SEP\n ")[:-1]

    print("query list length:", len(query_list))

    return query_list

def generate_embeddings_and_save_time(query_list, model, filename):
    # Firstly run some warm up examples
    _ = generate_query_embeddings_time(model, query_list[:50])
    time_list = generate_query_embeddings_time(model, query_list)

    print("average time each query:", sum(time_list)/len(time_list))

    np.save(filename, np.array(time_list))
    return 0

openbook_test_query_list = load_query("openbook_test_query.txt")
squad_test_query_list = load_query("squad__test_query.txt")
nq_test_query_list = load_query("nq_test_query.txt")

nltk.download('punkt')
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"  # @param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
model = hub.load(module_url)

openbook_test_time_list = generate_query_embeddings_time(model, openbook_test_query_list)

