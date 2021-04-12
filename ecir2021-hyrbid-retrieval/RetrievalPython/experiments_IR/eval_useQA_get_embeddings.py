'''
The following command needs to be types before running the python script.
!pip install tensorflow_text
!pip install simpleneighbors
!pip install nltk
'''

import json
import nltk
import os
import pprint
import random
import simpleneighbors
import urllib
from IPython.display import HTML, display

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import numpy as np

from google.colab import output
from google.colab import files

import pickle

def generate_query_embeddings(model, instances_list):
    batch_size = 100

    embeddings_list = list()

    print('Computing embeddings for %s questions' % len(instances_list))
    slices = zip(*(iter(instances_list),) * batch_size)
    num_batches = int(len(instances_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        question_batch = list([question_dict["question"] for question_dict in s])
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if num_batches*batch_size<len(instances_list):
        question_batch = list([question_dict["question"] for question_dict in instances_list[num_batches*batch_size:]])
        encodings = model.signatures['question_encoder'](tf.constant(question_batch))
        for i in range(len(question_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    return np.array(embeddings_list)

def generate_document_embeddings(model, response_list, sent_list, doc_list):
    '''

    :param model: the use-QA model
    :param response_list: a tuple (sent_id, doc_id)
    :param sent_list: a list of strings
    :param doc_list: a list of strings
    :return:
    '''

    batch_size = 100

    embeddings_list = list()

    print('Computing embeddings for %s sentences' % len(response_list))
    slices = zip(*(iter(response_list),) * batch_size)
    num_batches = int(len(response_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        response_batch = list([sent_list[int(sent_id)] for sent_id, doc_id in s])
        context_batch = list([doc_list[int(doc_id)] for sent_id, doc_id in s])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(response_batch),
            context=tf.constant(context_batch)
        )
        for i in range(len(response_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if batch_size*num_batches<len(response_list):
        response_batch = list([sent_list[int(sent_id)] for sent_id, doc_id in response_list[num_batches*batch_size:]])
        context_batch = list([doc_list[int(doc_id)] for sent_id, doc_id in response_list[num_batches*batch_size:]])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(response_batch),
            context=tf.constant(context_batch)
        )
        for i in range(len(response_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))


    return np.array(embeddings_list)


def generate_document_embeddings_no_context(model, response_list):
    '''
    :param model: the use-QA model
    :param response_list: a list of strings
    :return:
    '''

    batch_size = 100

    embeddings_list = list()

    print('Computing embeddings for %s sentences' % len(response_list))
    slices = zip(*(iter(response_list),) * batch_size)
    num_batches = int(len(response_list) / batch_size)
    for n, s in enumerate(slices):
        output.clear(output_tags='progress')
        with output.use_tags('progress'):
            print('Processing batch %s of %s' % (n + 1, num_batches))

        # according to https://tfhub.dev/google/universal-sentence-encoder-qa/3, we should repeat answer_batch if there is no context.
        answer_batch = list([sent for sent in s])
        #context_batch = list([" "])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(answer_batch),
            context=tf.constant(answer_batch)
        )
        for i in range(len(answer_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    if batch_size * num_batches < len(response_list):
        answer_batch = list([sent for sent in response_list[num_batches * batch_size:]])
        encodings = model.signatures['response_encoder'](
            input=tf.constant(answer_batch),
            context=tf.constant(answer_batch)
        )
        for i in range(len(answer_batch)):
            embeddings_list.append(np.array(encodings['outputs'][i]))

    return np.array(embeddings_list)

def main(dataset = "openbook", get_query_embd = True, get_doc_embd = True, doc_range = (0,100000)):

    nltk.download('punkt')
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"  # @param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
    model = hub.load(module_url)


    if dataset=="openbook":
        with open("openbook_useqa_retrieval_data.pickle", "rb") as handle:
            openbook_retrieval_data = pickle.load(handle)

        if get_query_embd:
            questions_dev_embds =generate_query_embeddings(model, openbook_retrieval_data["dev_list"])
            questions_test_embds =generate_query_embeddings(model, openbook_retrieval_data["test_list"])

            print(questions_dev_embds.shape)
            print(questions_test_embds.shape)

            np.save("openbook_ques_dev_embds.npy", questions_dev_embds)
            np.save("openbook_ques_test_embds.npy", questions_test_embds)

        if get_doc_embd:
            sentences_embds = generate_document_embeddings_no_context(model, openbook_retrieval_data["kb"])
            print(sentences_embds.shape)
            np.save("openbook_sents_embds.npy", sentences_embds)

    if dataset=="nq":
        with open("nq_retrieval_data.pickle", "rb") as handle:
            nq_retrieval_data = pickle.load(handle)

        if get_query_embd:
            questions_train_embds =generate_query_embeddings(model, nq_retrieval_data["train_list"])

            print(questions_train_embds.shape)

            np.save("nq_ques_train_embds.npy", questions_train_embds)

        if get_doc_embd:
            sentences_embds = generate_document_embeddings(model, nq_retrieval_data["resp_list"],
                                                           nq_retrieval_data["sent_list"],
                                                           nq_retrieval_data["doc_list"])
            print(sentences_embds.shape)
            np.save("nq_sents_embds.npy", sentences_embds)

    if dataset=="squad":
        with open("squad_retrieval_data.pickle", "rb") as handle:
            squad_retrieval_data = pickle.load(handle)

        if get_query_embd:
            questions_train_embds =generate_query_embeddings(model, squad_retrieval_data["train_list"])
            questions_dev_embds =generate_query_embeddings(model, squad_retrieval_data["dev_list"])

            print(questions_train_embds.shape)
            print(questions_dev_embds.shape)

            np.save("squad_ques_train_embds.npy", questions_train_embds)
            np.save("squad_ques_dev_embds.npy", questions_dev_embds)

        if get_doc_embd:
            sentences_embds = generate_document_embeddings(model, squad_retrieval_data["resp_list"],
                                                           squad_retrieval_data["sent_list"],
                                                           squad_retrieval_data["doc_list"])
            print(sentences_embds.shape)
            np.save("squad_sents_embds.npy", sentences_embds)

    return 0

