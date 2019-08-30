import os
import os,csv,sys
from copy import deepcopy
from typing import List, Union, Dict, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
#from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader
from rte.mithun.read_fake_news_data import load_fever_DataSet
from rte.mithun.log import setup_custom_logger

from tqdm import tqdm

import argparse
import logging
import sys
import json
import numpy as np
from sklearn.externals import joblib
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def eval_model(data, mithun_logger, path_to_trained_models_folder, name_of_trained_model_to_use,cuda_device) -> Model:

    mithun_logger.info("got inside eval_model ")
    mithun_logger.info(f"value of path_to_trained_models_folder is:{path_to_trained_models_folder} ")
    mithun_logger.info(f"value of name_of_trained_model_to_use is:{name_of_trained_model_to_use} ")
    mithun_logger.info(f"value of cuda_device is:{cuda_device} ")
    mithun_logger.info(f"value of path_to_trained_models_folder is:{path_to_trained_models_folder} ")
    mithun_logger.info(f"value of path_to_trained_models_folder is:{path_to_trained_models_folder} ")

    archive = load_archive(os.getcwd()+"/"+path_to_trained_models_folder + name_of_trained_model_to_use, cuda_device)
    model = archive.model
    model.eval()


    actual = []
    predicted = []
    pred_dict = defaultdict(int)

    for item in tqdm(data.instances):
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            # Handles some edge case we presume, never really gets used
            cls = "NOT ENOUGH INFO"
        else:
            prediction = model.forward_on_instance(item, cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]


        if "label" in item.fields:
            actual.append(item.fields["label"].label)
        predicted.append(cls)
        pred_dict[cls] += 1


        if "label" in item.fields:
            mithun_logger.debug(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")
        else:
            mithun_logger.debug(json.dumps({"predicted":cls})+"\n")



    if len(actual) > 0:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

        mithun_logger.info(accuracy_score(actual, predicted))
        mithun_logger.info(classification_report(actual, predicted))
        mithun_logger.info(confusion_matrix(actual, predicted))



    return model


def eval_model_fnc_data(db: FeverDocDB, args,mithun_logger,name_of_trained_model_to_use,path_to_trained_models_folder,cuda_device,operation,path_to_fnc_annotated_data) -> Model:



    print("got inside eval_model_fnc_data")
    archive = load_archive(path_to_trained_models_folder+name_of_trained_model_to_use, cuda_device)
    config = archive.config
    ds_params = config["dataset_reader"]


    model = archive.model
    model.eval()

    reader = FEVERReader(db,
                         sentence_level=ds_params.pop("sentence_level", False),
                         wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                         claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                         token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    # do annotation on the fly  using pyprocessors. i.e creating NER tags, POS Tags etcThis takes along time.
    #  so almost always we do it only once, and load it from disk . Hence do_annotation_live = False
    do_annotation_live = False




    data = reader.read_annotated_fnc_and_do_ner_replacement( args, operation, do_annotation_live,mithun_logger,path_to_fnc_annotated_data).instances
    joblib.dump(data, "fever_dev_dataset_format.pkl")
    #
    ###################end of running model and saving

    path=os.getcwd()

    #data=joblib.load(path+"fever_dev_dataset_format")

    actual = []

    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")
    if_ctr, else_ctr = 0, 0
    pred_dict = defaultdict(int)

    for item in tqdm(data):
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
            if_ctr += 1
        else:
            else_ctr += 1

            prediction = model.forward_on_instance(item, args.cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]


        if "label" in item.fields:
            actual.append(item.fields["label"].label)
        predicted.append(cls)
        pred_dict[cls] += 1

        if args.log is not None:
            if "label" in item.fields:
                f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")
            else:
                f.write(json.dumps({"predicted":cls})+"\n")
    print(f'if_ctr = {if_ctr}')
    print(f'else_ctr = {else_ctr}')
    print(f'pred_dict = {pred_dict}')


    if args.log is not None:
        f.close()


    if len(actual) > 0:
        print(accuracy_score(actual, predicted))
        print(classification_report(actual, predicted))
        print(confusion_matrix(actual, predicted))

    return model


def convert_fnc_to_fever_and_annotate(db: FeverDocDB, path_to_trained_models, logger, cuda_device,path_to_pyproc_annotated_data_folder) -> Model:
    archive = load_archive(path_to_trained_models, cuda_device=cuda_device)
    config = archive.config
    ds_params = config["dataset_reader"]
    model = archive.model
    model.eval()


    cwd = os.getcwd()
    fnc_data_set = load_fever_DataSet()

    #to annotate with pyprocessors- comment this part out if you are doing just evaluation
    stances,articles= fnc_data_set.read_parent(cwd, "train_bodies.csv", "train_stances_csc483583.csv")

    dict_articles = {}
    # copy all bodies into a dictionary
    for article in articles:
        dict_articles[int(article['Body ID'])] = article['articleBody']

    load_fever_DataSet.annotate_fnc(cwd, stances,dict_articles,logger,path_to_pyproc_annotated_data_folder)
    print("done with annotation. going to exit")
    sys.exit(1)

    ###########end of pyprocessors annotation

    data=reader.read_fnc(fnc_data_set).instances



    actual = []
    predicted = []

    if args.log is not None:
        f = open(args.log,"w+")

    for item in data:
        if item.fields["premise"] is None or item.fields["premise"].sequence_length() == 0:
            cls = "NOT ENOUGH INFO"
        else:
            prediction = model.forward_on_instance(item, args.cuda_device)
            cls = model.vocab._index_to_token["labels"][np.argmax(prediction["label_probs"])]

        actual.append(item.fields["label"].label)
        predicted.append(cls)

        if args.log is not None:
            f.write(json.dumps({"actual":item.fields["label"].label,"predicted":cls})+"\n")

    if args.log is not None:
        f.close()

    print(accuracy_score(actual, predicted))
    print(classification_report(actual, predicted))
    print(confusion_matrix(actual, predicted))

    return model



def eval_da(dataset_to_work_on,args,operation,mithun_logger):
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)




    params = Params.from_file(args.param_path,args.overrides)
    uofa_params = params.pop('uofa_params', {})
    path_to_saved_db = uofa_params.pop("path_to_saved_db")
    db = FeverDocDB(path_to_saved_db)





    mithun_logger.info("inside main function going to call eval on " + str(dataset_to_work_on))
    mithun_logger.info("path_to_pyproc_annotated_data_folder " + str(path_to_pyproc_annotated_data_folder))
    mithun_logger.info("value of name_of_trained_model_to_use: " + str(name_of_trained_model_to_use))
    mithun_logger.info("value of dataset_to_work_on: " + str(dataset_to_work_on))




    if(dataset_to_work_on== "fnc"):
        fever_dataset_details = uofa_params.pop('fever_dataset_details', {})
        dev_partition_details = fever_dataset_details.pop('dev_partition_details', {})
        name_of_trained_model_to_use = dev_partition_details.pop('name_of_trained_model_to_use', {})
        path_to_pyproc_annotated_data_folder = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        debug_mode = uofa_params.pop('debug_mode', {})
        path_to_trained_models_folder = uofa_params.pop('path_to_trained_models_folder', {})
        path_to_fnc_annotated_data = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        eval_model_fnc_data (db,args,mithun_logger,name_of_trained_model_to_use,path_to_trained_models_folder,cuda_device,operation,path_to_fnc_annotated_data)

    elif (dataset_to_work_on == "fever"):
        fever_dataset_details = uofa_params.pop('fever_dataset_details', {})
        dev_partition_details = fever_dataset_details.pop('dev_partition_details', {})
        name_of_trained_model_to_use = dev_partition_details.pop('name_of_trained_model_to_use', {})
        path_to_pyproc_annotated_data_folder = dev_partition_details.pop('path_to_pyproc_annotated_data_folder', {})
        debug_mode = uofa_params.pop('debug_mode', {})
        path_to_trained_models_folder = uofa_params.pop('path_to_trained_models_folder', {})

        eval_model(db,args,mithun_logger,path_to_trained_models_folder,name_of_trained_model_to_use)

