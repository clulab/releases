import os

from copy import deepcopy

from allennlp.commands.train import prepare_environment
from typing import List, Union, Dict, Any
from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.data import Vocabulary, DataIterator, DatasetReader, Tokenizer, TokenIndexer,Dataset
from allennlp.models import Model, archive_model
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader
from sklearn.externals import joblib
from rte.mithun.log import setup_custom_logger

import argparse
import logging
import sys
import json

logger = logging.getLogger(__name__)  # pylint:    disable=invalid-name


def train_model(db: FeverDocDB, params: Union[Params, Dict[str, Any]], cuda_device:int,
                serialization_dir: str, filtering: str, randomseed:int, slice:int,mithun_logger,train_data_instances) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """






    #uofa_params = params.pop('uofa_params', {})
    #my_seed = uofa_params.pop('random_seed', {})

    SimpleRandom.set_seeds_from_config_file(randomseed)



    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling    the required parts for the Trainer.

    ds_params = params.pop('dataset_reader', {})

    dataset_reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level",False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})),
                                 filtering=filtering)

    train_data_path = params.pop('train_data_path')
    logger.info("Reading training data from %s", train_data_path)




    #train_data_instances = dataset_reader.read(train_data_path,run_name,do_annotation_on_the_fly,mithun_logger).instances
    #joblib.dump(train_data, "fever_tr_dataset_format.pkl")

    #if you want to train on a smaller slice
    #uofa_params = params.pop('uofa_params', {})
    #training_slice_percent = uofa_params.pop('training_slice_percent', {})

    training_slice_percent=slice


    total_training_data = len(train_data_instances)

    print(total_training_data)

    training_slice_count= int(total_training_data * training_slice_percent/100)
    print(training_slice_count)

    train_data_slice=(train_data_instances[0:training_slice_count]).instances
    train_data=train_data_slice



    all_datasets = [train_data]
    datasets_in_vocab = ["train"]

    validation_data_path = params.pop('validation_data_path', None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        run_name = "dev"
        validation_data = dataset_reader.read(validation_data_path,run_name)
        all_datasets.append(validation_data)
        datasets_in_vocab.append("validation")
        joblib.dump(validation_data, "fever_dev_dataset_format.pkl")
    else:


        validation_data = None

    logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_in_vocab))
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   Dataset([instance for dataset in all_datasets
                                            for instance in dataset.instances]))

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))


    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))

    train_data.index_instances(vocab)
    if validation_data:
        validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    if cuda_device is not None:
        trainer_params["cuda_device"] = cuda_device
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params)
    print("going to start training")
    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model


def train_model_uofa_version( params: Union[Params, Dict[str, Any]], cuda_device: int,
                serialization_dir: str, slice: int, mithun_logger,
                train_data_instances) -> Model:
    mithun_logger.info(f"got inside train_model_uofa_version")
    training_slice_percent = slice
    total_training_data = len(train_data_instances.instances)
    training_slice_count = int(total_training_data * training_slice_percent / 100)
    train_data_slice = (train_data_instances.instances[0:(training_slice_count-1)])
    train_data = train_data_slice

    mithun_logger.info(f"value of total_training_data is {total_training_data}")
    mithun_logger.info(f"value of training_slice_count is {training_slice_count}")
    mithun_logger.info(f"length of the new slice is is {len(train_data)}")
    mithun_logger.info(f"value of the first entry in the new slice is is {(train_data[0])}")



    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)

    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    train_data=Dataset(train_data)
    all_datasets=[train_data]
    datasets_in_vocab = ["train"]

    mithun_logger.info("Creating a vocabulary using %s data.", ", ".join(datasets_in_vocab))
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),Dataset([instance for dataset in all_datasets
                                            for instance in dataset.instances]))
    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    model = Model.from_params(vocab, params.pop('model'))
    iterator = DataIterator.from_params(params.pop("iterator"))

    train_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    if cuda_device is not None:
        trainer_params["cuda_device"] = cuda_device
    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  None,
                                  trainer_params)
    print("going to start training")
    trainer.train()

    # Now tar up results
    archive_model(serialization_dir)

    return model



def train_da(ds,operation,logger_mode):
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)



    params = Params.from_file(args.param_path,args.overrides)
    uofa_params = params.pop('uofa_params', {})
    path_to_saved_db = uofa_params.pop("path_to_saved_db")
    db = FeverDocDB(path_to_saved_db)
    read_random_seed_from_commandline = uofa_params.pop('read_random_seed_from_commandline', {})
    # debug_mode = uofa_params.pop('debug_mode', {})
    # features = uofa_params.pop('features', {})
    # print(f"value of features is{features}")
    # lowercase_tokens = features.pop('lowercase_tokens', {})
    # print(f"value of lowercase_tokens is{lowercase_tokens}")


    #
    # slice = ""
    # random_seed = ""
    #
    # if(read_random_seed_from_commandline):
    #     slice=args.slice
    #     random_seed=args.randomseed
    # else:
    if(ds=="fever" and operation=="train") :
        fever_dataset_details = uofa_params.pop('fever_dataset_details', {})
        train_partition_details = fever_dataset_details.pop('train_partition_details', {})
        slice = train_partition_details.pop('slice_percent', {})
        random_seed = uofa_params.pop('random_seed', {})



    log_file_name="training_feverlog.txt"+str(slice)+"_"+str(random_seed)
    mithun_logger = setup_custom_logger('root', logger_mode ,log_file_name)

    mithun_logger.info(f"Going to train on  {args.slice} percentage of training data with random seed value{args.randomseed}.")
    #todo:call train_model
