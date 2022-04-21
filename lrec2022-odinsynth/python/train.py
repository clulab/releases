import argparse

import json
from pathlib import Path
from typing import Dict

from queryparser import QueryParser
from torch.utils.data.dataloader import DataLoader
from utils import init_random
from model import BaseModel, PointwiseBM
from data_model import make_databundle, make_databundle_from_names
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import datasets
from datasets.load import load_dataset
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from shutil import copyfile

"""
Functions for training in stages.
Each function is responsible with:
    - Creating its dataset, according to the respective stage

"""

"""
Unrolled training stage 
This training stage represents training with the unrolled data.
This means that each batch contains, minimally:
    <sentence_k, rule> and the prediction is either 1, meaning that <rule> is good for <sentence_k> or 0 otherwise
    Notice that even if there can be multiple sentences in some specs, we only treat them one by one


Parameters allowed in kwargs:
    train_path                    -> from where to load the train data (default=config.DATA_DIR/'pointwise_unrolled_train.tsv')
    test_path                     -> from where to load the test data (default=config.DATA_DIR/'pointwise_unrolled_test.tsv')
    cache_dir                     -> where to save the huggingface datasets cache (default='/data/nlp/corpora/huggingface-datasets-cache/')
    split_dataset_on_length       -> whether to split the train dataset on length to obtain two datasets, one strictly smaller and the other one bigger or equals to thresold
    split_dataset_threshold       -> the threshold used to split the dataset
    split_dataset_training        -> a dictionary containing vvarious parameters for training (e.g. train on small + train on big + train on both or train only on small + big or only on small)
            small_dataset_modelcheckpoint -> ModelCheckpoint to use when training on small
            train_on_big_dataset          -> whether to train on the big dataset as well (default=False)
            big_dataset_max_epochs        -> max epochs when training on the big dataset
            big_dataset_modelcheckpoint   -> ModelCheckpoint to use when training on big
            train_on_both                 -> whether to train on small + big after training on them separately
            both_datasets_max_epochs      -> max epochs when training on the small dataset
            both_datasets_modelcheckpoint -> ModelCheckpoint to use when training on both
    resume_from_checkpoint        -> load the model from here or not (default=None)


After each creation of a Trainer we create a new EarlyStop to reset it
The resulting model after each trainer is the one corresponding to the last epoch

"""
def unrolled_training_stage(model, params, training_params, callbacks, **kwargs):
    model.training_regime = 1

    checkpoint_prefix = kwargs.get('checkpoint_prefix', 's1')
    checkpoint_suffix = kwargs.get('checkpoint_suffix', 's1')
    return_values = []
    parser = QueryParser()
    dataset = kwargs['dataset']
    dl_t = DataLoader(dataset['train'], batch_size=params.get('train_batch_size', 256), collate_fn = lambda x: model.collate_fn(model.tokenizer, model.symbols, model.symbol_tensors, parser, x), shuffle=True, num_workers=32)
    dl_v = DataLoader(dataset['test'],  batch_size=params.get('val_batch_size', 256), collate_fn = lambda x: model.collate_fn(model.tokenizer, model.symbols, model.symbol_tensors, parser, x), num_workers=32)
    cp = ModelCheckpoint(
        monitor    = 'f1',
        save_top_k = 7,
        mode       = 'max',
        save_last=True,
        filename=checkpoint_prefix + 's-' + kwargs.get('checkpoint_format', 'epoch={epoch}-step={step}-val_loss={val_loss:.3f}-f1={f1:.3f}-p={p:.3f}-r={r:.3f}')
    )
    # cp = kwargs.get('split_dataset_training', {}).get('dataset_modelcheckpoint', base_cp)
    lrm = LearningRateMonitor(logging_interval='step')

    es = EarlyStopping(
        monitor  = 'f1',
        patience = 3,
        mode     = 'max'
    )

    # model.num_training_steps = len(dl_t)
    # Set suffix/name such that the plots in the logger are not overlapping
    # model.logging_suffix = '_' + checkpoint_prefix + checkpoint_suffix
    # model.lr_scheduler_name = '_' + checkpoint_prefix + checkpoint_suffix
    trainer = pl.Trainer(
        **training_params,
        callbacks = [*callbacks, lrm, es, cp],
        # accumulate_grad_batches = 8,
        # val_check_interval=0.25,
        # limit_train_batches=50000,
        # limit_val_batches=25000,
        # limit_test_batches=1
        # resume_from_checkpoint='/home/rvacareanu/projects/odinsynth/python/logs/odinsynth/version_678/checkpoints/s1-epoch=epoch=0-step=step=323748-val_loss=val_loss=0.000-f1=f1=0.631-p=p=0.586-r=r=0.682.ckpt',
    )

    trainer.fit(model, dl_t, dl_v)

    return cp


def unrolled_training_stage_orchestrator(params, training_params, callbacks, **kwargs):
    hparams = {**params, 'training_params': training_params}
    if 'load_from_arrow_dir' in params:
        dataset = datasets.load_from_disk(params['load_from_arrow_dir'])
    else:
        dataset = load_dataset(
            'text', 
            data_files = {
                # 'train': config.DATA_DIR/'pointwise_unrolled_train.tsv', 
                # 'test' : config.DATA_DIR/'pointwise_unrolled_test.tsv'
                'train': params.get('train_path', config.DATA_DIR/'pointwise_unrolled_train.tsv'),
                'test' : params.get('test_path', config.DATA_DIR/'pointwise_unrolled_test.tsv')
                }, 
            cache_dir = params.get('cache_dir', '/data/nlp/corpora/huggingface-datasets-cache/')
        )
        def encode(example):
            return {'text': [x.split('\t') for x in example['text']]}
            
        dataset = dataset.map(encode, batched=True, num_proc=8)#.shuffle()

    # If we split on dataset length, then we train either on the smaller one or on the bigger one
    if params.get('split_dataset_on_length', False):
        threshold = params.get('split_dataset_threshold', config.SPLIT_DATASET_THRESHOLD)
        print(f'dataset split into two, a shorter version (<{threshold}) and a longer version')
        dataset_small = dataset.filter(function = lambda x: len(x['text'][0].split(' ')) < threshold)
        dataset_big   = dataset.filter(function = lambda x: len(x['text'][0].split(' ')) >= threshold)

        if params.get('train_on_small_dataset', False):
            kwargs['dataset'] = dataset_small
        elif params.get('train_on_big_dataset', False):
            kwargs['dataset'] = dataset_big
    else:
        kwargs['dataset'] = dataset

    # hparams['num_training_steps'] depends on the learning rate scheduler used. If none, it doesn't matter. If CyclicLR, it sets
    # the value for a cycle. If decay, it sets 
    # We set it to the value passed explicitly (if any). Otherwise, len(dataset)/batch_size * scaling_factor (scaling_factor is from cli or config)
    hparams['num_training_steps'] = params.get('num_training_steps', int((len(kwargs['dataset']['train'])/(params['train_batch_size'] * params['accumulate_grad_batches'])) * params['num_training_steps_factor']))
    if 'load_from_checkpoint' in params:
        # Load from checkpoint with the hparams parameters
        # Used when continuing training
        model = PointwiseBM.load_from_checkpoint(params['load_from_checkpoint'], hparams=hparams)
    else:
        model = PointwiseBM(hparams)

    print(model.num_training_steps)
    cp = unrolled_training_stage(model, params, training_params, callbacks, **kwargs)

    return cp


"""
Rolled training stage 
This training stage represents training with the rolled data.
This means that each batch contains all the sentences and all the possible continuations from a given rule:

Parameters allowed in kwargs:
    specs_dir              -> path to the specs used for training (default=config.SPECS_DIR)
    steps_dir              -> path to the steps used for training (default=config.STEPS_DIR)
    model_checkpoint       -> ModelCheckpoint to use when training
    resume_from_checkpoint -> path from where to resume training (default=None)
"""
def rolled_training_stage(params, training_params, callbacks, **kwargs):
    hparams = {**params, 'training_params': training_params}

    checkpoint_prefix = params.get('checkpoint_prefix', 's2')
    parser = QueryParser()
    data = make_databundle_from_names(specs_dir=Path(params['specs_dir']), steps_dir=Path(params['steps_dir']))

    hparams['num_training_steps'] = params.get('num_training_steps', int((len(data['train'])/(params['train_batch_size'] * params['accumulate_grad_batches'])) * params['num_training_steps_factor']))
    if 'load_from_checkpoint' in params:
        # Load from checkpoint with the hparams parameters
        # Used when continuing training
        model = PointwiseBM.load_from_checkpoint(params['load_from_checkpoint'], hparams=hparams)
    else:
        model = PointwiseBM(hparams)

    model.training_regime = 2
    dl_t = DataLoader(data['train'], batch_size=params.get('train_batch_size', 8), collate_fn = lambda x: BaseModel.collate_fn_rolled(model.tokenizer, model.symbols, model.symbol_tensors, parser, x, model.collate_fn), shuffle=True, num_workers=48)
    dl_v = DataLoader(data['test'],  batch_size=params.get('val_batch_size', 16), collate_fn = lambda x: BaseModel.collate_fn_rolled(model.tokenizer, model.symbols, model.symbol_tensors, parser, x, model.collate_fn), num_workers=48)

    base_cp = ModelCheckpoint(
        monitor    = 'mrr',
        save_top_k = 7,
        mode       = 'max',
        save_last=True,
        filename='s2-' + kwargs.get('checkpoint_format', 'epoch={epoch}-step={step}-val_loss={val_loss:.3f}-f1={f1:.3f}-p={p:.3f}-r={r:.3f}-mrr={mrr:.3f}')
    )
    cp = kwargs.get('model_checkpoint', base_cp)

    lrm = LearningRateMonitor(logging_interval='step')

    es = EarlyStopping(
        monitor  = 'mrr',
        patience = 3,
        mode     = 'max'
    )
    # model.lr_scheduler_name = '_' + checkpoint_prefix + '_joint'
    trainer = pl.Trainer(
        **training_params,
        callbacks = [*callbacks, lrm, es, cp],
    )

    trainer.fit(model, dl_t, dl_v)

    print('Finished the second stage')
    print(trainer.test(model, dl_v))
    # print('--------------------------------------------------------')
    # print(trainer.test(model, dl_t))
    print('---------------------------------------------------------')
    print(f'Path to best: {cp.best_model_path}. The score is: {cp.best_model_score}')

    return cp

"""
Reinforcement training stage 
See rl_cleanrl_implementation.py
"""
def rl_training_stage(model, params, callbacks, **kwargs):
    pass

def main(params: Dict):

    print('setting random seed ...')
    init_random(params['random_seed'])

    print('setting tensorboard ...')
    logger = TensorBoardLogger('logs', name=config.MODEL_NAME)

    print('making model ...')

    # Parameters for the trainer
    training_params = {
        'gpus'                    : params.get('gpus', 1),
        'max_epochs'              : params.get('max_epochs', 5),#config.NUM_EPOCHS,
        'min_epochs'              : 0,
        'logger'                  : logger,
        'precision'               : params.get('precision', 16),
        'gradient_clip_val'       : params.get('gradient_clip_val', 1e1),
        'accumulate_grad_batches' : params.get('accumulate_grad_batches', 1),
        'num_sanity_val_steps'    : params.get('num_sanity_val_steps', 5000),
        'resume_from_checkpoint'  : params.get('resume_from_checkpoint', None),
    }
    print("Training params")
    print(training_params)

    kwargs = {
        'checkpoint_format': 'epoch={epoch}-step={step}--train_loss={train_loss:.3f}-val_loss={val_loss:.3f}-f1={f1:.3f}-p={p:.3f}-r={r:.3f}-mrr={mrr:.3f}',
    }
    
    callbacks = []
    
    # kwargs['checkpoint_prefix'] = 's1'
    # print(kwargs)
    if params['training_regime'] == 1:
        cp = unrolled_training_stage_orchestrator(params, training_params, callbacks, **kwargs)
    elif params['training_regime'] == 2:
        cp = rolled_training_stage(params, training_params, callbacks, **kwargs)
    elif params['training_regime'] == 3:
        raise ValueError("This training regime is not supported at the moment")

    print(f'path: {cp.best_model_path}\t{cp.best_model_score} ({cp.monitor})')
    copyfile(cp.best_model_path, f'{cp.dirpath}/best.ckpt')
    if 'save_best_in' in params:
        copyfile(cp.best_model_path, params['save_best_in'])
        
    

    kwargs['checkpoint_prefix'] = 's2'
    # cps = rolled_training_stage(om, params, callbacks, **kwargs)
    # for cp in cps:
    #     print(f'path: {cp.best_model_path}\t{cp.best_model_score} ({cp.monitor})')

  
# python train.py --config-file configs/config1.json --split-dataset-on-length True --split-dataset-threshold 20 --train-on-small-dataset True
# or
# python train.py --config-file configs/config1.json -sdol True -sdt 20 -tosd True
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Entry point of the application.")
    parser = BaseModel.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)

    # Set good defaults inside a standard config file (e.g. configs/config1.json). Override in the command-line as needed
    parser.add_argument('--config-file', type=str, required=True, 
        help="Path to the file containig the config. Any command line parameter overrides the config value")

    parser.add_argument('--max-epochs', type=int, required=False,
        help="Maximum number of epochs (default=5)")

    parser.add_argument('--random-seed', type=int, required=False,
        help="What random seed to use")

        
    # Dataset split thresholds
    parser.add_argument('-sdol', '--split-dataset-on-length', type=bool, required=False, 
        help="Whether to split the dataset on length or not")
    parser.add_argument('-sdt', '--split-dataset-threshold', type=int, required=False, 
        help="The threshold to use when spliting on length (default=20)")
    parser.add_argument('-tosd', '--train-on-small-dataset', type=bool, required=False, 
        help="Train on the small dataset (< threshold)")
    parser.add_argument('-tobd', '--train-on-big-dataset', type=bool, required=False, 
        help="Train on the big dataset (>= threshold")


    parser.add_argument('--training-regime', type=int, required=False,
        help="Which training training regime to use (default=1) ({1,2,3}, where 1 -> binary prediction, 2 -> max margin, 3 -> RL (not supported at the moment))")

    # Training paths
    parser.add_argument('--train-path', type=str, required=False,
        help="Train path for training with training regime 1 training")
    parser.add_argument('--test-path', type=str, required=False,
        help="Val path for training with training regime 1 training")
    parser.add_argument('--specs-dir', type=str, required=False,
        help="Train path for training with training regime 2 training")
    parser.add_argument('--steps-dir', type=str, required=False,
        help="Val path for training with training regime 2 training")

    parser.add_argument('--num-training-steps-factor', type=float, required=False, 
        help='When using a scheduler, we set the num_training_steps=(len(train)/batch_size) * num_training_steps_factor (For CyclicalLR we set this value to step_size_up). This factor helps, for example for CyclicalLR with triangular if you want multiple triangles per epoch (or one triangle for multiple epochs). Also, there is a specific parameter for setting this explicitly')
    parser.add_argument('--num-warmup-steps-factor', type=float, required=False, 
        help='When using a scheduler, we set the num_warmup_steps=(len(train)/batch_size) * num_warmup_steps_factor. Also, there is a specific parameter for setting this explicitly')



    parser.add_argument('--train-batch-size', type=int, help="Batch size to use for training")
    parser.add_argument('--val-batch-size', type=int, help="Batch size to use for validation")
    parser.add_argument('--checkpoint-prefix', type=str, help="Add a prefix to the filename when saving it")

    parser.add_argument('--load-from-arrow-dir', type=str, help="If specified, load from a directory that was creadet with dataset.save_to_disk")
    parser.add_argument('--resume-from-checkpoint', type=str, help="If set, resume training from a checkpoint")
    parser.add_argument('--load-from-checkpoint', type=str, help="If set, load model from a checkpoint")

    # PL Trainer. Not adding parser because it has defaults, overriding the values set in the config
    parser.add_argument('--precision', type=int, help="PL Trainer parameter. Precision {16, 32}")
    parser.add_argument('--gpus', type=int, help="PL Trainer parameter. Number of gpus")
    parser.add_argument('-nsvs', '--num-sanity-val-steps', type=int, help="PL Trainer parameter. Number of batches to do for sanity check")
    parser.add_argument('-gcv', '--gradient-clip-val', type=int, help="PL Trainer parameter. Gradient clip")
    parser.add_argument('-agb', '--accumulate-grad-batches', type=int, help="PL Trainer parameter. How many batches to accumulate the gradient for")

    parser.add_argument('--save-best-in', type=str, help="Where to save the best model. The checkpoints are saved in logs regardless of this parameter. This is useful for piping commands")

    result = parser.parse_args()
    result = vars(result)

    if result['config_file'] is not None:
        config_params = json.load(open(result['config_file']))
    else:
        config_params = {}

    params = {**config_params} # Can use config | result for python version >= 3.9.0
    
    # Override config with parameters from command line. We don't provide defaults because we check for None (and we could not check for defaults)
    for key, value in result.items():
        if value is not None:
            params[key] = value
    print("Params:")
    print(params)
    # exit()

    # result['train_path'] = Path(result['train_path'])
    # result['test_path'] = Path(result['test_path'])
    # result['specs_dir'] = Path(result['specs_dir'])
    # result['steps_dir'] = Path(result['steps_dir'])

    # config = json.load(open(result.config_file))

    main(params)
