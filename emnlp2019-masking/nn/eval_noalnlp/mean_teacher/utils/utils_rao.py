import numpy as np
import torch
import os
import re
import mmap
import torch
from torch.utils.data import  DataLoader
from tqdm import tqdm
import random
from mean_teacher.model import architectures
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from mean_teacher.utils.logger import LOG

# #### General utilities

def set_seed_everywhere(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        LOG.info(f"found that cuda is available. ALso setting the manual seed as {args.random_seed} ")
    else:
        torch.manual_seed(args.random_seed)
        LOG.info(f"found that cuda is not available . ALso setting the manual seed as {args.random_seed} ")


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def generate_batches(dataset,workers,batch_size,device ,shuffle=True,
                     drop_last=True ):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    if(shuffle==True):
        labeled_idxs = dataset.get_all_label_indices(dataset)
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler_local = BatchSampler(sampler, batch_size, drop_last=True)
        dataloader=DataLoader(dataset,batch_sampler=batch_sampler_local,num_workers=workers,pin_memory=True)
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,pin_memory=True,drop_last=False,num_workers=workers)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def get_num_lines(file_path):

    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    total_lines = get_num_lines(glove_filepath)
    with open(glove_filepath, "r") as fp:
        for index, line in tqdm(enumerate(fp),total=total_lines, desc="glove"):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """

    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings,embedding_size

def initialize_double_optimizers(model, args):

    '''
        The code for decomposable attention we use , utilizes two different optimizers
    :param model:
    :param args:
    :return:
    '''
    input_optimizer = None
    inter_atten_optimizer = None
    para1 = model.para1
    para2 = model.para2
    if args.optimizer == 'adagrad':
        input_optimizer = torch.optim.Adagrad(para1, lr=args.learning_rate, weight_decay=args.weight_decay)
        inter_atten_optimizer = torch.optim.Adagrad(para2, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = torch.optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = torch.optim.Adadelta(para2, lr=args.lr)
    else:
        #LOG.info('No Optimizer.')
        print('No Optimizer.')
        import sys
        sys.exit()
    assert input_optimizer != None
    assert inter_atten_optimizer != None

    return input_optimizer,inter_atten_optimizer

def update_optimizer_state(input_optimizer, inter_atten_optimizer,args):
    for group in input_optimizer.param_groups:
        for p in group['params']:
            state = input_optimizer.state[p]
            state['sum'] += args.Adagrad_init
    for group in inter_atten_optimizer.param_groups:
        for p in group['params']:
            state = inter_atten_optimizer.state[p]
            state['sum'] += args.Adagrad_init
    return input_optimizer, inter_atten_optimizer


def create_model(logger_object, args_in,  num_classes_in, word_vocab_embed, word_vocab_size, wordemb_size_in,ema=False,):
    logger_object.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained '
            if args_in.pretrained else '',
            ema='EMA '
            if ema else '',arch=args_in.arch))
    model_factory = architectures.__dict__[args_in.arch]
    model_params = dict(pretrained=args_in.pretrained, num_classes_in=num_classes_in)
    model_params['word_vocab_embed'] = word_vocab_embed
    model_params['word_vocab_size'] = word_vocab_size
    model_params['wordemb_size'] = wordemb_size_in
    model_params['hidden_size'] = args_in.hidden_sz
    model_params['update_pretrained_wordemb'] = args_in.update_pretrained_wordemb
    model_params['para_init'] = args_in.para_init
    model_params['use_gpu'] = args_in.use_gpu
    logger_object.debug(f"value of word_vocab_embed={word_vocab_embed}")
    logger_object.debug(f"value of word_vocab_size={word_vocab_size}")

    model = model_factory(**model_params)

    args_in.device=None
    if(args_in.use_gpu) and torch.cuda.is_available():
        torch.cuda.set_device(0)
        args_in.device = torch.device('cuda')
    else:
        args_in.device = torch.device('cpu')

    model = model.to(device=args_in.device)

    if ema:
        for param in model.parameters():
            param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model

    return model
