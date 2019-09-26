import re
import argparse
import logging

from . import architectures, datasets


LOG = logging.getLogger('main')

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
    parser.add_argument('--dataset', metavar='DATASET', default='conll',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: conll)')
    parser.add_argument('--train-subdir', type=str, default='rte/fever/train/',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--results_subdir', type=str, default='results/',
                        help='the subdirectory where the output will be stored under run_name')
    parser.add_argument('--eval-subdir', type=str, default='rte/fever/dev/',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--glove_subdir', type=str, default='glove/',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default=None, type=str, #metavar='FILE',
                        help='% of labeled data to be used for the NLP task (randomly selected)')
    parser.add_argument('--run_student_only', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='simple_MLP_embed',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled_batch_size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight_decay', help='l2 regularization',
                        type=float, default=5e-5)
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: 1) . Turn it to None when using a simple feed forward network')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--checkpoint-epochs', default=10, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1). Note: this is '
                                               'a way to calculate/find the best epoch. i.e instead of running your training for say 256 epochs, it validates in between say at 20th epoch and 40th epoch to check'
                                               'if any of them give good performance. i.e args.epochs does not mean your training will run non stop from 1 to 256. It will keep evaluating in between')
    parser.add_argument('--print_freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,default=False,
                        help='if you want to do evaluation i think using a loaded checkpoint from disk=.')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--wordemb_size', default=300, type=int,
                        help='size of the word-embeddings to be used in the simple_MLP_embed model (default: 300)')
    parser.add_argument('--hidden_size', default=200, type=int,
                        help='size of the hidden layer to be used in the simple_MLP_embed model (default: 50)')
    parser.add_argument('--pretrained_wordemb', default=True, type=str2bool, metavar='BOOL',
                        help='Use pre-trained word embeddings to be loaded from disk, if True; else random initialization of word-emb (default: True)')
    parser.add_argument('--pretrained_wordemb_file', type=str, default='glove.6B.100d.txt',
                        help='pre-trained word embeddings file')
    parser.add_argument('--update_pretrained_wordemb', default=False, type=str2bool, metavar='BOOL',
                        help='Update the pre-trained word embeddings during training, if True; else keep them fixed (default: False)')
    parser.add_argument('--random-initial-unkown', default=False, type=str2bool, metavar='BOOL',
                        help='Randomly initialize unkown words embedding. It only works when --pretrained-wordemb is True')
    parser.add_argument('--word-frequency', default='2', type=int,
                        help='only the word with higher frequency than this number will be added to vocabulary')
    parser.add_argument('--random-seed', default='20', type=int,
                        help='random seed')
    parser.add_argument('--run-name', default='', type=str, metavar='PATH',
                        help='Name of the run used in storing the results for post-precessing (default: none)')
    parser.add_argument('--word-noise', default='drop:1', type=str,
                        help='What type of noise should be added to the input (NLP) and how much; format= [(drop|replace):X], where replace=replace a random word with a wordnet synonym, drop=random word dropout, X=number of words (default: drop:1) ')
    parser.add_argument('--save-custom-embedding', default=True, type=str2bool, metavar='BOOL',
                        help='Save the custom embedding generated from the LSTM-based custom_embed model (default: True)')
    parser.add_argument('--max-entity-len', default=1000, type=int,
                        help='maximum number of words in entity, extra words would be truncated. update, mithun uses this as --truncate_words_length')
    parser.add_argument('--max-inbetween-len', default='50', type=int,
                        help='maximum number of words in between of two entities, extra words would be truncated')
    parser.add_argument('--ckpt-file', type=str, default='best.ckpt', help='best checkpoint file')
    parser.add_argument('--ckpt-path', type=str, default='', help='path where best checkpoint file locates')
    parser.add_argument('--subset-labels', type=str, default='None',
                        help='if not \'None\', only datpoints with the specified subset of test labels are considered, for both train/dev/test; currently only implemented for fullyLex and headLex of Riedel')
    parser.add_argument('--data_dir', type=str, default='None',
                        help='link to the folder where training and dev data is kept')
    parser.add_argument('--train_input_file', type=str, default='None',
                        help='path to the training data file.folder path is hard coded via:data-local/rte/fever/train')
    parser.add_argument('--dev_input_file', type=str, default='None',
                        help='path to the dev data file. folder path is hard coded via:data-local/rte/fever/dev')
    parser.add_argument('--output_folder', type=str, default='outputs/',
                        help='')

    parser.add_argument('--truncate_words_length', type=int, default=1000,
                        help='if claim or evidence goes beyond.')
    parser.add_argument('--para_init', help='parameter initialization gaussian',
                        type=float, default=0.01)
    parser.add_argument('--use_gpu', help="use or don't use CPU",
                        type=str2bool, default=False)
    parser.add_argument('--log_level', type=str, default='DEBUG',
                        help='AT WHAT LEVEL do you want your logger to be. INFO, DEBUG, ERROR WARNING.')
    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='adagrad')
    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients',
                        type=float, default=0.)
    parser.add_argument('--use_double_optimizers', default=False, type=str2bool, metavar='BOOL',
                        help='libowen code has 2 optimizers and doesnt propagate through one')
    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                                   to have the norm equal to max_grad_norm',
                        type=float, default=5)
    parser.add_argument('--type_of_data', help='in this project we will be feeding in lexicalized and delexicalized data (NER replaced, for example. Options can be plain, ner_replaced etc',
                        type=str, default='plain')
    parser.add_argument('--np_rand_seed', type=int, default=256,
                        help='seed used for all numpy.rand calculations. reproducability.')
    parser.add_argument('--use_local_glove', default=True, type=str2bool,
                        help="you dont want to copy the 5gig size of glove always. if this is false, use it from another hardcoded path", metavar='BOOL')

    print(parser.parse_args())
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
