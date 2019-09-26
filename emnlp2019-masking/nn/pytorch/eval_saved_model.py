import torch
import torch.nn as nn
import os
from torch.utils.data.sampler import BatchSampler
import time

from mean_teacher import architectures, datasets, cli
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import random
import numpy as np
import contextlib
import logging
import torch.nn.functional as F
from mean_teacher.processNLPdata.processNECdata import *

LOG = logging.getLogger('main')
args = None
NA_label = -1
test_student_pred_match_noNA = 0.0
test_student_pred_noNA = 0.0
test_student_true_noNA = 0.0
test_teacher_pred_match_noNA = 0.0
test_teacher_pred_noNA = 0.0
test_teacher_true_noNA = 0.0


def main():
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pr_log = 'pr_files/' + args.run_name + '.pr.scores'
    ema_pr_log = 'pr_files/' + args.run_name + '.pr.scores.ema'
    ckpt_file = 'results/' + args.ckpt_path + '/' + args.ckpt_file  # args.ckpt_path: main_log_gids_l1.0_64_e100_cons1_ramp5_pre_update_rand1000_wf20_fullyLex/2018-05-24_06:46:19/0/transient
    ckpt = torch.load(ckpt_file)
    LOG.info("Loading the checkpoint from :{ckpt_file} ".format(ckpt_file=ckpt_file))
    args.arch = ckpt['arch']

    dataset_config = datasets.__dict__[args.dataset]()
    if args.dataset != 'riedel':
        args.subset_labels = 'None'
        args.labels_set = []
    else:
        if args.subset_labels == '5':
            args.labels_set = ['NA', '/people/person/place_lived', '/people/deceased_person/place_of_death', '/people/person/ethnicity', '/people/person/religion']

        elif args.subset_labels == '10':
            args.labels_set = ['NA', '/people/person/nationality', '/location/country/administrative_divisions', '/people/person/place_of_birth', '/people/deceased_person/place_of_death', '/location/us_state/capital', '/business/company/place_founded', '/sports/sports_team/location', '/people/deceased_person/place_of_burial', '/location/br_state/capital']

        elif args.subset_labels == '20':
            args.labels_set = ['NA', '/location/location/contains', '/people/person/nationality', '/people/person/place_lived', '/location/country/administrative_divisions', '/business/person/company', '/people/person/place_of_birth', '/business/company/founders', '/people/deceased_person/place_of_death', '/business/company/major_shareholders', '/location/us_state/capital', '/location/us_county/county_seat', '/business/company/place_founded', '/location/province/capital', '/sports/sports_team/location', '/people/deceased_person/place_of_burial', '/business/company/advisors', '/people/person/religion', '/time/event/locations', '/location/br_state/capital']

        elif args.subset_labels == 'all':
            args.labels_set = ['NA', '/location/location/contains', '/people/person/nationality', '/location/country/capital', '/people/person/place_lived', '/location/country/administrative_divisions', '/location/administrative_division/country', '/business/person/company', '/people/person/place_of_birth', '/people/ethnicity/geographic_distribution', '/business/company/founders', '/people/deceased_person/place_of_death', '/location/neighborhood/neighborhood_of', '/business/company/major_shareholders', '/location/us_state/capital', '/people/person/children', '/location/us_county/county_seat', '/business/company/place_founded', '/people/person/ethnicity', '/location/province/capital', '/sports/sports_team/location', '/people/place_of_interment/interred_here', '/people/deceased_person/place_of_burial', '/business/company_advisor/companies_advised', '/business/company/advisors', '/people/person/religion', '/time/event/locations', '/location/country/languages_spoken', '/location/br_state/capital', '/film/film_location/featured_in_films', '/film/film/featured_film_locations', '/base/locations/countries/states_provinces_within']
        elif args.subset_labels == 'None':
            args.labels_set = []

    eval_loader, dataset = create_data_loaders(**dataset_config, args=args)

    num_classes = len(dataset.categories)
    print('number of classes: ' + str(num_classes))

    evaldir = os.path.join(dataset_config['datadir'], args.eval_subdir)
    w2vfile = evaldir + "/../../" + args.pretrained_wordemb_file



    if args.pretrained_wordemb:
        dataset.gigaW2vEmbed, dataset.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
        dataset.word_vocab_embed = dataset.create_word_vocab_embed(args)

    else:
        print("Not loading the pretrained embeddings ... ")
        assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                               "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
        dataset.word_vocab_embed = None

    word_vocab_embed = dataset.word_vocab_embed
    word_vocab_size = dataset.word_vocab.size()


    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)

        model_params['word_vocab_embed'] = word_vocab_embed
        model_params['word_vocab_size'] = word_vocab_size
        model_params['wordemb_size'] = args.wordemb_size
        model_params['hidden_size'] = args.hidden_size
        model_params['update_pretrained_wordemb'] = args.update_pretrained_wordemb

        model = model_factory(**model_params)
        LOG.info("--------------------IMPORTANT: REMOVING nn.DataParallel for the moment --------------------")
        if torch.cuda.is_available():
            model = model.cuda()    # Note: Disabling data parallelism for now
        else:
            model = model.cpu()

        if ema:
            for param in model.parameters():
                param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # Init model state dicts
    model.load_state_dict(ckpt['state_dict'])
    ema_model.load_state_dict(ckpt['ema_state_dict'])

    LOG.info(parameters_string(model))

    test_student_pred_file = 'test_results/' + args.run_name + '_test_student_pred.tsv'
    test_teacher_pred_file = 'test_results/' + args.run_name + '_test_teacher_pred.tsv'
    with contextlib.suppress(FileNotFoundError):
        os.remove(test_student_pred_file)
        os.remove(test_teacher_pred_file)
        os.remove(pr_log)
        os.remove(ema_pr_log)

    LOG.info("Evaluating the primary model:")
    f1 = validate(eval_loader, model, pr_log, dataset, "student")
    LOG.info("Evaluating the EMA model:")
    ema_f1 = validate(eval_loader, ema_model, ema_pr_log, dataset, "teacher")
    LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):

    global NA_label
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False

    if args.dataset in ['riedel', 'gids']:

        LOG.info("traindir : " + traindir)
        LOG.info("evaldir : " + evaldir)
        dataset = datasets.REDataset(traindir, args, train_transformation)
        LOG.info("Type of Noise : "+ dataset.WORD_NOISE_TYPE)
        LOG.info("Size of Noise : "+ str(dataset.NUM_WORDS_TO_REPLACE))

        dataset_test = datasets.REDataset(evaldir, args, eval_transformation)

        eval_loader = torch.utils.data.DataLoader(dataset_test,
                                                  pin_memory=pin_memory,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=2 * args.workers,
                                                  drop_last=False)

        NA_label = dataset.categories.index('NA')


    return eval_loader, dataset_test

# ps - model_type is just string
def validate(eval_loader, model, pr_log, dataset, model_type):
    global NA_label
    global test_student_pred_match_noNA
    global test_student_pred_noNA
    global test_student_true_noNA
    global test_teacher_pred_match_noNA
    global test_teacher_pred_noNA
    global test_teacher_true_noNA

    if torch.cuda.is_available():
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval() ### From the documentation (nn.module,py) : i) Sets the module in evaluation mode. (ii) This has any effect only on modules such as Dropout or BatchNorm. (iii) Returns: Module: self

    end = time.time()

    # we will use this to keep track of the data point index
    data_seen = 0
    # i is index of the minibatch
    for i, datapoint in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        if args.dataset in ['riedel', 'gids']:

            input = datapoint[0]
            lengths = datapoint[1]
            target = datapoint[2]

            if torch.cuda.is_available():
                input_var = torch.autograd.Variable(input, volatile=True).cuda()
                seq_lengths = torch.cuda.LongTensor([x for x in lengths])

            else:
                input_var = torch.autograd.Variable(input, volatile=True).cpu()
                seq_lengths = torch.LongTensor([x for x in lengths])

        if torch.cuda.is_available():
            target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        else:
            target_var = torch.autograd.Variable(target.cpu(), volatile=True) ## NOTE: AJAY - volatile: Boolean indicating that the Variable should be used in inference mode,

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()

        if labeled_minibatch_size == 0:
            print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AJAY: Labeled_minibatch_size == 0 ....%%%%%%%%%%%%%%%%%%%%%%%")
            continue
        ###################################################
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        if args.dataset in ['riedel', 'gids']and args.arch == 'lstm_RE':
            output1, perm_idx_test = model((input_var, seq_lengths))
            target_var = target_var[perm_idx_test]

        ## AVG MODEL
        elif args.dataset in ['riedel', 'gids'] and args.arch == 'simple_MLP_embed_RE':
            output1 = model((input_var, seq_lengths, dataset.pad_id))

            if torch.cuda.is_available():
                perm_idx_test = torch.cuda.LongTensor([i for i in range(len(input_var))])
            else:
                perm_idx_test = torch.LongTensor([i for i in range(len(input_var))])

        else:
            output1 = model(input_var) ##, output2 = model(input_var)
        #softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)

        class_loss = class_criterion(output1, target_var) / minibatch_size

        output_softmax = F.softmax(output1, dim=1)

        # Print the scores for the data in the minibatch
        # output1: Variable
        # out_data is size (batch_size, num_classes)

        if args.dataset in ['riedel', 'gids']:
            correct_test, num_target_notNA_test, num_pred_notNA_test = prec_rec(output1.data, target_var.data, NA_label, topk=(1,))
            if num_pred_notNA_test > 0:
                prec_test = float(correct_test) / float(num_pred_notNA_test)
            else:
                prec_test = 0.0

            if num_target_notNA_test > 0:
                rec_test = float(correct_test) / float(num_target_notNA_test)
            else:
                rec_test = 0.0

            if prec_test + rec_test == 0:
                f1_test = 0
            else:
                f1_test = 2 * prec_test * rec_test / (prec_test + rec_test)

            meters.update('correct_test', correct_test, 1)
            meters.update('target_notNA_test', num_target_notNA_test, 1)
            meters.update('pred_notNA_test', num_pred_notNA_test, 1)

            if float(meters['pred_notNA_test'].sum) == 0:
                accum_prec_test = 0
            else:
                accum_prec_test = float(meters['correct_test'].sum) / float(meters['pred_notNA_test'].sum)

            if float(meters['target_notNA_test'].sum) == 0:
                accum_rec_test = 0
            else:
                accum_rec_test = float(meters['correct_test'].sum) / float(meters['target_notNA_test'].sum)

            if accum_prec_test + accum_rec_test == 0:
                accum_f1_test = 0
            else:
                accum_f1_test = 2 * accum_prec_test * accum_rec_test / (accum_prec_test + accum_rec_test)

            meters.update('class_loss', class_loss.data[0], labeled_minibatch_size)


            dump_result(i, args, output1.data, target_var.data, dataset, perm_idx_test, output_softmax, pr_log, 'test_'+model_type, topk=(1,))

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(eval_loader) - 1:
            if args.dataset in ['riedel', 'gids']:
                LOG.info(
                    'Test: [{0}/{1}]  '
                    'ClassLoss {meters[class_loss]:.4f}  '
                    'Precision {prec:.3f} ({accum_prec:.3f})  '
                    'Recall {rec:.3f} ({accum_rec:.3f})  '
                    'F1 {f1:.3f} ({accum_f1:.3f})'.format(
                        i, len(eval_loader), prec=prec_test, accum_prec=accum_prec_test, rec=rec_test, accum_rec=accum_rec_test, f1=f1_test, accum_f1=accum_f1_test, meters=meters))

    # FINAL PERFORMANCE
    if args.dataset in ['riedel', 'gids']:

        if model_type == 'student':
            if test_student_pred_noNA == 0.0:
                student_precision = 0.0
            else:
                student_precision = test_student_pred_match_noNA / test_student_pred_noNA
            if test_student_true_noNA == 0.0:
                student_recall = 0.0
            else:
                student_recall = test_student_pred_match_noNA / test_student_true_noNA
            if student_precision + student_recall == 0.0:
                student_f1 = 0.0
            else:
                student_f1 = 2 * student_precision * student_recall / (student_precision + student_recall)

            LOG.info('******* [Test] Student : Overall Precision {0}  Recall {1}  F1 {2}  ********'.format(
                    student_precision, student_recall, student_f1))

        else:
            if test_teacher_pred_noNA == 0.0:
                teacher_precision = 0.0
            else:
                teacher_precision = test_teacher_pred_match_noNA / test_teacher_pred_noNA
            if test_teacher_true_noNA == 0.0:
                teacher_recall = 0.0
            else:
                teacher_recall = test_teacher_pred_match_noNA / test_teacher_true_noNA
            if teacher_precision + teacher_recall == 0.0:
                teacher_f1 = 0.0
            else:
                teacher_f1 = 2 * teacher_precision * teacher_recall / (teacher_precision + teacher_recall)

            LOG.info('******* [Test] Teacher : Overall Precision {0}  Recall {1}  F1 {2}  ********'.format(
                teacher_precision, teacher_recall, teacher_f1))

    return accum_f1_test


def prec_rec(output, target, NA_label, topk=(1,)):
    maxk = max(topk)
    assert maxk == 1, "Right now only computing P/R/F for topk=1"
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # size of targe and pred: 256 (it's the batch...)
    # some percentage of target have a label

    # number of instances WE predict that are not NA
    # includes the UNLABELED --> number of supervised(labeled) + unsupervised(labeled)

    tp_fn_1 = target.ne(NA_label)  # 1s where not NA
    tp_fn_2 = target.ne(NO_LABEL)  # 1s where labels not NONE
    tp_fn_idx = tp_fn_1.eq(tp_fn_2)   # 1s where target labels are not NA and not NONE; Note tp_fn_1 and tp_fn_2 do not equal to 0 at same index, tp_fn_1[idx] =0 means NA, tp_fn_2[idx] =0 means no label, they do not happen at the same time
    tp_fn = tp_fn_idx.sum()  # number of target labels which are not NA and not NONE (number of non NA labels in ONLY supervised portion of target)

    # index() takes same size of pred with idx value 0 and 1, and only return pred[idx] where idx is 1
    tp_fp = pred.index(tp_fn_2).ne(NA_label).sum()  # number of non NA labels in pred where target labels are not NONE  (Note: corresponded target labels can be NA)

    tp = pred.index(tp_fn_idx).eq(target.view(1, -1).index(tp_fn_idx)).sum()  # number of matches where target labels are not NA and not NONE

    return tp, tp_fn, tp_fp


def dump_result(batch_id, args, output, target, dataset, perm_idx, output_softmax, pr_log, model_type='test_teacher', topk=(1,)):
    global test_student_pred_match_noNA
    global test_student_pred_noNA
    global test_student_true_noNA
    global test_teacher_pred_match_noNA
    global test_teacher_pred_noNA
    global test_teacher_true_noNA

    _, num_labels = output.shape

    # NA_label: Int == 4
    NA_label = dataset.categories.index('NA')

    maxk = max(topk)
    assert maxk == 1, "Right now only computing for topk=1"
    score, prediction = output.topk(maxk, 1, True, True)

    dataset_config = datasets.__dict__[args.dataset]()
    evaldir = os.path.join(dataset_config['datadir'], args.eval_subdir)
    student_pred_file = 'test_results/' + args.run_name + '_' + model_type + '_pred.tsv'
    teacher_pred_file = 'test_results/' + args.run_name + '_' + model_type + '_pred.tsv'

    if torch.cuda.is_available():
        order_idx = perm_idx.cpu().numpy()
    else:
        order_idx = perm_idx.numpy()

    lbl_categories = dataset.categories

    if model_type == 'test_teacher':
        oov_label_lineid = dataset.oov_label_lineid
        dataset_file = evaldir + '/' + args.eval_subdir + '.txt'
        f = open(dataset_file)
        lines = []
        for line_id, line in enumerate(f):
            if line_id not in oov_label_lineid:
                lines.append(line)

        with open(pr_log, "a") as fpr:
            for data_idx, scores in enumerate(output_softmax):
                line_id = int(batch_id * args.batch_size + order_idx[data_idx])
                line = lines[line_id].strip()
                entity_pair = '&&'.join(line.split('\t')[2:3]).replace(',', '')
                for label_idx in range(num_labels):
                    if torch.cuda.is_available():
                        label_confidence = scores[label_idx].data.cpu().numpy()[0]
                    else:
                        label_confidence = scores[label_idx].data.numpy()[0]
                    to_print = "{0},{1},{2},{3},{4}\n".format(line_id, label_idx, label_confidence, entity_pair, target[data_idx])
                    fpr.write(to_print)

        with open(teacher_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                line_id = int(batch_id * args.batch_size + order_idx[p])
                line = lines[line_id].strip()
                lbl_id = int(pre)
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                vals = line.split('\t')
                true_label = vals[4].strip()
                match = pred_label == target_label

                if len(args.labels_set) == 0 or true_label in args.labels_set:
                    assert true_label == target_label

                if match and target_label != 'NA':
                    test_teacher_pred_match_noNA += 1.0
                if pred_label != 'NA':
                    test_teacher_pred_noNA += 1.0
                if target_label != 'NA':
                    test_teacher_true_noNA += 1.0

                line = line + '\t' + target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)

    elif model_type == 'test_student':
        oov_label_lineid = dataset.oov_label_lineid
        dataset_file = evaldir + '/' + args.eval_subdir + '.txt'
        f = open(dataset_file)
        lines = []
        for line_id, line in enumerate(f):
            if line_id not in oov_label_lineid:
                lines.append(line)

        with open(pr_log, "a") as fpr:
            for data_idx, scores in enumerate(output_softmax):
                line_id = int(batch_id * args.batch_size + order_idx[data_idx])
                line = lines[line_id].strip()
                entity_pair = '&&'.join(line.split('\t')[2:3]).replace(',', '')
                for label_idx in range(num_labels):
                    if torch.cuda.is_available():
                        label_confidence = scores[label_idx].data.cpu().numpy()[0]
                    else:
                        label_confidence = scores[label_idx].data.numpy()[0]
                    to_print = "{0},{1},{2},{3},{4}\n".format(line_id, label_idx, label_confidence, entity_pair, target[data_idx])
                    fpr.write(to_print)

        with open(student_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                line_id = int(batch_id * args.batch_size + order_idx[p])
                line = lines[line_id].strip()
                lbl_id = int(pre)
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                vals = line.split('\t')
                true_label = vals[4].strip()
                match = pred_label == target_label

                if len(args.labels_set) == 0 or true_label in args.labels_set:
                    assert true_label == target_label

                if match and target_label != 'NA':
                    test_student_pred_match_noNA += 1.0
                if pred_label != 'NA':
                    test_student_pred_noNA += 1.0
                if target_label != 'NA':
                    test_student_true_noNA += 1.0

                line = line + '\t' + target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)


if __name__ == '__main__':

    args = cli.parse_commandline_args()
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        # torch.cuda.manual_seed_all(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    print('----------------')
    print("Running with args:")
    print('----------------')
    print(args)
    print('----------------')
    # main(RunContext(__file__, 0, args.run_name))
    main()
