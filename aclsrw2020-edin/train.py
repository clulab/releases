# from language import *
import argparse
import random
import pickle
import json

from model_dy import *
from bio_utils import *

from nltk.translate.bleu_score import corpus_bleu

if __name__ == '__main__':
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir') 
    parser.add_argument('embedding')   
    parser.add_argument('outdir')
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()

    if os.path.exists(args.datadir):
        input_lang, pl1, char, rule_lang, raw_train   = pickle.load(open('%s/train'%args.datadir, "rb"))
        input2_lang, pl2, char2, rule_lang2, raw_test = pickle.load(open('%s/dev'%args.datadir, "rb"))
    else:
        print ('Data file not found!')
        exit()

    
    embeds = load_embeddings(args.embedding, input_lang)
    model = LSTMLM(input_lang.n_words, char.n_words, 50, 50, 100, 200, pl1.n_words, 5, len(input_lang.labels), 
        100, 200, rule_lang.n_words, 200, 2, embeds)
    trainning_set = list()
    test = list()
    i = j = 0
    for datapoint in raw_train:
        if datapoint[3][0] != -1:
            i += len(datapoint[3])
            trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
                datapoint[1],#entity
                datapoint[2],#entity position
                datapoint[3],#trigger position
                [input_lang.label2id[l] for l in datapoint[4]],#trigger label
                [pl1.word2index[p] for p in datapoint[5]]+[0],#positions
                [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
                [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]]))
        else:
            j += 1
            trainning_set.append(([input_lang.word2index[w] for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], [0], 
                [pl1.word2index[p] for p in datapoint[5]]+[0],
                [[char.word2index[c] for c in w] for w in datapoint[0]+["EOS"]],
                [rule_lang.word2index["EOS"]]))
    print(i,j)
    i = j = 0
    for datapoint in raw_test:
        if datapoint[3][0] != -1:
            i += len(datapoint[3])
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],#entity
                datapoint[2],#entity position
                datapoint[3],#trigger position
                [input_lang.label2id[l] for l in datapoint[4]],#trigger label
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],#positions
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [[rule_lang.word2index[p] for p in rule + ["EOS"]] for rule in datapoint[6]]))
        else:
            j += 1
            test.append(([input_lang.word2index[w] if w in input_lang.word2index else 2 for w in datapoint[0]]+[1],
                datapoint[1],
                datapoint[2],
                datapoint[3], [0], 
                [pl1.word2index[p] if p in pl1.word2index else 2 for p in datapoint[5]]+[0],
                [[char.word2index[c] if c in char.word2index else 2 for c in w] for w in datapoint[0]+["EOS"]],
                [rule_lang.word2index["EOS"]]))
    print(i,j)
    for i in range(100):
        random.shuffle(trainning_set)
        model.train(trainning_set)
        if (i % 10) == 0 :
            predict = 0.0
            label_correct = 0.0
            trigger_correct = 0.0
            both_correct = 0.0
            tp = [0,0,0]
            s = [0,0,0]
            r = [0,0,0]
            references = [[],[],[]]
            candidates = [[],[],[]]
            for datapoint in test:
                triggers = datapoint[3]
                labels = datapoint[4]  
                rules = datapoint[-1]             
                sentence = datapoint[0]
                entity = datapoint[2]
                pos = datapoint[5]
                chars = datapoint[6]

                pred_triggers, score, contexts, hidden, pred_rules = model.get_pred(sentence, pos, chars, entity)
                if len(pred_triggers) != 0 or triggers[0] != -1:
                    if len(pred_triggers) != 0:
                        for t in pred_triggers:
                            s[t[1]-1] += 1
                        # s += len(pred_triggers)
                    if triggers[0] != -1:
                        for t in labels:
                            r[t-1] += 1
                        # r += len(triggers)
                    for k, t in enumerate(pred_triggers):
                        if t[0] in triggers and labels[triggers.index(t[0])] == t[1]:
                            tp[t[1]-1] += 1
                            j = triggers.index(t[0])
                            if rules[j][0] != 0:
                                references[t[1]-1].append([rules[j]])
                                candidates[t[1]-1].append(pred_rules[k])
            precision = [0,0,0]
            recall = [0,0,0]
            f1 = [0,0,0]
            for j in range(1):
                precision = tp[j]/s[j] if s[j]!= 0 else 0
                recall = tp[j]/r[j]
                f1 = 2*(precision*recall)/(recall+precision) if recall+precision != 0 else 0
                try:
                    bleu = corpus_bleu(references[j], candidates[j])
                except:
                    bleu = 0
                print ("%s Recall: %.4f Precision: %.4f F1: %.4f BLEU: %.4f"%(input_lang.labels[j+1], recall, precision, f1, bleu))
            
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
            model.save("%s/%d"%(args.outdir,i/10))

