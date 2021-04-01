import csv
import random
import w2v
import itertools as it
import numpy as np
import dynet_config as dy_conf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import sys

python_rand_seed = int(sys.argv[1])
word_embd_sel = int(sys.argv[2])
char_embd_sel = int(sys.argv[3])

print('python random seed:', python_rand_seed)
print('word embd:', word_embd_sel)
print('char embd:', char_embd_sel)


#python_rand_seed=65535
random.seed(python_rand_seed)
np.random.seed(python_rand_seed)
dy_conf.set(random_seed=python_rand_seed)

import dynet as dy
from utils import *
from rnn import *

def main(input_path):
    with open(input_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    #embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")
    embeddings = w2v.load_embeddings("/Users/zhengzhongliang/NLP_Research/2019_ASDF/medPubDict.pkl.gz")


    print("There are %i rows" % len(data))

    instances = [Instance.from_dict(d) for d in data]

    char_embeddings = build_char_dict(instances)



        
    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances)
    labels = [1 if instance.polarity else 0 for instance in instances]

    print("There are %i instances" % len(instances))

    # char_embd_choices = {'no-char-embd':0, 'biGRU-char-embd':1}
    # char_embd_sel = char_embd_choices['biGRU-char-embd']
    # word_embd_choices = {'no-med-pub':0,'med-pub':1}
    # word_embd_sel = word_embd_choices['no-med-pub']


    
    # Store the vocabulary of the missing words (from the pre-trained embeddings)
#    with open("w2v_vocab.txt", "w") as f:
#        for w in embeddings_index.w2v_index.to_list():
#            f.write(w + "\n")

    # Training loop
    #trainer = dy.SimpleSGDTrainer(params, learning_rate=0.005)

    # use this to test whether a smaller learning rate can boost the performance of pre-trained models. delete
    # this line when generating formal results.
    # trainer.learning_rate = trainer.learning_rate*0.5
    
    # split data and do cross-validation
    skf = StratifiedKFold(n_splits=5)
    
    elements = {}
    embeddings_indices = {}
    embeddings_char_indices = {}
    trainers = {}
    params = {}
    
    epochs = 10
    f1_results = np.zeros((epochs, 6))
    for i in range(5):
        elements[i] = build_model(embeddings, char_embeddings, word_embd_sel, char_embd_sel)
        embeddings_indices[i] = WordEmbeddingIndex(elements[i].w2v_emb, embeddings)
        embeddings_char_indices[i] = CharEmbeddingIndex(elements[i].c2v_embd, char_embeddings)
        params[i] = elements[i].param_collection
        trainers[i] = dy.AdamTrainer(params[i])
        trainers[i].set_clip_threshold(4.0)

    for e in range(epochs):
    

        
        test_pred_dict = {}
        test_label_dict = {}
        test_loss_dict = {}
        test_reach_pred_dict={}
    
        training_losses = list()
        bad_grad_count = 0
        
        
        for m_index, (train_indices, test_indices) in enumerate(skf.split(instances, labels)):
            element = elements[m_index]
            embeddings_index = embeddings_indices[m_index]
            embeddings_char_index = embeddings_char_indices[m_index]
            trainer = trainers[m_index]
            
            W_np = element.W.npvalue()
            print('W sum:',np.sum(W_np), 'W std:',np.std(W_np))
            print('learning rate:',trainer.learning_rate)
            
            for i, sample_index in enumerate(train_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, element, embeddings_index, embeddings_char_index, char_embd_sel)

                loss = prediction_loss(instance, prediction)

                loss.backward()
                try:
                    trainer.update()
                except RuntimeError:
                    #print('encountered bad gradient, instance skipped.')
                    bad_grad_count+=1
                loss_value = loss.value()
                training_losses.append(loss_value)

            # Now do testing

            # testing_losses = list()
            # testing_predictions = list()
            # testing_labels = [1 if instances[index].polarity else 0 for index in test_indices]
            
            fold_preds = list([])
            fold_labels = list([])
            for i, sample_index in enumerate(test_indices):
                instance = instances[sample_index]
                prediction = run_instance(instance, element, embeddings_index, embeddings_char_index, char_embd_sel)
                y_pred = 1 if prediction.value() >= 0.5 else 0
                loss = prediction_loss(instance, prediction)
                loss_value = loss.value()
                
                if instance.neg_count not in test_pred_dict:
                    test_pred_dict[instance.neg_count]=list([])
                    test_label_dict[instance.neg_count]=list([])
                    test_loss_dict[instance.neg_count]=list([])
                    test_reach_pred_dict[instance.neg_count]=list([])
                    
                test_pred_dict[instance.neg_count].append(y_pred)
                test_label_dict[instance.neg_count].append([1 if instance.polarity else 0])
                test_loss_dict[instance.neg_count].append(loss_value)
                test_reach_pred_dict[instance.neg_count].append([1 if instance.pred_polarity else 0])
            trainer.learning_rate = trainer.learning_rate*0.1
            
        print('===================================================================')
        print('number of bad grads:', bad_grad_count)
        print("Epoch %i average training loss: %f" % (e+1, np.average(training_losses)))
        
        print('---------------LSTM result------------------------- ')
        all_pred = list([])
        all_label = list([])
        for neg_count in test_pred_dict.keys():
            f1 = f1_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            precision = precision_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            recall = recall_score(test_label_dict[neg_count], test_pred_dict[neg_count])
            print("Neg Count: %d\tN Samples: %d\tPrecision: %f\tRecall: %f\tF1: %f" % (neg_count, len(test_pred_dict[neg_count]), precision, recall, f1))
            all_pred.extend(test_pred_dict[neg_count])
            all_label.extend(test_label_dict[neg_count])
        all_f1 = f1_score(all_label, all_pred)
        all_recall = recall_score(all_label, all_pred)
        all_precision = precision_score(all_label, all_pred)

        f1_results[e,0:3] = [all_f1, all_recall, all_precision]
        print('overall f1:', all_f1)
        
        print('---------------REACH result------------------------- ')
        all_pred = list([])
        all_label = list([])
        for neg_count in test_pred_dict.keys():
            f1 = f1_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            precision = precision_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            recall = recall_score(test_label_dict[neg_count], test_reach_pred_dict[neg_count])
            print("Neg Count: %d\tN Samples: %d\tPrecision: %f\tRecall: %f\tF1: %f" % (neg_count, len(test_pred_dict[neg_count]), precision, recall, f1))
            all_pred.extend(test_reach_pred_dict[neg_count])
            all_label.extend(test_label_dict[neg_count])
        all_f1 = f1_score(all_label, all_pred)
        all_recall = recall_score(all_label, all_pred)
        all_precision = precision_score(all_label, all_pred)
        f1_results[e,3:6] = [all_f1, all_recall, all_precision]

        print('overall f1:', all_f1)
            
#            if sum(testing_predictions) >= 1:
#                report = classification_report(testing_labels, testing_predictions)
#                #print(report)
#            if avg_loss <= 3e-3:
#                break
#            print()

    file_name = 'Result/f1_score_seed_'+str(python_rand_seed)+'_wordEmbd_'+str(word_embd_sel)+'_charEmbd_'+str(char_embd_sel)+'.csv'
    np.savetxt(file_name, f1_results, delimiter=',')

    #params.save("model.dy")


if __name__ == "__main__":
    main("SentencesInfo_all_label_final.csv")
