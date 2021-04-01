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
import pickle


python_rand_seed = int(sys.argv[1])


print('python random seed:', python_rand_seed)



#python_rand_seed=65535
random.seed(python_rand_seed)
np.random.seed(python_rand_seed)
dy_conf.set(random_seed=python_rand_seed)

import dynet as dy
from utils import *
from rnn import *

def main(path_train, path_test_con, path_test_op):
    #embeddings = w2v.load_embeddings("/lhome/zhengzhongliang/CLU_Projects/2018_Automated_Scientific_Discovery_Framework/polarity/20181015/w2v/pubmed/medPubDict.pkl.gz")
    embeddings = w2v.load_embeddings("/Users/zhengzhongliang/NLP_Research/2019_ASDF/medPubDict.pkl.gz")
    #embeddings = w2v.load_embeddings("/work/zhengzhongliang/ASDF_Github/2019_polarity/medPubDict.pkl.gz")
        

    with open(path_train) as f:
        reader = csv.DictReader(f)
        data_train = list(reader)
    instances_train = [Instance.from_dict(d) for d in data_train]

    with open(path_test_con) as f:
        reader = csv.DictReader(f)
        data_test_con = list(reader)
    instances_test_con_ = [Instance.from_dict(d) for d in data_test_con]
    instances_test_con = list([])
    for instance in instances_test_con_:
        if instance.polarity!=2:
            instances_test_con.append(instance)

    with open(path_test_op) as f:
        reader = csv.DictReader(f)
        data_test_op = list(reader)
    instances_test_op_ = [Instance.from_dict(d) for d in data_test_op]
    instances_test_op = list([])
    for instance in instances_test_op_:
        if instance.polarity!=2:
            instances_test_op.append(instance)

    print('training:', len(instances_train), '  test con:', len(instances_test_con), '  test op:', len(instances_test_op))

    char_embeddings = build_char_dict(instances_train)

    # Shuffle the training instances
    random.Random(python_rand_seed).shuffle(instances_train)
    labels_train = [instance.polarity for instance in instances_train]
    labels_test_con = [instance.polarity for instance in instances_test_con]
    labels_test_op = [instance.polarity for instance in instances_test_op]

    reach_labels_con = [1 if instance.pred_polarity else 0 for instance in instances_test_con]
    reach_labels_op = [1 if instance.pred_polarity else 0 for instance in instances_test_op]

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
    
    epochs = 10
    f1_results = np.zeros((epochs, 12))
    
    element = build_model(embeddings, char_embeddings)
    embeddings_index = WordEmbeddingIndex(element.w2v_emb, embeddings)
    embeddings_char_index = CharEmbeddingIndex(element.c2v_embd, char_embeddings)
    params = element.param_collection
    trainer= dy.AdamTrainer(params)
    trainer.set_clip_threshold(4.0)

    for e in range(epochs):
    
        training_losses = list()
        bad_grad_count = 0
        
            
        lstm_labels_train = list()
        for i, instance in enumerate(instances_train):
            prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)
            y_pred = 1 if prediction.value() >= 0.5 else 0
            lstm_labels_train.append(y_pred)

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

        lstm_labels_con = list()
        for i, instance in enumerate(instances_test_con):
            prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)
            y_pred = 1 if prediction.value() >= 0.5 else 0
            lstm_labels_con.append(y_pred)

        lstm_labels_op = list()
        for i, instance in enumerate(instances_test_op):
            prediction = run_instance(instance, element, embeddings_index, embeddings_char_index)
            y_pred = 1 if prediction.value() >= 0.5 else 0
            lstm_labels_op.append(y_pred)

            
        trainer.learning_rate = trainer.learning_rate*0.9
            
        print('===================================================================')
        print('number of bad grads:', bad_grad_count)
        print("Epoch %i average training loss: %f" % (e+1, np.average(training_losses)))

        print('training f1:', f1_score(labels_train, lstm_labels_train))
        lstm_f1_con = f1_score(labels_test_con, lstm_labels_con)
        lstm_precision_con = precision_score(labels_test_con, lstm_labels_con)
        lstm_recall_con = recall_score(labels_test_con, lstm_labels_con)

        lstm_f1_op = f1_score(labels_test_op, lstm_labels_op)
        lstm_precision_op = precision_score(labels_test_op, lstm_labels_op)
        lstm_recall_op = recall_score(labels_test_op, lstm_labels_op)

        reach_f1_con = f1_score(labels_test_con, reach_labels_con)
        reach_precision_con = precision_score(labels_test_con, reach_labels_con)
        reach_recall_con = recall_score(labels_test_con, reach_labels_con)

        reach_f1_op = f1_score(labels_test_op, reach_labels_op)
        reach_precision_op = precision_score(labels_test_op, reach_labels_op)
        reach_recall_op = recall_score(labels_test_op, reach_labels_op)

        print('lstm con f1:', lstm_f1_con, '    lstm op f1:', lstm_f1_op)
        print('reach con f1:', reach_f1_con, '    reach op f1:', reach_f1_op)

        f1_results[e,:] = [lstm_f1_con, lstm_precision_con, lstm_recall_con, lstm_f1_op, lstm_precision_op, lstm_recall_op, reach_f1_con, reach_precision_con, reach_recall_con, reach_f1_op, reach_precision_op, reach_recall_op]

        if e==0:
            print(labels_test_con, lstm_labels_con)
            print(labels_test_op, lstm_labels_op)
        if e==epochs-1:
            labels_list = list([])
            labels_list.append(labels_test_con)
            labels_list.append(lstm_labels_con)
            labels_list.append(labels_test_op)
            labels_list.append(lstm_labels_op)
            file_name = 'Result/f1_score_seed_'+str(python_rand_seed)+'_biLSTM_labels.pkl'
            with open(file_name, 'wb') as f:
                pickle.dump(labels_list, f)
                
    file_name = 'Result/f1_score_seed_'+str(python_rand_seed)+'_biLSTM.csv'
    np.savetxt(file_name, f1_results, delimiter=',')

    #params.save("model.dy")


if __name__ == "__main__":
    main("SentencesInfo_all_label_final_ExactRecur_train.csv", "SentencesInfo_con_label_final_ExactRecur_test.csv", "SentencesInfo_op_label_final_ExactRecur_test.csv",)
