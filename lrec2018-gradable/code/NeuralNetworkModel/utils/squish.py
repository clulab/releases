import pickle as pk
import sys
import torch
import math
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
#import torchtext.vocab as vocab
import torchwordemb
import numpy as np
import os
from itertools import accumulate, chain, repeat, tee
from utils.linearReg import convert_variable
from utils.read_write_data import readRawTurkDataFile
from sklearn.metrics import r2_score
from torch.autograd import Variable
from tqdm import tqdm
from utils.grounding import get_features_labels_from_data
from utils.grounding import get_features_dev
torch.manual_seed(1)

dense1_size=1

noOfFoldsCV=4
noOfEpochs=1000
learning_rate=1e-5
patience_max=5;

rsq_file="rsq_file.txt"
rsq_file_nfcv="rsq_file_nfcv.txt"
rsq_file_nfcv_avrg="rsq_file_nfcv_avrg.txt"
rsq_per_epoch_dev_four_chunks="rsq_per_epoch_dev_4_chunks.txt"

training_data="trainingData.csv"
#test_data="test.csv"
#test_data="test_no_random_seed.csv"
#test_data="test_rand_seed1.csv"
test_data="test_no_random_seed2.csv"

random_seed=1
useRandomSeed=True

class AdjEmb(nn.Module):
    def __init__(self,turkCount,addTurkerOneHot):
        super(AdjEmb,self).__init__()


        cwd=os.getcwd()
        path = cwd+"/data/"

        #load a subset of glove which contains embeddings for the adjectives we have
        self.vocab, self.vec = torchwordemb.load_glove_text(path+"glove_our_adj.csv")
        self.noOfTurkers=turkCount


        self.embeddings = nn.Embedding(self.vec.shape[0], self.vec.shape[1])
        self.embeddings.weight.data.copy_(self.vec)

        #dont update embeddings
        self.embeddings.weight.requires_grad=False


        self.linear1 = nn.Linear(self.vec.size(1), dense1_size)


        #the last step: whatever the output of previous layer was concatenate it with the mu and sigma and one-hot vector for turker
        if(addTurkerOneHot):
            self.fc = torch.nn.Linear(dense1_size+turkCount+2, 1)
        else:
            #use this when you dont have one hot for turkers
            self.fc = torch.nn.Linear(dense1_size+2, 1)



    #init was where you where just defining what embeddings meant. Here we actually use it
    def forward(self, adj, feats):

        emb=self.vec[self.vocab[adj]]#.numpy()
        embV=Variable(emb,requires_grad=False)
        out=F.tanh(self.linear1(embV))
        feature_squished = (torch.cat((feats, out)))
        retValue=(self.fc(feature_squished))
        return retValue


#take the adjectives and give it all an id
def convert_adj_index(listOfAdj):

    adj_Indices={}
    for eachAdj in tqdm(listOfAdj,total=len(listOfAdj),desc="adj_train:"):
        if eachAdj not in adj_Indices:
                    adj_Indices[eachAdj] = len(adj_Indices)
    return adj_Indices
    file_Name2 = "adj_Indices.pkl"
    # open the file for writing
    fileObject2 = open(file_Name2,'wb')
    pk.dump(adj_Indices, fileObject2)

def convert_scalar_to_variable(features):

    x2 =torch.from_numpy(np.array([features]))

    return Variable(x2)

def convert_to_variable(features):

    x2 =torch.from_numpy(features)

    return Variable(x2)

'''train on training data, print its rsquared against the same training data, then test with dev, print its rsquared. Do this at each epoch.
This is all done for tuning purposes'''
def  train_dev_print_rsq(dev,features, allY, list_Adj, all_adj,uniq_turker,addTurkerOneHot):
    #take the list of adjectives and give it all an index
    adj_index=convert_adj_index(list_Adj)

    #there are 193 unique turkers
    model=AdjEmb(193,addTurkerOneHot)


    params_to_update = filter(lambda p: p.requires_grad==True, model.parameters())
    rms = optim.RMSprop(params_to_update, learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    loss_fn = nn.MSELoss(size_average=True)

    allIndex = np.arange(len(features))

    cwd = os.getcwd()

    #empty out the existing file
    with open(cwd + "/outputs/" + rsq_file, "w+")as rsq_values:
        rsq_values.write("Epoch \t Train \t\t Dev \n")
    rsq_values.close()

    #append the rest of the values
    with open(cwd+"/outputs/" +rsq_file,"a")as rsq_values:
        for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):
            pred_y_total=[]
            y_total=[]
            adj_10_emb={}

            #shuffle for each epoch
            np.random.shuffle(allIndex)

            for eachRow in tqdm(allIndex, total=len(features), desc="each_adj:"):

                model.zero_grad()

                feature=features[eachRow]
                y = allY[eachRow]
                each_adj = all_adj[eachRow]

                featureV= convert_to_variable(feature)
                pred_y = model(each_adj, featureV)


                adj_10_emb[each_adj]=pred_y

                #the complete linear regression code- only thing is features here will include the squished_emb
                # Reset gradients

                true_variable_y = convert_scalar_to_variable(y)
                y_total.append(y)
                pred_y_total.append(pred_y.data.cpu().numpy()[0])
                loss_training = loss_fn(pred_y, true_variable_y)

                # Backward pass
                loss_training.backward()

                rms.step()


            rsquared_value_training=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
            rsq_values.write(str(epoch)+"\t"+str(rsquared_value_training)+"\t")
            rsq_values.flush()
            foundGoodModel=tuneOnDev(model,dev,cwd, uniq_turker,rsq_values,rsquared_value_training,loss_training,addTurkerOneHot,epoch)

            learned_weights = model.fc.weight.data

            if(epoch==120):
                # the model is trained by now-store it to disk
                file_Name122 = "adj_data_80-10-10-120-epochs.pkl"
                # open the file for writing
                fileObject122 = open(file_Name122, 'wb')
                pk.dump(model, fileObject122)




    #the model is trained by now-store it to disk
    file_Name_200 = "adj_data_80-10-10-145epochs.pkl"
    # open the file for writing
    fileObject_200 = open(file_Name_200,'wb')
    pk.dump(model, fileObject_200)

    return model


'''from: http://wordaligned.org/articles/slicing-a-list-evenly-with-python'''
def chunk(xs, n):
    assert n > 0
    L = len(xs)
    s, r = divmod(L, n)
    widths = chain(repeat(s+1, r), repeat(s, n-r))
    offsets = accumulate(chain((0,), widths))
    b, e = tee(offsets)
    next(e)
    return [xs[s] for s in map(slice, b, e)]


'''experiment: out of 4 chunks, keep one for testing, one for dev, and the rest two as training'''
def run_nfoldCV_on_turk_data_4chunks(features, allY, uniq_adj, all_adj,addTurkerOneHot,useEarlyStopping,use4Chunks):
    # shuffle before splitting
    if (useRandomSeed):
        np.random.seed(random_seed)



    allIndex = np.arange(len(features))
    print("str(len(features)):")
    print(str(len(features)))

    np.random.shuffle(allIndex)



    overall_pred_y=[]
    overall_gold_y=[]
    overall_adj=[]



    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)
    chunkIndices = np.arange(len(split_data))


    rsq_total=[]
    cwd=os.getcwd()

    # write rsq per chunk to disk
    with open(cwd + "/outputs/" + rsq_file_nfcv, "w+")as nfcv:
        #empty out the existing file before loop does append
        nfcv.write("Chunk \t RSQ\n")
        nfcv.close()

        # to write rsq per epoch  to disk
        # first empty out the existing file before loop does append
    with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "w+")as nfcv_four:
        nfcv_four.write("")
        nfcv_four.close()


    with open(cwd + "/outputs/" + rsq_file_nfcv, "a")as nfcv:

        # for each chunk in the training data, keep that one out, and train on the rest
        # append the rest of the values
        #note:test_fold_index starts at zero
        for test_fold_index in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):


            print("**************Starting next fold, fold number:"+str(test_fold_index)+" out of: "+str(len(chunkIndices))+"\n")

            model_4chunk = AdjEmb(193, addTurkerOneHot)

            params_to_update = filter(lambda p: p.requires_grad == True, model_4chunk.parameters())
            rms = optim.RMSprop(params_to_update, lr=learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            loss_fn = nn.MSELoss(size_average=True)

            dev_fold_index = (test_fold_index + 1) % 4

            # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
            tr_fold_indices = []
            for i in chunkIndices:
                if (i != test_fold_index and i != dev_fold_index):
                    tr_fold_indices.append(i)


            training_data=[]

            #for each of these left over chunks, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the training data
            for eachChunk in tr_fold_indices:
                for eachElement in split_data[eachChunk]:
                    training_data.append(eachElement)

            test_data=[]

            #for the left out test chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            for eachElement in split_data[test_fold_index]:
                    test_data.append(eachElement)


            # for the left out dev chunk, pull out its data points, and concatenate all into one single huge list of
            # data points-this is the test data
            dev_data = []
            for eachElement_dev in split_data[dev_fold_index]:
                dev_data.append(eachElement_dev)

            uniqAdj_dev={}
            uniqAdj_test={}
            uniqAdj_training={}
            for eachDev in dev_data:
                each_adj_tr = all_adj[eachDev]
                uniqAdj_dev[each_adj_tr] = uniqAdj_dev.get(each_adj_tr, 0) + 1

            for eachDev in test_data:
                each_adj_tr = all_adj[eachDev]
                uniqAdj_test[each_adj_tr] = uniqAdj_test.get(each_adj_tr, 0) + 1

            for eachDev in training_data:
                each_adj_tr = all_adj[eachDev]
                uniqAdj_training[each_adj_tr] = uniqAdj_training.get(each_adj_tr, 0) + 1


            for (k,v) in uniqAdj_dev.items():
                if k not in uniqAdj_training:
                   print("WARNING: " + k+" this adj from dev was not there in training")

            for (k,v) in uniqAdj_test.items():
                if k not in uniqAdj_training:
                   print("WARNING: " + k+" this adj from test was not there in training")



            #the patience counter starts from patience_max and decreases till it hits 0
            patienceCounter = patience_max


            #the best epochs per fold learned from tuning on dev=
            best_epochs= [231,899,990,983]

            #run n epochs on the left over training data
            with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "a")as nfcv_four:
                nfcv_four.write("test_fold_index:" + str(test_fold_index)+"\n")
                nfcv_four.write("dev_fold_index:"+str(dev_fold_index)+"\n")
                nfcv_four.write("tr_fold_indices:" + str(tr_fold_indices) + "\n")
                nfcv_four.write("Epoch \t RSQ_tr  \t RSQ_dev\n")

                '''found the best epochs per fold. after tuning on dev'''
                if(test_fold_index==0):
                    noOfEpochs=best_epochs[0]
                else:
                    if(test_fold_index==1):
                        noOfEpochs=best_epochs[1]
                    else:
                        if(test_fold_index==2):
                            noOfEpochs=best_epochs[2]
                        else:
                            if(test_fold_index==3):
                                noOfEpochs=best_epochs[3]


                for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                    y_total_tr_data=[]
                    pred_y_total_tr_data=[]

                    # shuffle before each epoch
                    np.random.shuffle(training_data)



                    '''for each row in the training data, predict y_test value for itself, and then back
                    propagate the loss'''
                    for each_data_item_index in tqdm(training_data, total=len(training_data), desc="trng_data_point:"):


                        #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                        model_4chunk.zero_grad()

                        feature=features[each_data_item_index]
                        y_test = allY[each_data_item_index]
                        each_adj_tr = all_adj[each_data_item_index]

                        featureV= convert_to_variable(feature)
                        pred_y_training = model_4chunk(each_adj_tr, featureV)

                        y_total_tr_data.append(y_test)
                        pred_y_total_tr_data.append(pred_y_training.data.cpu().numpy())


                        batch_y = convert_scalar_to_variable(y_test)

                        loss = loss_fn(pred_y_training, batch_y)


                        # Backward pass
                        loss.backward()

                        rms.step()


                    pred_y_total_dev_data = []
                    y_total_dev_data = []

            print("done with all epochs")



            #Testing phase
            # after all epochs in the given chunk, (i.e test once per fold)
            # for each element in the test data, calculate its predicted value, and append it to predy_total

            y_total_test_data=[]
            pred_y_total_test_data=[]

            for test_data_index in test_data:
                this_feature = features[test_data_index]
                featureV_dev= convert_to_variable(this_feature)
                y_test = allY[test_data_index]
                each_adj_test = all_adj[test_data_index]
                pred_y_test = model_4chunk(each_adj_test, featureV_dev)
                y_total_test_data.append(y_test)
                pred_y_total_test_data.append(pred_y_test.data.cpu().numpy())

                overall_gold_y.append(y_test)
                overall_pred_y.append(pred_y_test.data.cpu().numpy())
                overall_adj.append(each_adj_test)




            #calculate the rsquared value for this  held out
            rsquared_value_test=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
            print("\n")
            print("rsquared_value_on_test_after_chunk_"+str(test_fold_index)+":"+str(rsquared_value_test))
            print("\n")
            nfcv.write(str(test_fold_index) + "\t" + str(rsquared_value_test) + "\n")
            nfcv.flush()
            rsq_total.append(rsquared_value_test)


    #TO FIND AVERAGE OF gold and predicted values- using the overall method
        nfcv.write("\noverall_gold_y\toverall_pred_y\toverall_adj\n")
        for oindex,eachOverall in enumerate(overall_gold_y):
            nfcv.write(str(overall_gold_y[oindex])+"\t"+str(overall_pred_y[oindex])+"\t"+str(overall_adj[oindex])+"\n")
            nfcv.flush()

        rsquared_value_test_overalll=r2_score(overall_gold_y, overall_pred_y, sample_weight=None, multioutput='uniform_average')
        nfcv.write("overall rsq:"+str(rsquared_value_test_overalll)+"\n")
        nfcv.flush()

    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 10 such values,
    # each corresponding to one chunk being held out


    print("done with all chunks")

    rsq_cumulative=0;

    for eachRsq in rsq_total:
        rsq_cumulative=rsq_cumulative+eachRsq


    rsq_average=rsq_cumulative/(len(rsq_total))

    print("rsq_average:")
    print(str(rsq_average))

    # empty out the existing file
    with open(cwd + "/outputs/" + rsq_file_nfcv_avrg, "w+")as rsq_values_avg:
        rsq_values_avg.write("rsq_average: \t "+str(rsq_average))
    rsq_values_avg.close()



'''get all uniq adjectives. Divide into 4 chunks.: out of 4 chunks, keep one for testing,
 one for dev, and the rest two as training. Then pick it corresponding data points. DO cross validation'''
def nfoldCV_adj_grouped_turk_data_4chunks(raw_turk_data,features, allY, uniq_adj, all_adj,addTurkerOneHot):
    # shuffle before splitting
    if (useRandomSeed):
        np.random.seed(random_seed)


    #create an arange for length of total number of unique adjectives
    allIndex = np.arange(len(uniq_adj))


    np.random.shuffle(allIndex)






    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)

    #this is for cross validation. if there are 4 chunks, there will be four indices {0,1,2,3}
    chunkIndices = np.arange(len(split_data))





    rsq_total=[]
    cwd=os.getcwd()

    # push all unique adjectives into a list
    uniq_adj_list = []
    for k, v in uniq_adj.items():
        uniq_adj_list.append(k)

    overall_pred_y=[]
    overall_gold_y=[]
    overall_adj=[]


    # to write rsq per chunk to disk
    with open(cwd + "/outputs/" + rsq_file_nfcv, "w+")as nfcv:
        #empty out the existing file before loop does append
        nfcv.write("Chunk \t RSQ\n")
        nfcv.close()

        # to write rsq per epoch  to disk
        # first empty out the existing file before loop does append
    with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "w+")as nfcv_four:
        nfcv_four.write("")
        nfcv_four.close()




    with open(cwd + "/outputs/" + rsq_file_nfcv, "a")as nfcv:

        # for each chunk in the chunkIndices, keep that one out, and its neighbor becomes dev chunk. train on the rest

        #note:test_fold_index starts at zero
        for test_fold_index in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):

            print("**************Starting next fold, fold number:"+str(test_fold_index)+" out of: "+str(len(chunkIndices))+"\n")

            model_4chunk = AdjEmb(193, addTurkerOneHot)

            params_to_update = filter(lambda p: p.requires_grad == True, model_4chunk.parameters())
            rms = optim.RMSprop(params_to_update, lr=learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
            loss_fn = nn.MSELoss(size_average=True)

            dev_fold_index = (test_fold_index + 1) % 4

            # to get the indices of the folds/chunks we have in training:
            # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
            tr_fold_indices = []
            for i in chunkIndices:
                if (i != test_fold_index and i != dev_fold_index):
                    tr_fold_indices.append(i)





            #to get the list of adj ids which are present in each data chunk
            adjs_ids_training_data=[]
            adjs_id_test_data = []
            adjs_ids_dev_data = []


            #in each of these  chunks, pull out the list of adjectives and assign it to tr,
            # dev and test folds. note that these are just the indices of adjectives, not the actual data


            #string values of each of the adjectives in each fold
            trainingData_adj_str = []
            dev_adj_str = []
            test_adj_str = []


            #since there are more than one chunks in training, for each of the chunk id:
            for eachChunk2 in tr_fold_indices:
                for eachElement in split_data[eachChunk2]:
                    adjs_ids_training_data.append(eachElement)
                    #also get teh str value of the adjective to keep
                    trainingData_adj_str.append(uniq_adj_list[eachElement])





            for eachElement3 in split_data[dev_fold_index]:
                adjs_ids_dev_data.append(eachElement3)
                #also get teh str value of the adjective to keep
                dev_adj_str.append(uniq_adj_list[eachElement3])






            for eachElement4 in split_data[test_fold_index]:
                adjs_id_test_data.append(eachElement4)
                #also get teh str value of the adjective to keep
                test_adj_str.append(uniq_adj_list[eachElement4])


            #this where the actual turk data is divided.
            # go through the indices of the entire data and assign the indicies based on which fold its adjective falls into.
            #so if the turk data which has index 1234 has an adjective which is in training fold, assign it to training fold
            #then all you have to do is hand it over to the NFCV code or seen data

            training_data = []
            test_data = []
            dev_data = []



            #for each adjective in the training fold.
            for each_tr_adj in tqdm(trainingData_adj_str,total=len(trainingData_adj_str), desc="trainingData_adj_str:"):

                # Go through the raw data- if you find a line which has the same adjective, add its index to training_data
                for index, eachline in raw_turk_data.iterrows():
                    thisadj = eachline['adjective']


                    if (thisadj == each_tr_adj):
                        training_data.append(index)

            # for each adjective in the dev fold.
            for each_dev_adj in tqdm(dev_adj_str, total=len(dev_adj_str),
                                    desc="dev_adj_str:"):

                # Go through the raw data- if you find a line which has the same adjective, add its index to training_data
                for index, eachline in raw_turk_data.iterrows():
                    thisadj = eachline['adjective']


                    if (thisadj == each_dev_adj):
                        dev_data.append(index)

            # for each adjective in the test fold.
            for each_test_adj in tqdm(test_adj_str, total=len(test_adj_str),
                                     desc="test_adj_str:"):

                # Go through the raw data- if you find a line which has the same adjective, add its index to training_data
                for index, eachline in raw_turk_data.iterrows():
                    thisadj = eachline['adjective']


                    if (thisadj == each_test_adj):
                        test_data.append(index)



            #run n epochs on the  training data
            with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "a")as nfcv_four:
                nfcv_four.write("test_fold_index:" + str(test_fold_index)+"\n")
                nfcv_four.write("dev_fold_index:"+str(dev_fold_index)+"\n")
                nfcv_four.write("tr_fold_indices:" + str(tr_fold_indices) + "\n")
                nfcv_four.write("Epoch \t RSQ_tr  \t RSQ_dev\n")

                # '''this is to be used after dev tunin.
                # found the best epochs per fold. after tuning on dev'''
                if(test_fold_index==0):
                    noOfEpochs=214

                else:
                    if(test_fold_index==1):
                        noOfEpochs=405
                    else:
                        if(test_fold_index==2):
                            noOfEpochs=433
                        else:
                            if(test_fold_index==3):
                                noOfEpochs=748

                for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

                    y_total_tr_data=[]
                    pred_y_total_tr_data=[]

                    #shuffle before each epoch
                    np.random.shuffle(training_data)

                    '''for each row in the training data, predict y_test value for itself, and then back
                    propagate the loss'''
                    for each_data_item_index in tqdm((training_data), total=len(training_data), desc="trng_data_point:"):


                        #every time you feed forward, make sure the gradients are emptied out. From pytorch documentation
                        model_4chunk.zero_grad()


                        #print("each_data_item_index:"+str(each_data_item_index))
                        feature=features[each_data_item_index]
                        y_test = allY[each_data_item_index]
                        each_adj_tr = all_adj[each_data_item_index]


                        featureV= convert_to_variable(feature)
                        pred_y_training = model_4chunk(each_adj_tr, featureV)

                        y_total_tr_data.append(y_test)
                        pred_y_total_tr_data.append(pred_y_training.data.cpu().numpy())


                        batch_y = convert_scalar_to_variable(y_test)

                        loss = loss_fn(pred_y_training, batch_y)


                        # Backward pass
                        loss.backward()

                        rms.step()



                    #after every epoch, i.e after training on n data points,-calculate rsq for trainign also
                    rsquared_value_tr = r2_score(y_total_tr_data, pred_y_total_tr_data, sample_weight=None,
                                              multioutput='uniform_average')

                    # #after every epoch, i.e after training on n data points,
                    #  run on dev data and calculate rsq

                    pred_y_total_dev_data = []
                    y_total_dev_data = []







            print("done with all epochs- i.e done with all training and devs")
            #  #the model is trained by now-store it to disk
            file_Name55 = str(test_fold_index)+"nfcv_group_by_adj.pkl"
            # open the file for writing
            fileObject55 = open(file_Name55,'wb')
            pk.dump(model_4chunk, fileObject55)



            #Testing phase
            # after all epochs in the given chunk, (i.e test once per fold)
            # for each element in the test data, calculate its predicted value, and append it to predy_total

            y_total_test_data=[]
            pred_y_total_test_data=[]


            print("starting testing ")
            adj_gold_pred={}
            previous_adj=""
            this_adj_gold_y=[]
            this_adj_pred_y=[]

            data_point_per_adj=0

            for indext,test_data_index in enumerate(test_data):
                this_feature = features[test_data_index]
                featureV_dev= convert_to_variable(this_feature)
                y_test = allY[test_data_index]
                each_adj_test = all_adj[test_data_index]

                pred_y_test = model_4chunk(each_adj_test, featureV_dev)
                y_total_test_data.append(y_test)
                pred_y_total_test_data.append(pred_y_test.data.cpu().numpy())

                overall_gold_y.append(y_test)
                overall_pred_y.append(pred_y_test.data.cpu().numpy())
                overall_adj.append(each_adj_test)

                #calculate rsq values per adjective
                #for each data point which has the same adjective, store its goldY and predY values
                #note: there is an assumption here that data points of all adjectives are together. might bite you soon.
                current_adj=each_adj_test




                #very first time initialize the previous_adj=current_adj
                if(indext==0):
                    previous_adj=current_adj
                    data_point_per_adj=data_point_per_adj+1

                #append to the value the tuple of [gold, predicted] if exists
                if each_adj_test in adj_gold_pred:
                    adj_gold_pred[each_adj_test] += [y_test,pred_y_test.data.cpu().numpy()]
                else:
                    adj_gold_pred[each_adj_test] = [y_test,pred_y_test.data.cpu().numpy()]


                if(current_adj==previous_adj):
                    data_point_per_adj=data_point_per_adj+1
                    this_adj_gold_y.append(y_test)
                    this_adj_pred_y.append(pred_y_test.data.cpu().numpy()[0])


                #if the adjectives are different, it means that we are switching to a new one. calculate rsquared. update previous_adj
                else:
                    data_point_per_adj=0
                    previous_adj=current_adj
                    rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')

                    #print("adj:"+current_adj+" rsq:"+str(rsquared_value_per_adj)+" len:"+str(data_point_per_adj))
                    nfcv.write("adj:"+current_adj+" rsq:"+str(rsquared_value_per_adj)+" len:"+str(len(this_adj_gold_y))+"\n")
                    nfcv.flush()
                    this_adj_gold_y=[]
                    this_adj_pred_y=[]

                if(indext==(len(test_data)-1)):
                    rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')
                    nfcv.write("adj:"+current_adj+" rsq:"+str(rsquared_value_per_adj)+" len:"+str(len(this_adj_gold_y))+"\n")
                    nfcv.flush()
                    this_adj_gold_y=[]
                    this_adj_pred_y=[]





            #calculate the rsquared value for this  held out
            rsquared_value_test=r2_score(y_total_test_data, pred_y_total_test_data, sample_weight=None, multioutput='uniform_average')
            print("\n")
            print("rsquared_value_on_test_after_chunk_"+str(test_fold_index)+":"+str(rsquared_value_test))
            print("len(test_fold_index)"+str(len(test_data)))
            print("\n")
            nfcv.write("\n")
            nfcv.write("rsquared_value_on_test_after_chunk_"+str(test_fold_index)+":"+str(rsquared_value_test)+"\n")
            nfcv.write("len(test_fold_index): "+str(len(test_data)))
            nfcv.flush()
            rsq_total.append(rsquared_value_test)



        #TO FIND AVERAGE OF overall gold and predicted values
        nfcv.write("\noverall_gold_y\toverall_pred_y\toverall_adj\n")
        for oindex,eachOverall in enumerate(overall_gold_y):
            nfcv.write(str(overall_gold_y[oindex])+"\t"+str(overall_pred_y[oindex])+"\t"+str(overall_adj[oindex])+"\n")
            nfcv.flush()

        rsquared_value_test_overalll=r2_score(overall_gold_y, overall_pred_y, sample_weight=None, multioutput='uniform_average')
        nfcv.write(str(rsquared_value_test_overalll)+"\n")
        nfcv.flush()

    #  After all chunks are done, calculate the average of each element in the list of predicted rsquared values.
    # There should be 10 such values,
    # each corresponding to one chunk being held out
    print("done with all chunks")

    rsq_cumulative=0;

    for eachRsq in rsq_total:
        rsq_cumulative=rsq_cumulative+eachRsq


    rsq_average=rsq_cumulative/(len(rsq_total))

    print("rsq_average:")
    print(str(rsq_average))

    # empty out the existing file
    with open(cwd + "/outputs/" + rsq_file_nfcv_avrg, "w+")as rsq_values_avg:
        rsq_values_avg.write("rsq_average: \t "+str(rsq_average))
    rsq_values_avg.close()


def load_nfoldCV_adj_grouped_turk_data_4chunks(raw_turk_data,features, allY, uniq_adj, all_adj,addTurkerOneHot):
    # shuffle before splitting
    if (useRandomSeed):
        np.random.seed(random_seed)


    #create an arange for length of total number of unique adjectives
    allIndex = np.arange(len(uniq_adj))

    np.random.shuffle(allIndex)


    #split it into folds. n=number of folds. almost even sized.
    n=noOfFoldsCV
    split_data=chunk(allIndex,n)

    #this is for cross validation. if there are 4 chunks, there will be four indices {0,1,2,3}
    chunkIndices = np.arange(len(split_data))



    cwd=os.getcwd()

    # push all unique adjectives into a list
    uniq_adj_list = []
    for k, v in uniq_adj.items():
        uniq_adj_list.append(k)

    overall_pred_y=[]
    overall_gold_y=[]
    overall_adj=[]


    # to write rsq per chunk to disk
    with open(cwd + "/outputs/" + rsq_file_nfcv, "w+")as nfcv:
        #empty out the existing file before loop does append
        nfcv.write("Chunk \t RSQ\n")
        nfcv.close()

        # to write rsq per epoch  to disk
        # first empty out the existing file before loop does append
    with open(cwd + "/outputs/" + rsq_per_epoch_dev_four_chunks, "w+")as nfcv_four:
        nfcv_four.write("")
        nfcv_four.close()




    with open(cwd + "/outputs/" + rsq_file_nfcv, "a")as nfcv:

        # for each chunk in the chunkIndices, keep that one out, and its neighbor becomes dev chunk. train on the rest

        #note:test_fold_index starts at zero
        for test_fold_index in tqdm(chunkIndices,total=len(chunkIndices), desc="n-fold-CV:"):


                print("**************Starting next fold, fold number:"+str(test_fold_index)+" out of: "+str(len(chunkIndices))+"\n")

                model_4chunk = AdjEmb(193, addTurkerOneHot)

                params_to_update = filter(lambda p: p.requires_grad == True, model_4chunk.parameters())
                rms = optim.RMSprop(params_to_update, lr=learning_rate, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
                loss_fn = nn.MSELoss(size_average=True)

                dev_fold_index = (test_fold_index + 1) % 4

                # to get the indices of the folds/chunks we have in training:
                # create a  list of all the indices of chunks except the test and dev chunk you are keeping out
                tr_fold_indices = []
                for i in chunkIndices:
                    if (i != test_fold_index and i != dev_fold_index):
                        tr_fold_indices.append(i)




                #to get the list of adj ids which are present in each data chunk
                adjs_ids_training_data=[]
                adjs_id_test_data = []
                adjs_ids_dev_data = []

              #in each of these  chunks, pull out the list of adjectives and assign it to tr,
                # dev and test folds. note that these are just the indices of adjectives, not the actual data


                #string values of each of the adjectives in each fold
                trainingData_adj_str = []
                dev_adj_str = []
                test_adj_str = []


                #since there are more than one chunks in training, for each of the chunk id:
                for eachChunk2 in tr_fold_indices:
                    for eachElement in split_data[eachChunk2]:
                        adjs_ids_training_data.append(eachElement)
                        #also get teh str value of the adjective to keep
                        trainingData_adj_str.append(uniq_adj_list[eachElement])





                for eachElement3 in split_data[dev_fold_index]:
                    adjs_ids_dev_data.append(eachElement3)
                    #also get teh str value of the adjective to keep
                    dev_adj_str.append(uniq_adj_list[eachElement3])






                for eachElement4 in split_data[test_fold_index]:
                    adjs_id_test_data.append(eachElement4)
                    #also get teh str value of the adjective to keep
                    test_adj_str.append(uniq_adj_list[eachElement4])






                #this where the actual turk data is divided.
                # go through the indices of the entire data and assign the indicies based on which fold its adjective falls into.
                #so if the turk data which has index 1234 has an adjective which is in training fold, assign it to training fold
                #then all you have to do is hand it over to the NFCV code or seen data

                training_data = []
                test_data = []
                dev_data = []


                # for each adjective in the test fold.
                for each_test_adj in tqdm(test_adj_str, total=len(test_adj_str),
                                         desc="test_adj_str:"):

                    # Go through the raw data- if you find a line which has the same adjective, add its index to training_data
                    for index, eachline in raw_turk_data.iterrows():
                        thisadj = eachline['adjective']

                        # results = [eachline["turker"], eachline["adjective"], eachline["mean"],
                        #            eachline["onestdev"],
                        #            eachline["had_negative"], eachline["logrespdev"]]

                        if (thisadj == each_test_adj):
                            test_data.append(index)

                #for each fold load the corresponding trained model and run test
                file_Name55 = str(test_fold_index)+"nfcv_group_by_adj.pkl"
                trained_model_nfcv = pk.load(open(file_Name55, "rb"))



                #Testing phase
                # after all epochs in the given chunk, (i.e test once per fold)
                # for each element in the test data, calculate its predicted value, and append it to predy_total

                y_total_test_data=[]
                pred_y_total_test_data=[]


                print("starting testing ")
                previous_adj=""
                current_adj=""
                adj_gold_pred={}
                previous_adj=""
                current_adj=""
                this_adj_gold_y=[]
                this_adj_pred_y=[]

                data_point_per_adj=0

                for indext,test_data_index in enumerate(test_data):
                    this_feature = features[test_data_index]
                    featureV_dev= convert_to_variable(this_feature)
                    y_test = allY[test_data_index]
                    each_adj_test = all_adj[test_data_index]

                    pred_y_test = trained_model_nfcv(each_adj_test, featureV_dev)
                    y_total_test_data.append(y_test)
                    pred_y_total_test_data.append(pred_y_test.data.cpu().numpy())

                    overall_gold_y.append(y_test)
                    overall_pred_y.append(pred_y_test.data.cpu().numpy())
                    overall_adj.append(each_adj_test)


                    #below code is to calculate rsq values per adjective
                    #for each data point which has the same adjective, store its goldY and predY values
                    #note: there is an assumption here that data points of all adjectives are together. might bite you soon.
                    current_adj=each_adj_test




                    #very first time initialize the previous_adj=current_adj
                    if(indext==0):
                        previous_adj=current_adj
                        data_point_per_adj=data_point_per_adj+1

                    #append to the value the tuple of [gold, predicted] if exists
                    if each_adj_test in adj_gold_pred:
                        adj_gold_pred[each_adj_test] += [y_test,pred_y_test.data.cpu().numpy()]
                    else:
                        adj_gold_pred[each_adj_test] = [y_test,pred_y_test.data.cpu().numpy()]


                    if(current_adj==previous_adj):
                        data_point_per_adj=data_point_per_adj+1
                        this_adj_gold_y.append(y_test)
                        this_adj_pred_y.append(pred_y_test.data.cpu().numpy()[0])


                    #if the adjectives are different, it means that we are switching to a new one. calculate rsquared. update previous_adj
                    else:
                        data_point_per_adj=0
                        previous_adj=current_adj
                        rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')

                        nfcv.write("adj:"+current_adj+" rsq:"+str(rsquared_value_per_adj)+" len:"+str(len(this_adj_gold_y))+"\n")
                        nfcv.flush()
                        this_adj_gold_y=[]
                        this_adj_pred_y=[]

                    if(indext==(len(test_data)-1)):
                        rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')
                        nfcv.write("adj:"+current_adj+" rsq:"+str(rsquared_value_per_adj)+" len:"+str(len(this_adj_gold_y))+"\n")
                        nfcv.flush()


        #TO FIND AVERAGE OF overall gold and predicted values
        nfcv.write("\noverall_gold_y\toverall_pred_y\toverall_adj\n")
        for oindex,eachOverall in enumerate(overall_gold_y):
            nfcv.write(str(overall_gold_y[oindex])+"\t"+str(overall_pred_y[oindex])+"\t"+str(overall_adj[oindex])+"\n")
            nfcv.flush()

        rsquared_value_test_overalll=r2_score(overall_gold_y, overall_pred_y, sample_weight=None, multioutput='uniform_average')
        nfcv.write(str(rsquared_value_test_overalll)+"\n")
        nfcv.flush()



def predictAndCalculateRSq(allY, features, all_adj, trained_model,epoch):
    pred_y_total = []
    y_total = []



    loss_fn = nn.MSELoss(size_average=True)

    adj_gold_pred={}
    previous_adj=""
    current_adj=""
    this_adj_gold_y=[]
    this_adj_pred_y=[]

    for index,feature in tqdm(enumerate(features), total=len(features), desc="predict:"):


            featureV= convert_to_variable(feature)
            y = allY[index]
            each_adj = all_adj[index]
            pred_y = trained_model(each_adj, featureV)
            y_total.append(y)
            pred_y_total.append(pred_y.data.cpu().numpy()[0])

            #below code is to calculate rsq values per adjective
            #for each data point which has the same adjective, store its goldY and predY values
            current_adj=each_adj

            #very first time initialize the previous_adj=current_adj
            if(index==0):
                previous_adj=current_adj
                # print("foujnd that index==0")

            if(current_adj==previous_adj):
                this_adj_gold_y.append(y)
                this_adj_pred_y.append(pred_y.data.cpu().numpy()[0])


            #if the adjectives are different, it means that we are switching to a new one. calculate rsquared. update previous_adj
            else:
                previous_adj=current_adj
                rsquared_value_per_adj=r2_score(this_adj_gold_y, this_adj_pred_y, sample_weight=None, multioutput='uniform_average')


    rsquared_value=r2_score(y_total, pred_y_total, sample_weight=None, multioutput='uniform_average')
    return rsquared_value


def tuneOnDev(trained_model,dev,cwd, uniq_turker,rsq_values,rsquared_value_training,loss_training,addTurkerOneHot,epoch):
    # test on dev data
    features, y, adj_lexicon, all_adj = get_features_dev(cwd, dev, False, uniq_turker,addTurkerOneHot)

    # calculate rsquared
    rsquared_dev_value = predictAndCalculateRSq(y, features, all_adj, trained_model,epoch)

    rsq_values.write(str(rsquared_dev_value)+"\n")
    rsq_values.flush()


def runOnTestPartition(trained_model,dev,cwd, uniq_turker,rsq_values,addTurkerOneHot,epoch):
    # read the test
    features, y, adj_lexicon, all_adj = get_features_dev(cwd, dev, False, uniq_turker,addTurkerOneHot)
    print("done reading test data:")

    # calculate rsquared
    rsquared_test_value = predictAndCalculateRSq(y, features, all_adj, trained_model,epoch)


    print("rsquared_value_on_test:\n")
    print(str(rsquared_test_value))
    print("")
    rsq_values.write(str(rsquared_test_value)+"\n")
    rsq_values.flush()

