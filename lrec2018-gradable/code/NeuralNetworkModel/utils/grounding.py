import math
import numpy as np
import sys
#import torchtext.vocab as vocab
import torchwordemb
from tqdm import tqdm

from utils.linearReg import runLR
from utils.read_write_data import loadEmbeddings
from utils.read_write_data import readRawTurkDataFile
from utils.read_write_data import writeCsvToFile
from utils.read_write_data import writeToFileWithHeader
from utils.read_write_data import writeToFileWithPd

cbow4 = "glove_vectors_syn_ant_sameord_difford.txt"
random_seed=1
useRandomSeed=True

#take all the data in the given file.
def get_features_dev(cwd, turkFile, useOneHot,uniq_turker,addTurkerOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)

        #create a hash table to store unique adj
        uniq_adj={}
        counter=0

        #create a total list of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:
            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1



        turk_counter=0
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)







        #Split data in to train-dev-test
        noOfRows=df_raw_turk_data.shape[0]



        #create an numpy array of that range
        allIndex=np.arange(noOfRows)

        #now shuffle it and split
        #np.random.seed(1)
        #np.random.shuffle(allIndex)

        #take 80% of the total data as training data- rest as testing
        #eighty=math.ceil(noOfRows*80/100)
        #twenty_index=math.ceil(noOfRows*80/100)

        #eighty= number of rows
        #trainingData_indices=allIndex[:eighty]
        #rest=allIndex[eighty:]






        y=np.array([],dtype="float32")
        features = []


        #list of all adjectives in the training data, including repeats
        all_adj=[]



        data_indices=np.arange(noOfRows)

        # #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(data_indices),total=noOfRows, desc="readData:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            all_adj.append(adj)

            #get the index of the adjective
            adjIndex=uniq_adj[adj]


            embV=[]
            if(useOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count

                one_hot_adj[adjIndex] = 1

                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                #emb = vec[vocab[adj]].numpy()
                embV=embV
                #embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]


            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)

            one_hotT[turkerIndex]=1


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]

            #localFeatures.extend(embV)


            localFeatures.append(mean.item())

            localFeatures.append(stddev)
            if(addTurkerOneHot):
                localFeatures.extend(one_hotT)






            ############feed this combined vector as a feature vector to the linear regression


            ylabelLocal=np.array([logRespDev], dtype="float32")
            #featuresLocal = np.array([adj_mean_stddev_turk])
            #featuresLocal=featuresLocal.transpose()

            features.append(localFeatures)

            combinedY=np.append(y,ylabelLocal)
            y=combinedY





        npfeatures=np.asarray(features, dtype="float32")


        return npfeatures,y, uniq_adj, all_adj

def get_features_labels_from_data(cwd, turkFile, useAdjOneHot, uniq_turker, addTurkerOneHot):
        df_raw_turk_data=readRawTurkDataFile(cwd, turkFile)

        #create a hash table to store unique adj
        uniq_adj={}
        counter=0

        #create a dictionary  of unique adj in this collection
        for a in df_raw_turk_data["adjective"]:
            if(a) not in uniq_adj:
                #if its not there already add it as the latest element
                uniq_adj[a]=counter
                counter=counter+1

        uniq_adj_list=[]
        # create a list  of unique adj in this collection
        for a in uniq_adj.keys():
                uniq_adj_list.append(a)



        turk_counter=0
        #create a total list of unique turkers in this collection
        for b in df_raw_turk_data["turker"]:
            if(b) not in uniq_turker:
                #if its not there already add it as the latest element
                uniq_turker[b]=turk_counter
                turk_counter=turk_counter+1

        uniq_adj_count=len(uniq_adj)
        uniq_turker_count=len(uniq_turker)





        #Split data in to train-dev-test
        noOfRows=df_raw_turk_data.shape[0]


        #create an numpy array of that range
        allIndex=np.arange(noOfRows)


        #take 80% of the total data as training data- rest as testing
        #eighty=math.ceil(noOfRows*80/100)
        #twenty_index=math.ceil(noOfRows*80/100)

        #eighty= number of rows
        #trainingData_indices=allIndex[:eighty]
        #rest=allIndex[eighty:]






        y=np.array([],dtype="float32")
        features = []


        #list of all adjectives in the training data, including repeats
        all_adj=[]



        data_indices=np.arange(noOfRows)

        #for each of the adjective create a one hot vector
        for rowCounter, eachTurkRow in tqdm(enumerate(data_indices),total=noOfRows, desc="readData:"):

            ########create a one hot vector for adjective
            # give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            all_adj.append(adj)

            #get the index of the adjective
            adjIndex=uniq_adj[adj]

            embV=[]
            if(useAdjOneHot):
                #####create a one hot vector for all adjectives
                # one_hot_adj=np.zeros(uniq_adj_count)
                one_hot_adj = [0] * uniq_adj_count
                one_hot_adj[adjIndex] = 1
                #todo : extend/append this new vector
                embV=one_hot_adj

            else:
                #pick the corresponding embedding from glove
                #emb = vec[vocab[adj]].numpy()
                embV=embV
                #embV=emb

            ################to create a one hot vector for turker data also
            #get the id number of of the turker
            turkerId=df_raw_turk_data["turker"][eachTurkRow]
            turkerIndex=uniq_turker[turkerId]

            #create a one hot vector for all turkers
            one_hotT=[0]*(uniq_turker_count)
            one_hotT[turkerIndex]=1


            ################get the mean and variance for this row and attach to this one hot
            #give this index to the actual data frame
            adj=df_raw_turk_data["adjective"][eachTurkRow]
            mean=df_raw_turk_data["mean"][eachTurkRow]
            stddev=df_raw_turk_data["onestdev"][eachTurkRow]
            logRespDev=df_raw_turk_data["logrespdev"][eachTurkRow]

            #############combine adj-1-hot to mean , variance and turker-one-hot

            localFeatures=[]

            localFeatures.append(mean.item())
            localFeatures.append(stddev)
            if(addTurkerOneHot):
                localFeatures.extend(one_hotT)





            ylabelLocal=np.array([logRespDev], dtype="float32")
            features.append(localFeatures)

            combinedY=np.append(y,ylabelLocal)
            y=combinedY





        npfeatures=np.asarray(features, dtype="float32")


        return npfeatures,y, uniq_adj, all_adj,uniq_turker,uniq_adj_list




