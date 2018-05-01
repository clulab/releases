from __future__ import division
import scipy
import os
import sys;
import csv;
import collections
import numpy as np
import itertools
import pickle as pk
import time




from utils.grounding import get_features_labels_from_data
from utils.squish import run_nfoldCV_on_turk_data_4chunks
from utils.squish import nfoldCV_adj_grouped_turk_data_4chunks
from utils.squish import load_nfoldCV_adj_grouped_turk_data_4chunks
from utils.squish import train_dev_print_rsq
from utils.squish import runOnTestPartition
from utils.read_write_data import readRawTurkDataFile

start_time = time.time()

cwd=os.getcwd()

#files from turk experiment
turkInterceptFile="turk_with_intercept.txt"
entire_turk_data="all_turk_data.csv"

#files after the data was split into training-dev-test
dev_entire_data= "dev.csv"
training_data="trainingData.csv"
#test_data="test.csv"
#test_data="test_no_random_seed.csv"
#test_data="test_rand_seed1.csv"
test_data="test_no_random_seed2.csv"


#files after the data was sorted based on adjectives and was split into training-dev-test
dev_adj="dev_adj.csv"
training_adj="trainingData_adj.csv"
test_adj="test_adj.csv"

addTurkerOneHot=False
addAdjOneHot=False
useEarlyStopping=False
use4Chunks=True;


rsq_on_test_all_data= "rsq_on_test.txt"
rsq_on_test_adj_based_data= "rsq_on_test_adj_based_data.txt"

if __name__ == "__main__":
    try:





        while True:
                print("                      ")
                print("          ******            ")

                print("Welcome to Grounding For Adjectives. Please pick one of the following:")

                print("To train and save using adj based split press :1")
                print("To train on all the data (not adj based split) press :2")
                print("To train with nfoldCV on entire data (no adj based split)  press:3")
                print("To train with nfoldCV on  adj based split)  press:4")
                print("To test using a saved model on alldata_test_partition which was trained on entire data 80-10-10 press:5")
                print("To test using a saved model on adj_based_data_test_partition which was trained on adj_based_split press:6")
                print("To test using a saved model on adj_based_data_test_partition which was trained on nfcv 4 chunks press:7")
                print("To exit Press:0")


                myInput=input("what is your choice:")

                uniq_turker = {}

                if(myInput=="3"):


                    #run1: run with leave one out cross validationon all the turk experiment data points-i.e no adjective based split

                    # read all the data. i.e without training-dev-split. This is for LOOCV
                    features, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, entire_turk_data,
                                                                                                                 addAdjOneHot, uniq_turker, addTurkerOneHot)

                    # run1: run with leave one out cross validation
                    #run_nfoldCV_on_turk_data(features, y, adj_lexicon, all_adj,addTurkerOneHot,useEarlyStopping,use4Chunks)
                    run_nfoldCV_on_turk_data_4chunks(features, y, adj_lexicon, all_adj, addTurkerOneHot, useEarlyStopping,
                                             use4Chunks)

                    print("done loocv for all turk data, going to exit")


                    #run 2 : do training and dev tuning separately--this is entire data, not based on adjectives.
                    # readtraining data
                    # uniq_turker = {}
                    # features, y, adj_lexicon, all_adj, uniq_turker = get_features_training_data(cwd, training_data,
                    #                                                                             False, uniq_turker)
                    # trained_model=do_training(features, y, adj_lexicon, all_adj)
                    # # test on dev data
                    # tuneOnDev(cwd, dev, False, uniq_turker)



                    # run 3: run both dev and training together and print rsq after each epoch
                    # mutual exclusive with run 2 above





                    #instead of splitting data into 80-10-10, do LOOCV based on adjectives
                    #run_loocv_per_adj(features, y, adj_lexicon, all_adj,addTurkerOneHot,uniq_adj_list)



                    print("done loocv for adj based turk data, going to exit")



                    adj_lexicon_flipped = dict()
                    #total number of unique adjectives
                    num_adj = len(adj_lexicon)

                    #key=index value=adjective
                    for a, idx in adj_lexicon.items():
                        adj_lexicon_flipped[idx] = a




                    #############use the trained model to test on test split


                    elapsed_time = time.time() - start_time
                    print("time taken:" + str(elapsed_time/60)+"minutes")

                else:
                    if(myInput=="6"):

                        #empty out the existing file
                        with open(cwd + "/outputs/" + rsq_on_test_adj_based_data, "w+")as rsq_values:
                            rsq_values.write("Epoch \t Train \t\t Dev \n")
                            rsq_values.close()

                        #append the rest of the values
                        with open(cwd+"/outputs/" +rsq_on_test_adj_based_data, "a")as rsq_values:

                            trained_model = pk.load( open( "adj_data_80-10-10.pkl", "rb" ))
                            runOnTestPartition(trained_model,test_adj,cwd, uniq_turker,rsq_values,addTurkerOneHot,1)



                            # features, y, adj_lexicon,all_adj= get_features_y(cwd, turkFile,False)
                            # adj_lexicon_flipped = dict()
                            # #total number of unique adjectives
                            # num_adj = len(adj_lexicon)
                            #
                            # #key=index value=adjective
                            # for a, idx in adj_lexicon.items():
                            #     adj_lexicon_flipped[idx] = a
                            #
                            #
                            # #run with leae one out cross validation
                            # run_loocv_on_turk_data(features, y, adj_lexicon, all_adj)

                            #run just with a classic train-dev-test partition
                            elapsed_time = time.time() - start_time
                            print("time taken:" + str(elapsed_time/60)+"minutes")
                    else:

                        if(myInput=="5"):

                            #empty out the existing file
                            with open(cwd + "/outputs/" + rsq_on_test_all_data, "w+")as rsq_values:
                                rsq_values.write("Epoch \t Train \t\t Dev \n")
                                rsq_values.close()

                            #append the rest of the values
                            with open(cwd+"/outputs/" +rsq_on_test_all_data, "a")as rsq_values:

                                trained_model = pk.load( open( "all_data_80-10-10.pkl", "rb" ))
                                #trained_model = pk.load(open("all_data_80-10-10-122-epochs.pkl", "rb"))
                                runOnTestPartition(trained_model,test_data,cwd, uniq_turker,rsq_values,addTurkerOneHot,1)



                                # features, y, adj_lexicon,all_adj= get_features_y(cwd, turkFile,False)
                                # adj_lexicon_flipped = dict()
                                # #total number of unique adjectives
                                # num_adj = len(adj_lexicon)
                                #
                                # #key=index value=adjective
                                # for a, idx in adj_lexicon.items():
                                #     adj_lexicon_flipped[idx] = a
                                #
                                #
                                # #run with leae one out cross validation
                                # run_loocv_on_turk_data(features, y, adj_lexicon, all_adj)

                                #run just with a classic train-dev-test partition
                                elapsed_time = time.time() - start_time
                                print("time taken:" + str(elapsed_time/60)+"minutes")

                        else:

                            if(myInput=="0"):
                                print("******Good Bye")
                                break;

                            else:
                                        if(myInput=="1"):


                                            #code that splits the data based on adjectives and not the entire data- should be used only once ideally
                                            #features, y, adj_lexicon, all_adj, uniq_turker = split_data_based_on_adj(cwd, entire_turk_data,
                                            #                                                                          False, uniq_turker)

                                            features, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, training_adj,
                                                                                                                                         addAdjOneHot, uniq_turker, addTurkerOneHot)

                                            #train on the adj based training split and tune on dev. All is done inside train_dev_print_rsq
                                            trained_model = train_dev_print_rsq(dev_adj,features, y, adj_lexicon, all_adj,uniq_turker,addTurkerOneHot)


                                            #instead of splitting data into 80-10-10, do LOOCV based on adjectives
                                            #run_loocv_per_adj(features, y, adj_lexicon, all_adj,addTurkerOneHot,uniq_adj_list)



                                            print("done loocv for adj based turk data, going to exit")


                                        else:

                                            if(myInput=="2"):
                                                #run 2 : do training and dev tuning separately--this is entire data, not based on adjectives.

                                                uniq_turker = {}
                                                #readtraining data
                                                features, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list= get_features_labels_from_data(cwd, training_data, False, uniq_turker, addTurkerOneHot)
                                                 #train on whatever data split is given and tune on dev. All is done inside train_dev_print_rsq
                                                #features here is the features you just read in the line above
                                                trained_model = train_dev_print_rsq(dev_entire_data,features, y, adj_lexicon, all_adj,uniq_turker,addTurkerOneHot)
                                            else:
                                                    if(myInput=="4"):
                                                            # read the raw  turk data as a pandas data frame
                                                            df_raw_turk_data = readRawTurkDataFile(cwd, entire_turk_data)


                                                            # # sort the entire data based on adjectives.
                                                            # sorted_df_raw_turk_data = df_raw_turk_data.sort_values(
                                                            #     "adjective")

                                                           # read all the data. i.e without training-dev-split. This is for LOOCV
                                                            features, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, entire_turk_data,
                                                                                                                                                         addAdjOneHot, uniq_turker, addTurkerOneHot)

                                                             # run with leave one out cross validation
                                                            nfoldCV_adj_grouped_turk_data_4chunks(df_raw_turk_data,features, y, adj_lexicon, all_adj,addTurkerOneHot)

                                                            print("done loocv for adj turk data, going to exit")

                                                    else:

                                                            if(myInput=="7"):

                                                                # read the raw  turk data as a pandas data frame
                                                                df_raw_turk_data = readRawTurkDataFile(cwd, entire_turk_data)


                                                               # read all the data. i.e without training-dev-split. This is for LOOCV
                                                                features, y, adj_lexicon, all_adj, uniq_turker,uniq_adj_list = get_features_labels_from_data(cwd, entire_turk_data,
                                                                                                                                                             addAdjOneHot, uniq_turker, addTurkerOneHot)

                                                                 # run with leave one out cross validation
                                                                load_nfoldCV_adj_grouped_turk_data_4chunks(df_raw_turk_data,features, y, adj_lexicon, all_adj,addTurkerOneHot)

                                                                print("done loocv for adj turk data, going to exit")









    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






