from __future__ import division
import scipy
import os
import sys;
import utils;
import csv;
import collections
import numpy as np
import itertools
from utils.read_write_data import readFile
import pickle as pk
from scipy.stats import kendalltau, spearmanr

import time





start_time = time.time()

if __name__ == "__main__":
    try:

        glove = "glove.txt"
        turk = "turk.txt"
        w2v = "w2v.txt"
        cbow1 = "turkMarneffe.txt"
        cbow2 = "turkMarneffeCbowDifford.txt"
        cbow3 = "glove_vectors_syn_ant.txt"
        cbow4 = "glove_vectors_syn_ant_sameord_difford.txt"
        paragram1="paragram_vectors_syn_ant_sameord_difford.txt"
        
        cwd = os.getcwd()

        base_dir_name = os.path.dirname(os.path.abspath(sys.argv[0]))
        if (base_dir_name != cwd):
            os.chdir(base_dir_name)


        #read the output produced by the scala code
        # #glove_data = utils.read_data.readFile(cwd, glove)
        turk_data = utils.read_write_data.readFile(cwd, turk)
        #w2v_data = utils.read_data.readFile(cwd, w2v)
        marneffe_data= utils.read_write_data.readFile(cwd, paragram1)

        #print(w2v_data)

        #spear_turk_glove=spearmanr(glove_data['intercept'], turk_data['intercept'])[0]
        #spear_turk_w2v = spearmanr(w2v_data['intercept'], turk_data['intercept'])[0]


        # turk_glove = spearmanr( turk_data['adj'],glove_data['adj'])
        # turk_w2v = spearmanr(turk_data['adj'],w2v_data['adj'])
        turk_marneffe = spearmanr(turk_data['adj'],marneffe_data['adj'])

        print("spearman correlation values varies between -1 and +1 with 0 implying no correlation.Correlations of -1 or +1 imply an exact monotonic relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.")
       # print("turk_glove:"+str(turk_glove))
       # print("turk_w2v:" + str(turk_w2v))
        print("turk_marneffe:" + str(turk_marneffe))





    ##################################end of dev phase####################
    except:
        import traceback
        print('generic exception: ' + traceback.format_exc())
        elapsed_time = time.time() - start_time
        print("time taken:" + str(elapsed_time))






