#!/usr/bin/env python
# coding: utf-8

# # Model Training
# > To facilitate a more automated training procedure, the model training is moved to a standalone python script.  
# This keeps Keras much happier in terms of required restarts and memory usage.

# In[1]:

from IPython import get_ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from importlib import reload

import sys
import numpy as np
import time # !
import json
from matplotlib import pyplot as plt

from keras.utils import to_categorical

import glove_helper
from loadutils import conll2003Data, saveProcessedData, retrieve_model
from common import vocabulary, utils


# In[2]:


start_time = time.time()

"""
Set the random seed
"""
random_seed = sys.argv[3]
np.random.seed(int(random_seed))

"""
Pick which language to train on
"""
LANGUAGE = sys.argv[1]
# LANGUAGE = "ca"
# LANGUAGE = "gd"

"""
Pick how much of the training data to use
"""
# TRAIN_AMOUNT = "100"
# TRAIN_AMOUNT = "50"
# TRAIN_AMOUNT = "10"
TRAIN_AMOUNT = sys.argv[2]

DIRECTORY = "../data/pos_tagging/"+LANGUAGE+"/"
# DIRECTORY = sys.argv[4]

#training file depends on low-resource or not; 100%, 50%, or 10% of training data used
TRAIN_FILE = DIRECTORY+"train_"+TRAIN_AMOUNT+".txt"

# dev, test, and vectors stay the same
DEV_FILE = DIRECTORY+"dev.txt"
TEST_FILE = DIRECTORY+"test.txt"
# VECTORS = "data/"+LANGUAGE+"/wiki."+LANGUAGE+".zip"
VECTORS = "data/"+LANGUAGE+"/cc."+LANGUAGE+".300.zip"
# VECTORS = sys.argv[5]

# out files for IPC
HYPER_PARAM_FILE = "hyper_params.json"

VOCAB_SIZE = 200000

## PRINT OUT HYPERPARAMETERS FOR REFERENCE
print("LANGUAGE:\t", LANGUAGE)
print("TRAIN_AMOUNT:\t", TRAIN_AMOUNT)
print("RANDOM SEED:\t", random_seed)


# ## Local helper utils

# In[3]:


# local untils

# timeit decorator
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' %                   (method.__name__, (te - ts) * 1000))
        return result
    return timed


# In[4]:


def construct_embedding_matrix(embed_dim, vocab_size):
    """
    construct embedding matrix from GloVe 6Bn word data
    
    reuse glove_helper code from w266 
    
    Returns: an embedding matrix directly plugged into keras.layers.Embedding(weights=[embedding_matrix])
    """
    reload(glove_helper)
    hands = glove_helper.Hands(vector_zip=VECTORS, ndim=embed_dim)
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    for i in range(vocabData.vocab.size):
        word = vocabData.vocab.ids_to_words([i])[0]
        try:
            embedding_vector = hands.get_vector(word)
        except:
            embedding_vector = hands.get_vector("<unk>")
        embedding_matrix[i] = embedding_vector

    return embedding_matrix


# In[5]:


def plot_history( history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# ## Load the Data

# In[6]:


# UPDATES!
global_max_features = 200000
windowLength = 11
#testNumSents = 20000

# Use training set to build vocab here
vocabData = conll2003Data(TRAIN_FILE)
vocabData.buildVocab( vocabSize=global_max_features, verbose=False)


# In[7]:


# Format training data
trainX, trainX_pos, trainX_capitals, trainY  = vocabData.formatWindowedData( 
                                                  vocabData.train_sentences, 
                                                  windowLength=windowLength,
                                                  verbose=False)

# read in dev data
devSents = vocabData.readFile( DEV_FILE)
devX, devX_pos, devX_capitals, devY = vocabData.formatWindowedData( 
                                              devSents, 
                                              windowLength=windowLength,
                                              verbose=False)

# read in the test data
testSents = vocabData.readFile( TEST_FILE)
testX, testX_pos, testX_capitals, testY = vocabData.formatWindowedData( 
                                                testSents, 
                                                windowLength=windowLength,
                                                verbose=False)


# In[8]:


# Load GloVe embedding matrix

# setting it to global instead of hyper_param dictionaries because embedding \
# dimensions need to be decided before the data is loaded for the decoder output
# global_embed_dim = 50
global_embed_dim = 300

embedding_matrix = construct_embedding_matrix( global_embed_dim, 
                                               global_max_features)


# In[9]:


# Get Y

# encoding 1-hot for ner targets
trainY_cat = to_categorical(trainY.astype('float32'))
devY_cat = to_categorical(devY.astype('float32'), num_classes=trainY_cat.shape[1])
testY_cat = to_categorical(testY.astype('float32'), num_classes=trainY_cat.shape[1])

trainY_cat = np.array(list(map( lambda i: np.array(i[3:], dtype=np.float), trainY_cat)), dtype=np.float)
devY_cat = np.array(list(map( lambda i: np.array(i[3:], dtype=np.float), devY_cat)), dtype=np.float)
testY_cat = np.array(list(map( lambda i: np.array(i[3:], dtype=np.float), testY_cat)), dtype=np.float)


# In[10]:


# Get decoder Y -- 50 dim embedding of center word

train_decoderY = embedding_matrix[trainX[:,4]]
dev_decoderY = embedding_matrix[devX[:,4]]
test_decoderY = embedding_matrix[testX[:,4]]


# In[11]:


# trainX => windows of size 11, center word is what we're predicting the label for
# trainY => tag/prediction for center word only!
# train_decoderY[0]


# In[13]:


# our take on keras's to_categorical method, but if we DON'T want a one-hot vector
# instead, the values of the vectors are 1 for ALL of the features present, so one token's vector
# could have a 1 in more than one dimension here

def to_almost_categorical(y, num_classes):
    almost_categorical_array = np.zeros((y.shape[0], y.shape[1], num_classes))
#     print(almost_categorical_array.shape)
    for i in range(len(y[0])):
#         print("window:\t",window)
        window = y[i]
        window_index = i
#         print("window_index:\t", i)
#         print("window:\t", window)
        for j in range(len(window)):
            token = window[j]
            token_index = j
#             print("token_index:\t", token_index)
#             print("token:\t",token)
            for value in token:
#                 print("value:\t",value)
#                 print("token_index:\t",token_index)
                value = int(value)
                almost_categorical_array[i,token_index,value] += 1
#                 print(almost_categorical_array[i])
    # print(almost_categorical_array)
    return almost_categorical_array
        


# In[14]:


# Get X pos tags

# encoding 1-hot for pos tags
trainX_pos_cat = to_almost_categorical(trainX_pos, num_classes=vocabData.posTags.size)
devX_pos_cat = to_almost_categorical(devX_pos, num_classes=vocabData.posTags.size) 
testX_pos_cat = to_almost_categorical(testX_pos, num_classes=vocabData.posTags.size)


# In[15]:


# encoding 1-hot for pos tags
# trainX_pos_cat = to_categorical(trainX_pos.astype('float32'))
# devX_pos_cat = to_categorical(devX_pos.astype('float32'), num_classes=trainX_pos_cat.shape[2]) 
# testX_pos_cat = to_categorical(testX_pos.astype('float32'), num_classes=trainX_pos_cat.shape[2])


# In[16]:



trainX_pos_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), trainX_pos_cat)), dtype=np.float)


# In[17]:


devX_pos_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), devX_pos_cat)), dtype=np.float)


# In[18]:


testX_pos_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), testX_pos_cat)), dtype=np.float)


# In[24]:


# Get X capitalization 

# encoding 1-hot for capitalization info  ("allCaps", "upperInitial", "lowercase", "mixedCaps", "noinfo")
trainX_capitals_cat = to_categorical(trainX_capitals.astype('float32'))
devX_capitals_cat = to_categorical(devX_capitals.astype('float32'), num_classes=trainX_capitals_cat.shape[2]) 
testX_capitals_cat = to_categorical(testX_capitals.astype('float32'), num_classes=trainX_capitals_cat.shape[2])


# In[25]:


trainX_capitals_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), trainX_capitals_cat)), dtype=np.float)
devX_capitals_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), devX_capitals_cat)), dtype=np.float)
testX_capitals_cat = np.array(list(map( lambda i: np.array(i[:,3:], dtype=np.float), testX_capitals_cat)), dtype=np.float)


# ## Set up model parameters

# In[27]:


# define hyper parameters for model
# CNN
hyper_param_cnn = {
    
    'max_features' : global_max_features,  # 20000
    'maxlen' : trainX.shape[1],  # window size (11)
    'poslen' : trainX_pos_cat.shape[2],  # pos classes (216)
    'capitallen' : trainX_capitals_cat.shape[2],  # capitalization classes (5)
    'ner_classes' : trainY_cat.shape[1],  # 18 
    'embed_dim' : global_embed_dim,  # word embedding size (300)
    'num_routing' : 3, 

    'use_glove' : True,
    'allow_glove_retrain' : False,
    'use_pos_tags' : True,
    'use_capitalization_info' : True,    
    
    'conv1_filters' : 256,
    'conv1_kernel_size' : 3,
    'conv1_strides' : 1,
    'conv1_padding' : 'valid',
    
    'conv2_filters' : 256,
    'conv2_kernel_size' : 3,
    'conv2_strides' : 1,
    'conv2_padding' : 'valid',
    
    'conv3_filters' : 128,
    'conv3_kernel_size' : 3,
    'conv3_strides' : 1,
    'conv3_padding' : 'valid',
    
    'max_pooling_size' : 3,
    'max_pooling_strides' : 1,
    'max_pooling_padding' : 'valid',
    'maxpool_dropout' : 0.3,
    
    'feed_forward_1' : 328,
    'ff1_dropout' : 0.3,
    'feed_forward_2' : 192,
    'ff2_dropout' : 0.3,
    
    'save_dir' : './result',
    'batch_size' : 100,
    'debug' : 2,
    'epochs' : 5,
    'stopping_patience' : 2, # default to same as epochs, ie don't use
    'dropout_p' : 0.25,
    'embed_dropout' : 0.25,  # set to 0 to disable dropout
    'lam_recon' : 0.0005,
    
    'optimizer' : 'Adam', #or 'SGD'
}


# ## Save All Data to Disk

# In[28]:

saveDirectory = "/tmp/cnn_nolearn_"+LANGUAGE+"_"+TRAIN_AMOUNT+"_"+random_seed

# save all loaded data for use by training process
saveProcessedData( trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat, devX_pos_cat, trainY_cat, devY_cat, embedding_matrix, train_decoderY, dev_decoderY, saveDirectory)


# ## Model Training Functions

# In[29]:


@timeit 
def trainModelSP( testFunc, modelName, hyper_params, embed_matrix=None, verbose=False):
    """
    testFunc - the name of the python file to run
    modelName - the internal name (ID) of the model to train
    hyper_params - a dict of hyper parameters
    """
    # save the hyperparams
    with open(HYPER_PARAM_FILE, mode='w') as fp:
        json.dump( hyper_params, fp)
    
    # saveDirectory = "savedProcessedData_"+LANGUAGE+"_"+TRAIN_AMOUNT+"_"+random_seed
    # call the train function
    # consider replacing with a call to subprocess!!
    get_ipython().system('python {testFunc} {modelName} {HYPER_PARAM_FILE} {saveDirectory}')


# In[30]:


@timeit 
def testFeatures( testFunc, modelName, hyper_params):
    """
    builds and trains models for the configuration in hyper_params,
    1 for each input feature configuration: base, pos, caps, pos + caps 
      (no longer training pos and caps independently)
    
    testFunc - the name of the python file to run
    modelName - the model name to use for labeling
    """
    hypers = hyper_params.copy()
    
    # try the embeddings with different features
    
    # base
    curModel = LANGUAGE + "_" + TRAIN_AMOUNT + "_" + modelName + "_" + random_seed
    trainModelSP( testFunc, curModel, hypers )
    
    # pos tags
#     curModel = modelName + "_" + LANGUAGE + "_" + TRAIN_AMOUNT + "_features"
#     hypers['use_pos_tags'] = True
#     hypers['use_capitalization_info'] = False
#     trainModelSP( testFunc, curModel, hypers )
    
    # capitalization info
    #curModel = modelName + "_caps"
    #hypers['use_pos_tags'] = False
    #hypers['use_capitalization_info'] = True
    #trainModelSP( testFunc, curModel, hypers )
    
    # both
#     curModel = modelName + "_pos_caps"
#     hypers['use_pos_tags'] = True
#     hypers['use_capitalization_info'] = True
#     trainModelSP( testFunc, curModel, hypers )
    


# ##  Training
# > the output isn't pretty, but we don't really need it since everything is stored in the history log. It is really just to show a sign of life.  
# > * The below is just an example of how to set hyper parameters and train multiple models.

# In[31]:

# ### train capsnet

# In[32]:


# ### train CNN

# In[33]:


# CNN training function
testFunc = "CapsNet_for_NER/code/trainCNNModel.py"

hypers = hyper_param_cnn.copy()
hypers['epochs'] = 10
hypers['stopping_patience'] = 3
hypers['use_pos_tags'] = False
hypers['use_capitalization_info'] = False


# # use glove, no learn
print("\n\nCNN No Learn")
hypers['use_glove'] = True
hypers['allow_glove_retrain'] = False
hypers['embed_dropout'] = 0.0
testFeatures( testFunc, "cnn_nolearn", hypers)


# In[ ]:

print("\n\n\nTRAINING SUMMARY\n\n")

print("LANGUAGE:\t", LANGUAGE)
print("TRAIN_AMOUNT:\t", TRAIN_AMOUNT)
print("RANDOM SEED:\t", random_seed)

print("Training Time for cnn nolearn")
print("--- %s seconds ---" % (time.time() - start_time))

