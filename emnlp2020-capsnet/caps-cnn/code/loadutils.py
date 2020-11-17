import numpy as np
import time
from importlib import reload
import glove_helper
import os
# capsule layers from Xifeng Guo 
# https://github.com/XifengGuo/CapsNet-Keras
from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask
from keras.models import model_from_json
from common import vocabulary, utils

DIRECTORY = "/home/u26/zupon/FIXME"

# a dict of all processed data filenames
TRAIN_DATA_FILES = { 'trainX' : 'trainX.npy',
                     'trainY' : 'trainY.npy',
                     'devX' : 'devX.npy',
                     'devY' : 'devY.npy',
                     'trainX_pos' : 'trainX_pos.npy',
                     'trainX_capitals' : 'trainX_capitals.npy',
                     'devX_pos' : 'devX_pos.npy',
                     'devX_capitals' : 'devX_capitals.npy',
                     'glove_embed' : 'glove_embed.npy',
                     'train_decoderY' : 'train_deocoderY.npy',
                     'dev_decoderY' : 'dev_decoderY.npy'}


DEV_RESULT_FILES = { 'raw_y_pred' : 'raw_y_pred.npy',
                     'raw_y_pred_decoder_embeddings' : 'raw_y_pred_decoder_embeddings.npy',
                     'y_pred' : 'y_pred.npy'}


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
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def saveDevPredictionsData(modelName, raw_y_pred, raw_y_pred_decoder_embeddings, y_pred, modelsDir='dev_Predictions'):
    """
    Save major prediction data and scores
    """
    path = str(modelsDir) + "/" + modelName + "_"
    np.save(path + DEV_RESULT_FILES['raw_y_pred'], raw_y_pred)
    np.save(path + DEV_RESULT_FILES['raw_y_pred_decoder_embeddings'], raw_y_pred_decoder_embeddings)
    np.save(path + DEV_RESULT_FILES['y_pred'], y_pred)
    
    
def loadDevPredictionsData(modelName, modelsDir='dev_Predictions'):
    """
    Load major prediction data and scores
    
    Argument:
        modelName : modelName given to model in training code. 
        see testFeatures( testFunc, modelName="encoder_50e", hypers) in model_training_tmpl.ipynb
        
    Returns:
        raw_y_pred : 1-hot dev set prediction of shape(?, number of ner classes)
        raw_y_pred_decoder_embeddings : dev set decoder embedding prediction of shape (?, embed_dim)
        y_pred : NER idx version of converted from raw_y_pred. of shape (?,). can apply vocabData.ner_vocab.ids_to_words()
    """
    path = str(modelsDir) + "/" + modelName + "_"
    raw_y_pred = np.load(path + DEV_RESULT_FILES['raw_y_pred'])
    try:
        raw_y_pred_decoder_embeddings = np.load(path + DEV_RESULT_FILES['raw_y_pred_decoder_embeddings'])
    except:
        raw_y_pred_decoder_embeddings = np.empty(0)        
    y_pred = np.load(path + DEV_RESULT_FILES['y_pred'])

    if y_pred.dtype == '<U15':
        raw_y_pred_decoder_embeddings = np.empty(0)
        y_pred = np.argmax(raw_y_pred, axis=1) + 3

    return raw_y_pred, raw_y_pred_decoder_embeddings, y_pred


def saveProcessedData( trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat, devX_pos_cat, trainY_cat, devY_cat, embedding_matrix, train_decoderY, dev_decoderY, saveDirectory):
    """
    Save all processed training data
    """
    DIRECTORY = saveDirectory
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX']), trainX)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX_capitals']), trainX_capitals_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX_pos']), trainX_pos_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX']), devX)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX_capitals']), devX_capitals_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX_pos']), devX_pos_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainY']), trainY_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devY']), devY_cat)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['glove_embed']), embedding_matrix)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['train_decoderY']), train_decoderY)
    np.save(os.path.join(DIRECTORY,TRAIN_DATA_FILES['dev_decoderY']), dev_decoderY)


def loadProcessedData(saveDirectory):
    """
    Load all processed training data

    returns:
    trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat,
    devX_pos_cat, trainY_cat, devT_cat, embedding_matrix
    """

    DIRECTORY = saveDirectory

    trainX = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX']))
    trainX_capitals_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX_capitals']))
    trainX_pos_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainX_pos']))
    devX = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX']))
    devX_capitals_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX_capitals']))
    devX_pos_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devX_pos']))
    trainY_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['trainY']))
    devY_cat = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['devY']))
    embedding_matrix = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['glove_embed']))
    train_decoderY = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['train_decoderY']))
    dev_decoderY = np.load(os.path.join(DIRECTORY,TRAIN_DATA_FILES['dev_decoderY']))

    return trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat, \
           devX_pos_cat, trainY_cat, devY_cat, embedding_matrix, train_decoderY, dev_decoderY


def save_model(model, name):
    json_string = model.to_json()
    architecture = name+'_architecture.json' 
    weights = name+'_weights.h5'
    open(architecture, 'w').write(json_string)
    model.save_weights(weights)

# uses path from hyper paramaters
def retrieve_model(modelName, hypers, weights=True):
    from keras.models import model_from_json
    import keras.backend as K
    from keras.utils import CustomObjectScope
    from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask
    print( 'Retrieving model: {0}'.format(modelName))
    architecture = hypers['save_dir'] + '/' + modelName + '_model_architecture.json'
    with open(architecture, 'r') as f:
        if len(f.readlines()) !=0:
            f.seek(0)
            model_saved = model_from_json(f.read(), custom_objects={'CapsuleLayer': CapsuleLayer, 'Length': Length})

    if weights:
        weights = hypers['save_dir'] + '/' + modelName + '_weights_model.h5'    
        model_saved.load_weights(weights)
    return model_saved

#decoder with hyper parameters
def retrieve_decoder_model(modelName, hypers, weights=True):
    from keras.models import model_from_json
    import keras.backend as K
    from keras.utils import CustomObjectScope
    from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask
    print( 'Retrieving model_eval: {0}'.format(modelName))
    architecture = hypers['save_dir'] + '/' + modelName + '_model_eval_architecture.json'
    with open(architecture, 'r') as f:
        if len(f.readlines()) !=0:
            f.seek(0)
            model_eval_saved = model_from_json(f.read(), custom_objects={'CapsuleLayer': CapsuleLayer, 'Length': Length,
                                                                            'Mask': Mask})
    if weights:
        weights = hypers['save_dir'] + '/' + modelName + '_weights_model_eval.h5'    
        model_eval_saved.load_weights(weights)
    return model_eval_saved

# specify path
def retrieve_model_path(modelName, path, weights=True):
    from keras.models import model_from_json
    import keras.backend as K
    from keras.utils import CustomObjectScope
    from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask
    print( 'Retrieving model: {0}'.format(modelName))
    architecture = path + '/' + modelName + '_model_architecture.json'
    with open(architecture, 'r') as f:
        if len(f.readlines()) !=0:
            f.seek(0)
            model_saved = model_from_json(f.read(), custom_objects={'CapsuleLayer': CapsuleLayer, 'Length': Length, 'Mask': Mask})

    if weights:
        weights = path + '/' + modelName + '_weights_model.h5'    
        model_saved.load_weights(weights)
    return model_saved


def construct_embedding_matrix(embed_dim, vocab_size, vocabData):
    """
    construct embedding matrix from GloVe 6Bn word data
    
    reuse glove_helper code from w266 
    
    Returns: an embedding matrix directly plugged into keras.layers.Embedding(weights=[embedding_matrix])
    """
    reload(glove_helper)
    hands = glove_helper.Hands(ndim=embed_dim)
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    for i in range(vocabData.vocab.size):
        word = vocabData.vocab.ids_to_words([i])[0]
        try:
            embedding_vector = hands.get_vector(word)
        except:
            embedding_vector = hands.get_vector("<unk>")
        embedding_matrix[i] = embedding_vector

    return embedding_matrix


class conll2003Data(object):
    """
    Keep track of data and processing operations for a single CoNLL2003 data file.
    """

    def __init__(self, filePath_train):
        """
        filePath(string): path to a CoNLL2003 raw data file for training the vocabulary
        """
        self.vocab = []
        self.posTags = []
        self.nerTags = []
        self.train_sentences = self.readFile(filePath_train)


    @timeit
    def readFile(self, filePath, canonicalize=True, verbose=False):
        """
        Read the conll2003 raw data file

        filename(string) - path to conll2003 file (train, test, etc.)
        
        Returns: a list of lists of lists corresponding to the words, pos tags, ner tags, capitalization
                 in each sentence

        """
        print ("----------------------------------------------------")
        print ("reading file from path", str(filePath))
        f = open(filePath)
        sentences = []
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith("-DOCSTART") or line[0] == '\n':
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                continue
            
            # input format is [ word, features, gender value, coarse tag]
            # we are ignoring the chunk tag
            splits = line.strip().split(' ')
            if canonicalize:
                word = [utils.canonicalize_word(splits[0]), splits[2], splits[3], self.capitalizaion(splits[0])]
            else:
                word = [splits[0], splits[2], splits[3], self.capitalizaion(splits[0])]
            sentence.append( word)
        
        # don't forget the last sentence
        if len(sentence) > 0:
            sentences.append(sentence)
            sentence = []
        
        if verbose: 
            print ("number of sentences on file =",len(sentences))
            print ("first 5 sentences:")
            print (sentences[:5])

        return sentences
    
    
    def capitalizaion(self, word):
        """
        check capitalization info for a word
        return 'lowercase' for 'sfsd'
        return 'allCaps' for 'SFSD'
        return 'upperInitial' for 'Sfsd'
        return 'mixedCaps' for 'SfSd'
        return 'noinfo' for '$#%@#' or '12334'
        """
        alphas = [c.isalpha() for c in word] 
        if sum(alphas) != len(word):
            return 'noinfo'
        caps = [char.lower()==char for char in word]
        if sum(caps) == len(word):
            return 'lowercase'
        elif sum(caps) == 0:
            return 'allCaps'
        elif caps[0] == False and sum(caps) == len(word)-1:
            return 'upperInitial'
        elif 0 < sum(caps) < len(word):
            return 'mixedCaps'
        else:
            return 'noinfo'    
    
    @timeit
    def buildVocab(self, vocabSize=None, verbose=False, return_vocab_objects=False):
        """
        Builds the vocabulary based on the initial data file
        
        vocabSize(int, default: None-all words) - max number of words to use for vocabulary
                                                  (only used for training)
        verbose(boolean, default: False)        - print extra info
        """    	
        print ("----------------------------------------------------")
        print ("building vocabulary from TRAINING data...")

        flatData = [w for w in zip(*utils.flatten(self.train_sentences))]

        # remember these vocabs will have the <s>, </s>, and <unk> tags in there
        # sizes need to be interpreted "-3" - consider replacing...
        self.vocab = vocabulary.Vocabulary( flatData[0], size=vocabSize)
        self.posTags = vocabulary.Vocabulary( "|".join(flatData[1]).split("|"))
        self.nerTags = vocabulary.Vocabulary( flatData[2])
        self.capitalTags = vocabulary.Vocabulary(flatData[3])

        if verbose:
            # print ("AZ TEST")
            # print (type(self.posTags))
            print ("vocabulary for words, posTags, nerTags built and stored in object")
            print ("vocab size =", vocabSize)
            print ("10 sampled words from vocabulary\n", list(self.vocab.wordset)[:10], "\n")
            print ("number of unique pos Tags in training =", self.posTags.size)
            print ("all posTags used\n", list(self.posTags.wordset), "\n")
            print ("number of unique NER tags in training =", self.nerTags.size)
            print ("all nerTags for prediction", list(self.nerTags.wordset), "\n")
            print ("number of unique capitalization tags in training =", self.capitalTags.size)
            print ('all capitalTags for prediction', list(self.capitalTags.wordset), "\n")

        if return_vocab_objects:
            return self.vocab, self.posTags, self.nerTags, self.capitalTags


    @timeit
    def formatWindowedData(self, sentences, windowLength=9, verbose=False):
        """
        Format the raw data by blocking it into context windows of a fixed length corresponding 
        to the single target NER tag of the central word.
        Make sure to call buildVocab first.
        
        sentences(list of lists of lists) - raw data from the CoNLL2003 dataset
        windowLength(int, default: 9)     - The length of the context window
                    NOTE - windowLength must be odd to have a central word. If itsn't, 1 will be added.
        verbose(boolean, default: False)  - print extra info
        
        Returns: 4 numpy arrays: vocabulary training data windowed and converted to IDs, 
                                 POS tags windowed and converted to IDs,
                                 Capitalization info windowed and converted into IDs,
                                 NER label tags converted to IDs
        """

        print ("----------------------------------------------------")
        print ("formatting sentences into input windows...")

        if windowLength % 2 == 0 or windowLength == 1:
            raise ValueError("window Length must be an odd number and greater than one.")
    
        pads = windowLength // 2

        # we have a list of lists (sentences) of lists ([word, posTag, nerTag])
        # parse through, pad each sentence with pads open and close tags, then convert to IDs
        vocabIDs = [ self.vocab.words_to_ids( ["<s>"] * pads + [word[0] for word in sent] + ["</s>"] * pads) \
                     for sent in sentences]
        # posIDs = [ self.posTags.words_to_ids( ["<s>"] * pads) + self.posTags.words_to_ids([word[1] for word in sent]) + self.posTags.words_to_ids(["</s>"] * pads) \
                   # for sent in sentences]

        posIDs = []
        for sent in sentences:
            padsStart = self.posTags.words_to_ids(["<s>"] * pads)
            padsEnd = self.posTags.words_to_ids(["</s>"] * pads)
            padsStartList = []
            for item in padsStart:
                padsStartList.append([item])
            padsEndList = []
            for item in padsEnd:
                padsEndList.append([item])
            sentFeats = []
            for word in sent:
                wordFeats = word[1].split("|")
                wordFeats = self.posTags.words_to_ids(wordFeats)
                sentFeats.append(wordFeats)
            combined = padsStartList + sentFeats + padsEndList
            posIDs.append(combined)

        capitalIDs = [self.capitalTags.words_to_ids(["<s>"]*pads + [word[3] for word in sent] + ["</s>"]*pads) \
                     for sent in sentences]
        nerIDs = [ self.nerTags.words_to_ids( ["<s>"] * pads + [word[2] for word in sent] + ["</s>"] * pads) \
                   for sent in sentences]
        
        if verbose: 
            print ("STEP 1/2 -- PADDING")
            print ("all sentences padded with {} pads to either end".format(pads))
            print ("vocab idx for first 5 sentences:\n", vocabIDs[:5], "\n")
            print ("pos idx for first 5 sentences:\n", posIDs[:5], "\n")
            print ("AZ TEST", "\n")
            print ("posIDs[0]:\t", posIDs[0], "\n")
            print ("posIDs len:\t", len(posIDs), "\n")
            print ("ner idx for first 5 sentences:\n", nerIDs[:5], "\n")
            print ("capitalization idx for first 5 sentences: \n", capitalIDs[:5], "\n")
            print ("number of sentences = {}".format(len(vocabIDs)), "\n")

        assert(len(vocabIDs) == len(posIDs) and len(posIDs) == len(nerIDs) == len(capitalIDs))

        if verbose: 
            print ("STEP 2/2 -- WINDOWING")

        # build the data to train on by sliding the window across each sentence
        # at this point, all 3 lists are the same size, so we can run through them all at once
        featsVocab, featsPOS, featsNER, featsCAPITAL = [], [], [], []
        for sentID in range( len(vocabIDs)):
            sent = vocabIDs[sentID]
            sentPOS = posIDs[sentID]
            sentNER = nerIDs[sentID]
            sentCAPITAL = capitalIDs[sentID]
            
            for ID in range( len(sent) - windowLength + 1):
                featsVocab.append( sent[ID:ID + windowLength])
                featsPOS.append( sentPOS[ID:ID + windowLength])
                featsCAPITAL.append(sentCAPITAL[ID:ID + windowLength])
                featsNER.append( sentNER[ID + windowLength // 2])
                
        if verbose:
            print ("sample windows:")
            for i in range(3):
                print ("Vocab for window {}".format(i))
                print (featsVocab[i])
                print (self.vocab.ids_to_words(featsVocab[i]))
                print ("PoS tags for window {}".format(i))
                print (featsPOS[i])
                # print (self.posTags.ids_to_words(featsPOS[i]))
                print (self.posTags.ids_to_feats(featsPOS[i]))
                print ("Capitalization tags for window {}".format(i))
                print (featsCAPITAL[i])
                print (self.capitalTags.ids_to_words(featsCAPITAL[i]))
                print ("NER tags for center word")
                print (featsNER[i])
                print (self.nerTags.ids_to_words([featsNER[i]]),"\n")
                
            print ("rows of vocab features = {}".format(len(featsVocab)))
            print ("rows of PoS features = {}".format(len(featsVocab)))
            print ("rows of Capitalization features = {}".format(len(featsCAPITAL)))
            print ("rows of NER features = {}".format(len(featsNER)))
            
            print ("numpy feature arrays are returned")

        assert(len(featsVocab) == len(featsVocab) == len(featsNER) == len(featsCAPITAL))
        return np.array(featsVocab), np.array(featsPOS), np.array(featsCAPITAL), np.array(featsNER)
    
