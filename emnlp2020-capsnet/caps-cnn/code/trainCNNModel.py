
import sys
import json
import buildCNNModel as cnn
from loadutils import retrieve_model, loadProcessedData, saveDevPredictionsData
from evaluation_helper import convert_raw_y_pred, get_f1, get_precision, get_recall
import numpy as np

def printUsage():
    print("USAGE:\n\ntrain a CNN model")
    print("All training data must have already been saved with loadutils.saveProcessedData()")
    print("<model name> <hyper parameters file (JSON)> ")


def main():
    """
    command line arguments:
    <model name> <hyper parameters file (JSON)> 
    """
    
    if len(sys.argv) < 3:
        printUsage()
        return -1
    
    
    modelName = sys.argv[1]
    
    with open(sys.argv[2]) as fp:
        hypers = json.load( fp)

    DIRECTORY = sys.argv[3]
    
    trainX, trainX_capitals_cat, trainX_pos_cat, devX, devX_capitals_cat, \
           devX_pos_cat, trainY_cat, devY_cat, embedding_matrix, train_decoderY, dev_decoderY = loadProcessedData(DIRECTORY)
    
    # contruct training dicts
    trainX_dict = {'x':trainX}
    devX_list_arrayS = [devX]
    trainY_dict = {'out_pred':trainY_cat}
    devY_list_arrayS = [devY_cat]
    
    # for final prediction
    devX_dict = {'x':devX} #for model_eval only    
    
    if hypers["use_pos_tags"]:
        trainX_dict["x_pos"] = trainX_pos_cat
        devX_list_arrayS += [devX_pos_cat]
        devX_dict["x_pos"] = devX_pos_cat #for model_eval only
        
    if hypers['use_capitalization_info']:
        trainX_dict["x_capital"] = trainX_capitals_cat
        devX_list_arrayS += [devX_capitals_cat]
        devX_dict["x_capital"] = devX_capitals_cat  #for model_eval only
        
    model = cnn.draw_cnn_model( hyper_param=hypers, embedding_matrix=embedding_matrix, verbose=True)
    model = cnn.compile_cnn_model( hypers, model)
    
    print( "Training Model:", modelName)
    cnn.fit_model( hypers, model, modelName, trainX_dict, devX_list_arrayS, trainY_dict, devY_list_arrayS)

    # save the last model in each epoch and its weights
    with open('./result/'+ modelName + '_model_architecture.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('./result/' + modelName + '_weights_model.h5')    
    
    raw_y_pred = model.predict(devX_dict, verbose=1)    
    
    y_true = convert_raw_y_pred(devY_cat)
    print ("prediction on dev set finished. raw 1-hot prediction has shape {}".format(raw_y_pred.shape))
    y_pred = convert_raw_y_pred(raw_y_pred)
    print ("prediction converted to class idx has shape {}".format(y_pred.shape))
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    f1_score = get_f1(y_true, y_pred)
    print ("precision on dev = {}".format(precision))
    print ("recall on dev = {}".format(recall))
    print ("f1 score on dev = {}".format(f1_score))

    # write out dev predictions
    modelsDir = 'CapsNet_for_NER/code/dev_Predictions'
    print ("saving prediction data under directory: {}".format(modelsDir))
    saveDevPredictionsData(modelName=modelName, raw_y_pred=raw_y_pred, raw_y_pred_decoder_embeddings=np.empty(0), y_pred=y_pred, modelsDir=modelsDir)
    print ("please use loadutils.loadDevPredictionsData(modelName, modelsDir='dev_Predictions') to load :\n raw_y_pred, raw_y_pred_decoder_embeddings(empty array for CNN), y_pred")
    
if __name__ == '__main__':
    main()
    
