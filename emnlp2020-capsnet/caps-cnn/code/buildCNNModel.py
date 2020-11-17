
import tensorflow as tf
import keras as K

from keras import callbacks, optimizers
from keras import backend as KB
from keras.engine import Layer
from keras.layers import Activation
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Reshape, Concatenate, MaxPooling1D, Flatten
from keras.layers import Flatten, SpatialDropout1D, Conv1D
from keras.models import Model
from keras.utils import plot_model

from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Add

# capsule layers from Xifeng Guo 
# https://github.com/XifengGuo/CapsNet-Keras
from capsulelayers import CapsuleLayer, PrimaryCap1D, Length, Mask


# JUST DUPLICATE CODE, NO ATTEMPT TO RUN YET
# Build the CNN model

def draw_cnn_model(hyper_param, embedding_matrix=None ,verbose=True):
    """
    Input: hyper_parameters dictionary
    
    Construct:
        input layers : x , x_pos(o), x_captialization(o)
        embedding matrix : use_glove or randomly initialize
        conv1 : first convolution layer
        conv2 : second convolution layer
        conv3 : third convolution layer
        max pooling
        flatten : concant maxpooled univariate vectors into one long vector
        ff1, ff2: two feed forward layers
        out_pred: softmax over all ner classes
    
    Returns: keras.models.Model object
    """

    # input layer(s)
    x = Input(shape=(hyper_param['maxlen'],), name='x')
    if hyper_param['use_pos_tags'] : 
        x_pos = Input(shape=(hyper_param['maxlen'],hyper_param['poslen']), name='x_pos')
    if hyper_param['use_capitalization_info'] : 
        x_capital = Input(shape=(hyper_param['maxlen'], hyper_param['capitallen']), name='x_capital') 
    
    # embedding matrix
    if hyper_param['use_glove']:
        embed = Embedding(hyper_param['max_features'], hyper_param['embed_dim'], weights=[embedding_matrix],\
                          input_length=hyper_param['maxlen'], trainable=hyper_param['allow_glove_retrain'])(x)
    else:
        embed = Embedding(hyper_param['max_features'], hyper_param['embed_dim'], input_length=hyper_param['maxlen'],\
                          embeddings_initializer="random_uniform" )(x)

    # concat embeddings with additional features
    if hyper_param['use_pos_tags'] and hyper_param['use_capitalization_info'] : 
        embed = Concatenate(axis=-1)([embed, x_pos, x_capital])
    elif hyper_param['use_pos_tags'] and (not hyper_param['use_capitalization_info']) :
        embed = Concatenate(axis=-1)([embed, x_pos])
    elif (not hyper_param['use_pos_tags']) and hyper_param['use_capitalization_info'] :
        embed = Concatenate(axis=-1)([embed, x_capital])    
    else :
        embed = embed
    
    # feed embeddings into conv1
    conv1 = Conv1D( filters=hyper_param['conv1_filters'], \
                   kernel_size=hyper_param['conv1_kernel_size'],\
                   strides=hyper_param['conv1_strides'], \
                   padding=hyper_param['conv1_padding'],\
                   activation='relu', name='conv1')(embed)
    
    # update this
    # make primary capsules
    conv2 = Conv1D( filters=hyper_param['conv2_filters'], \
                   kernel_size=hyper_param['conv2_kernel_size'],\
                   strides=hyper_param['conv2_strides'], \
                   padding=hyper_param['conv2_padding'],\
                   activation='relu', name='conv2')(conv1)    
    
    # update this
    # make primary capsules
    conv3 = Conv1D( filters=hyper_param['conv3_filters'], \
                   kernel_size=hyper_param['conv3_kernel_size'],\
                   strides=hyper_param['conv3_strides'], \
                   padding=hyper_param['conv3_padding'],\
                   activation='relu', name='conv3')(conv2)    
    
    # max pooling layer
    max_pooled = MaxPooling1D(pool_size=hyper_param['max_pooling_size'], \
                              strides=hyper_param['max_pooling_strides'], \
                              padding=hyper_param['max_pooling_padding'])(conv3)
    # dropout
    maxpooled_dropout = Dropout(hyper_param['maxpool_dropout'])(max_pooled)
    
    # flatten many univariate vectos into 1 long vector
    flattened = Flatten()(maxpooled_dropout)
    
    # to feed-forward layers
    ff1 = Dense(hyper_param['feed_forward_1'], activation='relu')(flattened)    
    ff1_dropout = Dropout(hyper_param['ff1_dropout'])(ff1)
    
    ff2 = Dense(hyper_param['feed_forward_2'], activation='relu')(ff1_dropout)    
    ff2_dropout = Dropout(hyper_param['ff2_dropout'])(ff2)    
    
    out_pred = Dense(hyper_param['ner_classes'], activation='softmax', name='out_pred')(ff2) #!
    
             
    if verbose:
        print ("x", x.get_shape())
        if hyper_param['use_pos_tags'] : print ("x_pos", x_pos.get_shape())
        if hyper_param['use_capitalization_info'] : print ("x_capital", x_capital.get_shape())
        print ("embed", embed.get_shape())
        print ("embed", embed.get_shape())
        
        print ("conv1", conv1.get_shape())
        print ("conv2", conv2.get_shape())   
        print ("conv3", conv3.get_shape())
        print ("max_pooled", max_pooled.get_shape())
        print ("flattened", flattened.get_shape())   
        print ("ff1", ff1.get_shape())
        print ("ff2", ff2.get_shape())
        print ("out_pred", out_pred.get_shape())        
  
    # return final model
    if hyper_param['use_pos_tags'] and hyper_param['use_capitalization_info'] : 
        cnnmodel = Model(inputs=[x, x_pos, x_capital], outputs=[out_pred])
    elif hyper_param['use_pos_tags'] and (not hyper_param['use_capitalization_info']) :
        cnnmodel = Model(inputs=[x, x_pos], outputs=[out_pred])
    elif (not hyper_param['use_pos_tags']) and hyper_param['use_capitalization_info'] :
        cnnmodel = Model(inputs=[x, x_capital], outputs=[out_pred])   
    else :
        cnnmodel = Model(inputs=[x], outputs=[out_pred])
        
    return cnnmodel


# compile the model
def compile_cnn_model(hyper_param, model): #!
    """
    Input: keras.models.Model object, see draw_capsnet_model() output. This is a graph with all layers drawn and connected
    
    do:
        compile with loss function and optimizer
    
    Returns: compiled model
    """
    if hyper_param['optimizer'] == "Adam":
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif hyper_param['optimizer'] == "SGD": 
        opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    elif hyper_param['optimizer'] == None:
        raise Exception("No optimizer specified")

    model.compile(optimizer=opt, #'adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


def fit_model( hyper_param, model, modelName, trainX_dict, devX_list_arrayS, trainY_dict, devY_list_arrayS):
    #Saving weights and logging
    log = callbacks.CSVLogger(hyper_param['save_dir'] + '/{0}_historylog.csv'.format(modelName))
    tb = callbacks.TensorBoard(log_dir=hyper_param['save_dir'] + '/tensorboard-logs', \
                               batch_size=hyper_param['batch_size'], histogram_freq=hyper_param['debug'])
    checkpoint = callbacks.ModelCheckpoint(hyper_param['save_dir'] + '/weights-{epoch:02d}.h5', \
                                           save_best_only=True, save_weights_only=True, verbose=1)
    es = callbacks.EarlyStopping(patience=hyper_param['stopping_patience'], verbose=2)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    model.summary()
    
    # Save a png of the model shapes and flow
    # must have installed pydot and graphviz...
    # conda install pydot
    # conda install -c anaconda graphviz
    # sometimes graphviz is a little squirrely, if so, use: pip install graphviz
    # plot_model( model, to_file=hyper_param['save_dir'] + '/{0}.png'.format(modelName), show_shapes=True)

    #loss = margin_loss
    
    data = model.fit( x=trainX_dict, # {'x':trainX, 'x_pos':trainX_pos_cat, 'x_capital':trainX_capitals_cat, (o)'decoder_y':trainY_cat}
                      y=trainY_dict, #!{'out_pred':trainY_cat, (o)'decoder_output':train_decoderY}
                      batch_size=hyper_param['batch_size'], 
                      epochs=hyper_param['epochs'], 
                      validation_data=[devX_list_arrayS, devY_list_arrayS], #! [devX, devX_pos_cat, devX_capitals_cat, (o)devY_cat], [devY_cat, (o)dev_decoderY]
                      callbacks=[log, tb, checkpoint, es], 
                      verbose=1)

