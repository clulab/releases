
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
from capsulelayers import CapsuleLayer, PrimaryCap, PrimaryCap1D, Length, Mask



# Build the CapsNet model

def draw_capsnet_model(hyper_param, embedding_matrix=None, verbose=True):
    """
    Input: hyper parameters dictionary
    
    Construct:
        input layers : x , x_pos(o), x_captialization(o)
        embedding matrix : use_glove or randomly initialize
        conv1 : first convolution layer
        primarycaps : conv2 and squash function applied
        ner_caps : make 8 ner capsules of specified dim
        out_pred : calc length of 8 ner capsules as 8 prob. predictions over 8 ner classes
    
    Returns: 
        if decoding/reconstruction disabled --> a single keras.models.Model object
        if decoding/reconstruction enabled --> three keras.models.Model objects
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
 
    # add dropout here
    if hyper_param['embed_dropout'] > 0.0:
        embed = SpatialDropout1D( hyper_param['embed_dropout'])(embed)

    # feed embeddings into conv1
    conv1 = Conv1D( filters=hyper_param['conv1_filters'], \
                   kernel_size=hyper_param['conv1_kernel_size'],\
                   strides=hyper_param['conv1_strides'], \
                   padding=hyper_param['conv1_padding'],\
                   activation='relu', name='conv1')(embed)

    # make primary capsules
    if hyper_param['use_2D_primarycaps']:
        convShape = conv1.get_shape().as_list()
        conv1 = Reshape(( convShape[1], convShape[2], 1))(conv1)
        primaryCapLayer = PrimaryCap
    else:
        primaryCapLayer = PrimaryCap1D    
    
    # make primary capsules
    primarycaps = primaryCapLayer(conv1, \
                             dim_capsule=hyper_param['primarycaps_dim_capsule'],\
                             n_channels=hyper_param['primarycaps_n_channels'],\
                             kernel_size=hyper_param['primarycaps_kernel_size'], \
                             strides=hyper_param['primarycaps_strides'], \
                             padding=hyper_param['primarycaps_padding'])

    # make ner capsules
    ner_caps = CapsuleLayer(num_capsule=hyper_param['ner_classes'], \
                            dim_capsule=hyper_param['ner_capsule_dim'], \
                            routings=hyper_param['num_dynamic_routing_passes'], \
                            name='nercaps')(primarycaps)

    # replace each ner capsuel with its length
    out_pred = Length(name='out_pred')(ner_caps)


    if verbose:
        print ("x", x.get_shape())
        if hyper_param['use_pos_tags'] : print ("x_pos", x_pos.get_shape())
        if hyper_param['use_capitalization_info'] : print ("x_capital", x_capital.get_shape())
        print ("embed", embed.get_shape())
        print ("conv1", conv1.get_shape())
        print ("primarycaps", primarycaps.get_shape())   
        print ("ner_caps", ner_caps.get_shape())
        print ("out_pred", out_pred.get_shape())


    if hyper_param['use_decoder']:
        decoder_y_cat = Input(shape=(hyper_param['ner_classes'],), name='decoder_y_cat')
        masked_by_y = Mask(name='masked_by_y')([ner_caps, decoder_y_cat]) # true label is used to mask during training
        masked = Mask()(ner_caps) # mask using capsule with maximal length for predicion

        # decoder for training 
        train_decoder_dense1 = Dense(hyper_param['decoder_feed_forward_1'], activation='relu',\
                               input_dim=hyper_param['ner_capsule_dim']*hyper_param['ner_classes'],\
                               name='train_decoder_dense1')(masked_by_y)
        train_decoder_dense1_dropout = Dropout(hyper_param['decoder_dropout'])(train_decoder_dense1)
        train_decoder_dense2 = Dense(hyper_param['decoder_feed_forward_2'], activation='relu',\
                                     name='train_decoder_dense2')(train_decoder_dense1_dropout)
        train_decoder_dense2_dropout = Dropout(hyper_param['decoder_dropout'])(train_decoder_dense2)
        train_decoder_output = Dense(hyper_param['embed_dim'], activation=None,\
                                     name='train_decoder_output')(train_decoder_dense2_dropout)


        # decoder for evaluation (prediction) 
        eval_decoder_dense1 = Dense(hyper_param['decoder_feed_forward_1'], activation='relu',\
                               input_dim=hyper_param['ner_capsule_dim']*hyper_param['ner_classes'],\
                               name='eval_decoder_dense1')(masked)
        eval_decoder_dense2 = Dense(hyper_param['decoder_feed_forward_2'], activation='relu',\
                                     name='eval_decoder_dense2')(eval_decoder_dense1)
        eval_decoder_output = Dense(hyper_param['embed_dim'], activation=None,\
                                     name='eval_decoder_output')(eval_decoder_dense2)
        
        
        if verbose:
            print ("Decoder model enabled for GloVe vector deconstruction...")
            print ("decoder_y_cat", decoder_y_cat.get_shape())
            print ("masked_by_y", masked_by_y.get_shape())
            print ("train_decoder_dense1", train_decoder_dense1.get_shape())
            print ("train_decoder_dense1_dropout", train_decoder_dense1_dropout.get_shape())
            print ("train_decoder_dense2", train_decoder_dense2.get_shape())
            print ("train_decoder_dense2_dropout", train_decoder_dense2_dropout.get_shape())
            print ("train_decoder_output", train_decoder_output.get_shape())
            print ("masked", masked.get_shape())
            print ("eval_decoder_dense1", eval_decoder_dense1.get_shape())
            print ("eval_decoder_dense2", eval_decoder_dense2.get_shape())
            print ("eval_decoder_output", eval_decoder_output.get_shape())

    # construct input list
    if hyper_param['use_pos_tags'] and hyper_param['use_capitalization_info'] : 
        input_list = [x, x_pos, x_capital]
    elif hyper_param['use_pos_tags'] and (not hyper_param['use_capitalization_info']) :
        input_list = [x, x_pos]
    elif (not hyper_param['use_pos_tags']) and hyper_param['use_capitalization_info'] :
        input_list = [x, x_capital]
    else:
        input_list = [x]


    if hyper_param['use_decoder']==False:
        print ("decoder/reconstruction DISabled")
        print ("returning 1 model")
        return Model(inputs=input_list, outputs=[out_pred])
    else :
        train_model = Model(inputs=input_list+[decoder_y_cat], outputs=[out_pred, train_decoder_output])
        eval_model = Model(inputs=input_list, outputs=[out_pred, eval_decoder_output])
        print ("decoder/reconstruction enabled")
        print ("returning a list of 2 models: train_model, eval_model")
        return train_model, eval_model


# marginal loss
def margin_loss(y_true, y_pred):
    L = y_true * KB.square(KB.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * KB.square(KB.maximum(0., y_pred - 0.1))
    return KB.mean(KB.sum(L, 1))


# decoder loss
def custom_cosine_proximity(y_true, y_pred):
    y_true = tf.nn.l2_normalize(y_true, dim=-1)
    y_pred = tf.nn.l2_normalize(y_pred, dim=-1)
    return -KB.sum(y_true * y_pred)


# compile the model
def compile_caps_model(hyper_param, model):
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

    if hyper_param.get('use_decoder') == True:
        if hyper_param['loss_function'] == 'custom_cosine':
            decodeLoss = custom_cosine_proximity
        else:
            decodeLoss = hyper_param['loss_function']
        
        model_loss = [margin_loss, decodeLoss] # work in progress
        loss_wts = [1, hyper_param['lam_recon']]
    else:
        model_loss = margin_loss
        loss_wts = None
    
    model.compile(optimizer=opt, #'adam',
                  loss=model_loss,
                  loss_weights=loss_wts,
                  metrics={'out_pred':'accuracy'})
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
    
    data = model.fit( x=trainX_dict, # {'x':trainX, 'x_pos':trainX_pos_cat, 'x_capital':trainX_capitals_cat, (o)'decoder_y_cat':trainY_cat}
                      y=trainY_dict, #!{'out_pred':trainY_cat, (o)'decoder_output':train_decoderY}
                      batch_size=hyper_param['batch_size'], 
                      epochs=hyper_param['epochs'], 
                      validation_data=[devX_list_arrayS, devY_list_arrayS], #! [devX, devX_pos_cat, devX_capitals_cat, (o)devY_cat], [devY_cat, (o)dev_decoderY]
                      callbacks=[log, tb, checkpoint, es], 
                      verbose=1)





