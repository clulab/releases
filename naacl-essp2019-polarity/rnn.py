import dynet as dy
from collections import namedtuple
import numpy as np
from utils import *

ModelElements = namedtuple("ModelElements", "W V b w2v_emb c2v_embd param_collection builder_fwd builder_bwd builder_char_fwd, builder_char_bwd")

#ModelElements_2 = namedtuple("ModelElements", "W V b W_a b_a W_a_2 b_a_2 w2v_emb param_collection builder")
def get_char_embd(word, model_elems, embeddings_char_index):
    gru_char_fwd = model_elems.builder_char_fwd.initial_state()
    gru_char_bwd = model_elems.builder_char_bwd.initial_state()

    #print('current word:', word)
    #print('number of characters in word:',len(word))

    if word=='':
        char_embd_list = [embeddings_char_index['']]

    else:
        char_embd_list = list([])
        for character in word:
            char_embd_list.append(embeddings_char_index[character])

    # print('forward char:', char_embd_list)
    # print('backward char:', char_embd_list[::-1])

    output_fwd = gru_char_fwd.transduce(char_embd_list)
    output_bwd = gru_char_bwd.transduce(char_embd_list[::-1])

    # print('length of input:', len(output_fwd))
    # print('length of output:', len(output_bwd))

    char_embd_vec = dy.concatenate([output_fwd[-1], output_bwd[-1]],d=0)

    # print('char embd vec dim:', char_embd_vec.dim())

    # input('press enter to continue')

    return char_embd_vec


def run_instance(instance, model_elems, embeddings, char_embeddings):

    # Renew the computational graph
    dy.renew_cg()

    builder_fwd = model_elems.builder_fwd
    builder_bwd = model_elems.builder_bwd
    #builder.set_dropouts(0, 0)   # currently 0.2, 0.2 gives the best result
    
    W = model_elems.W
    V = model_elems.V
    b = model_elems.b

    inputs = list([])
    #print('tokens of the sentence')
    #print([token for token in instance.tokens])
    # print('===new instance====')
    # print(instance.tokens)
    for word in instance.get_tokens():
        word_embd = embeddings[word]
        char_embd = get_char_embd(word, model_elems, char_embeddings)
        input_vec = dy.concatenate([word_embd,char_embd], d=0)

        #print('input vec dim:', input_vec.dim())

        inputs.append(input_vec)
    lstm_fwd = builder_fwd.initial_state()
    lstm_bwd = builder_bwd.initial_state()
    
    outputs_fwd = lstm_fwd.transduce(inputs)
    outputs_bwd = lstm_bwd.transduce(inputs[::-1])

    # Get the last embedding
    selected_fwd = outputs_fwd[-1]
    selected_bwd = outputs_bwd[-1]
    
    trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

    ff_input = dy.concatenate([trigger_expression, selected_fwd, selected_bwd])

    # Run the FF network for classification
    prediction = dy.logistic(V * (W * ff_input + b))

    #input('press enter to continue')

    # input('press enter to continue')

    return prediction



def prediction_loss(instance, prediction):
    # Compute the loss
    y_true = dy.scalarInput(1 if instance.polarity else 0)
    loss = dy.binary_log_loss(prediction, y_true)

    return loss


def build_model(w2v_embeddings, char_embeddings):
    WEM_DIMENSIONS = 100
    CEM_DIMENSIONS = 20


    NUM_LAYERS = 1
    HIDDEN_DIM = 30

    FF_HIDDEN_DIM = 10

    params = dy.ParameterCollection()


    w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, name="w2v-wemb")

    c2v_embd = params.add_lookup_parameters((len(char_embeddings)+1, CEM_DIMENSIONS), name="c2v-emb")



    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM*2+1), name="W")
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")

    builder_fwd = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS+CEM_DIMENSIONS*2, HIDDEN_DIM, params)
    builder_bwd = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS+CEM_DIMENSIONS*2, HIDDEN_DIM, params)
    builder_char_fwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)
    builder_char_bwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)


    ret = ModelElements(W, V, b, w2v_wemb, c2v_embd, params, builder_fwd, builder_bwd, builder_char_fwd, builder_char_bwd)

    return ret
