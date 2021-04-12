import dynet as dy
from collections import namedtuple
import numpy as np
from utils import *


ModelElements = namedtuple("ModelElements", "W V b w2v_emb c2v_embd param_collection builder")

ModelElements_1 = namedtuple("ModelElements", "W V b w2v_emb c2v_embd param_collection builder builder_char_fwd, builder_char_bwd")

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
    #output_bwd = gru_char_bwd.transduce(reversed(char_embd_list))   # this is the same as the expression above

    # print('length of input:', len(output_fwd))
    # print('length of output:', len(output_bwd))

    char_embd_vec = dy.concatenate([output_fwd[-1], output_bwd[-1]],d=0)

    # print('char embd vec dim:', char_embd_vec.dim())

    # input('press enter to continue')

    return char_embd_vec


def run_instance(instance, model_elems, embeddings, char_embeddings, char_embd_sel):

    # Renew the computational graph
    dy.renew_cg()

    builder = model_elems.builder
    builder.set_dropouts(0, 0)   # currently 0.2, 0.2 gives the best result
    
    W = model_elems.W
    V = model_elems.V
    b = model_elems.b

    if char_embd_sel==0:
        collected_vectors = list()

        # it is ok to have empty embedding in the token sequence. 
        
        inputs = [embeddings[w] for w in instance.get_tokens()] # in Enrique's master branch code, he uses get_toekn
        lstm = builder.initial_state()
        outputs = lstm.transduce(inputs)

        # Get the last embedding
        selected = outputs[-1]
        
        trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)
        #trigger_expression = dy.scalarInput(0)

        ff_input = dy.concatenate([trigger_expression, selected])
        #ff_input = selected

        # Run the FF network for classification
        prediction = dy.logistic(V * (W * ff_input + b))

        return prediction

    elif char_embd_sel==1:

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
        lstm = builder.initial_state()
        outputs = lstm.transduce(inputs)

        # Get the last embedding
        selected = outputs[-1]
        
        trigger_expression = dy.scalarInput(1 if instance.rule_polarity is True else 0)

        ff_input = dy.concatenate([trigger_expression, selected])

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


def build_model(w2v_embeddings, char_embeddings, word_embd_sel, char_embd_sel):
    WEM_DIMENSIONS = 100
    CEM_DIMENSIONS = 20


    NUM_LAYERS = 1
    HIDDEN_DIM = 30

    FF_HIDDEN_DIM = 10

    params = dy.ParameterCollection()

    if word_embd_sel==0:
        w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, name="w2v-wemb")

    elif word_embd_sel==1:
        w2v_wemb = params.add_lookup_parameters(w2v_embeddings.matrix.shape, init=w2v_embeddings.matrix, name="w2v-wemb")
    c2v_embd = params.add_lookup_parameters((len(char_embeddings)+1, CEM_DIMENSIONS), name="c2v-emb")



    # Feed-Forward parameters
    W = params.add_parameters((FF_HIDDEN_DIM, HIDDEN_DIM+1), name="W")
    b = params.add_parameters((FF_HIDDEN_DIM), name="b")
    V = params.add_parameters((1, FF_HIDDEN_DIM), name="V")
    
    # no attention
    if char_embd_sel==0:
        builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS, HIDDEN_DIM, params)

        ret = ModelElements(W, V, b, w2v_wemb, c2v_embd, params, builder)

    elif char_embd_sel==1:
        builder = dy.LSTMBuilder(NUM_LAYERS, WEM_DIMENSIONS+CEM_DIMENSIONS*2, HIDDEN_DIM, params)
        builder_char_fwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)
        builder_char_bwd = dy.GRUBuilder(NUM_LAYERS, CEM_DIMENSIONS, CEM_DIMENSIONS, params)


        ret = ModelElements_1(W, V, b, w2v_wemb, c2v_embd, params, builder, builder_char_fwd, builder_char_bwd)

    return ret
