import numpy as np
import dynet_config
dynet_config.set(
    mem=1024,
    random_seed=1,
    # autobatch=True
)
import dynet as dy

import pickle

class LSTMLM:

    def __init__(self, vocab_size, char_size, char_embedding_dim, char_hidden_size,
        word_embedding_dim, hidden_dim, pos_size, pos_embeddings_size, label_size,
        pattern_hidden_dim, pattern_embeddings_dim, rule_size, max_rule_length,  
        lstm_num_layers, pretrained):
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.hidden_dim = hidden_dim
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)
        self.label_size = label_size
        self.lstm_num_layers = lstm_num_layers
        self.char_hidden_size = char_hidden_size
        self.pos_size = pos_size
        self.pos_embeddings_size = pos_embeddings_size
        self.pretrained = pretrained
        self.pattern_hidden_dim = pattern_hidden_dim
        self.pattern_embeddings_dim = pattern_embeddings_dim
        self.rule_size = rule_size
        self.max_rule_length = max_rule_length
        if np.any(self.pretrained):
            self.word_embeddings = self.model.lookup_parameters_from_numpy(self.pretrained)
        else:
            self.word_embeddings = self.model.add_lookup_parameters((self.vocab_size, self.word_embedding_dim))
        self.pos_embeddings = self.model.add_lookup_parameters((self.pos_size, self.pos_embeddings_size))
        self.char_embeddings = self.model.add_lookup_parameters((self.char_size, self.char_embedding_dim))
        
        self.character_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.char_embedding_dim,
            self.char_hidden_size,
            self.model,
            dy.VanillaLSTMBuilder,
        )
        self.encoder_lstm = dy.BiRNNBuilder(
            self.lstm_num_layers,
            self.word_embedding_dim + self.char_hidden_size + self.pos_embeddings_size,
            self.hidden_dim,
            self.model,
            dy.VanillaLSTMBuilder,
        )

        self.self_attention_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))

        self.query_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))
        self.key_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))
        self.value_weight = self.model.add_parameters((self.hidden_dim, self.hidden_dim))

        self.attention_weight = self.model.add_parameters((self.pattern_hidden_dim, self.hidden_dim))

        self.lb = self.model.add_parameters((self.hidden_dim, 2 * self.hidden_dim))
        self.lb_bias = self.model.add_parameters((self.hidden_dim))

        self.lb2 = self.model.add_parameters((4, self.hidden_dim))
        self.lb2_bias = self.model.add_parameters((4))

        self.pattern_embeddings = self.model.add_lookup_parameters((self.rule_size, self.pattern_embeddings_dim))
        self.decoder_lstm = dy.LSTMBuilder(
            self.lstm_num_layers,
            self.hidden_dim + self.pattern_embeddings_dim,
            self.pattern_hidden_dim,
            self.model,
        )
        self.pt = self.model.add_parameters((self.rule_size, self.pattern_hidden_dim + self.hidden_dim))
        self.pt_bias = self.model.add_parameters((self.rule_size))

        self.gen_c = self.model.add_parameters((1, self.hidden_dim))
        self.gen_h = self.model.add_parameters((1, self.pattern_hidden_dim))
        self.gen_i = self.model.add_parameters((1, self.hidden_dim + self.pattern_embeddings_dim))
        self.gen_bias = self.model.add_parameters((1))


    def save(self, name):
        params = (
            self.vocab_size, self.char_size, self.char_embedding_dim, self.char_hidden_size, 
            self.word_embedding_dim, self.hidden_dim, self.pos_size, self.pos_embeddings_size,
            self.label_size, self.pattern_hidden_dim, self.pattern_embeddings_dim, 
            self.rule_size, self.max_rule_length,
            self.lstm_num_layers, self.pretrained
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = LSTMLM(*params)
            parser.model.populate(f'{name}.model')
            return parser

    def char_encode(self, word):
        c_seq = [self.char_embeddings[c] for c in word]
        return self.character_lstm.transduce(c_seq)[-1]

    def encode_sentence(self, sentence, pos, chars):
        embeds_sent = [dy.concatenate([self.word_embeddings[sentence[i]], self.char_encode(chars[i]), self.pos_embeddings[pos[i]]]) 
         for i in range(len(sentence))]
        features = [f for f in self.encoder_lstm.transduce(embeds_sent)]
        return features

    def self_attend(self, H_e):
        H = dy.concatenate_cols(H_e)
        keys = self.key_weight.expr() * H
        queries = self.query_weight.expr() * H
        values = self.value_weight.expr() * H
        context_vectors = []
        for q in dy.transpose(queries):
            S = dy.transpose(dy.transpose(q) * keys)
            A = dy.softmax(S)
            context_vectors.append(values * A)
            # S = dy.transpose(h_e) * self.self_attention_weight.expr() * H
            # S = dy.transpose(S)
            # A = dy.softmax(S)
            # context_vectors.append(H * A)
        return context_vectors

    def entity_attend(self, H_e, h_e):
        H = dy.concatenate_cols(H_e)
        keys = self.key_weight.expr() * H
        query = self.query_weight.expr() * h_e
        values = self.value_weight.expr() * H
        context_vectors = []
        S = dy.transpose(query) * keys
        A = dy.softmax(S)
        context_vectors = dy.cmult(values, A)
        return dy.transpose(context_vectors)

    def attend(self, H_e, h_t):
        H_e =dy.transpose(H_e)
        S = dy.transpose(h_t) * self.attention_weight.expr() * H_e
        S = dy.transpose(S)
        A = dy.softmax(S)
        context_vector = H_e * A
        return context_vector, A

    def train(self, trainning_set):
        losses = []
        for datapoint in trainning_set: 
            sentence = datapoint[0]
            chars = datapoint[6]
            pos = datapoint[5]
            entity = datapoint[2]
            triggers = datapoint[3]
            rules = datapoint[-1]
            features = self.encode_sentence(sentence, pos, chars)
            labels = datapoint[4]           

            entity_vec = features[entity]
            contexts = self.entity_attend(features, entity_vec)
            
            for i, c in enumerate(contexts):
                if i != entity:
                    h_t = dy.concatenate([c, entity_vec])
                    hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
                    out_vector = dy.softmax(self.lb2.expr() * hidden + self.lb2_bias.expr())
                    if i in triggers:
                        label = labels[triggers.index(i)]
                    else:
                        label = 0
                    losses.append(-dy.log(dy.pick(out_vector, label)))
                    if i in triggers and len(rules[triggers.index(i)])>1:
                        # Get decoding losses
                        last_output_embeddings = self.pattern_embeddings[0]
                        context = c
                        s = self.decoder_lstm.initial_state().add_input(dy.concatenate([context, last_output_embeddings]))
                        for pattern in rules[triggers.index(i)]:
                            h_t = s.output()
                            context, A = self.attend(contexts, h_t)
                            # p_gen = dy.logistic(self.gen_c * context + self.gen_h * h_t + self.gen_i * 
                            #     dy.concatenate([context, last_output_embeddings]) + self.gen_bias)
                            out_vector = self.pt.expr() * dy.concatenate([context, h_t]) + self.pt_bias.expr()
                            probs = dy.softmax(out_vector)
                            losses.append(-dy.log(dy.pick(probs, pattern)))
                            last_output_embeddings = self.pattern_embeddings[pattern]
                            s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            
            try:
                loss = dy.esum(losses)
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                losses = []
            except:
                pass

    def decode(self, features, c):
        last_output_embeddings = self.pattern_embeddings[0]
        s = self.decoder_lstm.initial_state().add_input(dy.concatenate([c, last_output_embeddings]))
        out = []
        for i in range(self.max_rule_length):
            h_t = s.output()
            context, A = self.attend(features, h_t)
            out_vector = self.pt.expr() * dy.concatenate([context, h_t]) + self.pt_bias.expr()
            probs = dy.softmax(out_vector).vec_value()
            last_output = probs.index(max(probs))
            last_output_embeddings = self.pattern_embeddings[last_output]
            s = s.add_input(dy.concatenate([context, last_output_embeddings]))
            if last_output != 0:
                out.append(last_output)
            else:
                return out
        return out

    def get_pred(self, sentence, pos,chars, entity):
        valid_tags = ['NN', 'VB', 'VBZ', 'VBD', 'JJ', 'VBG', 'NNS', 'VBP', 'VBN']
        features = self.encode_sentence(sentence, pos, chars)
        entity_embeds = features[entity]
        entity_vec = features[entity]
        contexts = self.entity_attend(features, entity_vec)
        res = []
        rules = []
        for i, c in enumerate(contexts):
            if i != entity:
                h_t = dy.concatenate([c, entity_vec])
                hidden = dy.tanh(self.lb.expr() * h_t + self.lb_bias.expr())
                out_vector = dy.softmax(self.lb2.expr() * hidden + self.lb2_bias.expr()).vec_value()
                # print (out_vector.npvalue())
                l = out_vector.index(max(out_vector))
                if l != 0:
                    res.append((i, l))
                    rules.append(self.decode(contexts, c))
        return res, out_vector, contexts, hidden, rules