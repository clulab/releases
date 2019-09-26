import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import io
from . import data
from .utils import export
import sys
from .processNLPdata.processNECdata import *
import os
import contextlib
import json
import logging
import re
words_in_glove =0
DEFAULT_ENCODING = 'utf8'
from tqdm import tqdm




@export
def fever():

    if RTEDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(RTEDataset.NUM_WORDS_TO_REPLACE, RTEDataset.OOV, RTEDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(RTEDataset.WORD_NOISE_TYPE)

    return {
        #'train_transformation': data.TransformTwiceNEC(addNoise),
        'train_transformation': None,
        'eval_transformation': None,
        #'datadir': 'data-local/rte/fever'
        #ask ajay what does this do? why comment out?
        # 'num_classes': 11
    }

class RTEDataset(Dataset):

    PAD = "<pad>"
    UNKNOWN = "<unk>"
    OOV = "</s>"
    ENTITY = "@ENTITY"
    OOV_ID = 0
    ENTITY_ID = -1
    NUM_WORDS_TO_REPLACE = 1
    WORD_NOISE_TYPE = "drop"

    LOG = logging.getLogger('main')
    LOG.setLevel(logging.INFO)

    def get_word_from_vocab_dict_given_word_id(self, word_id):
        return self.word_vocab_id_to_word[word_id]

    def sanitise_and_lookup_embedding(self, word_id):
        word_original = Gigaword.sanitiseWord(self.get_word_from_vocab_dict_given_word_id(word_id))

            #used to have .lower()- performance was less. so commented it out

        if word_original in self.lookupGiga:
            #todo : not sure what Gigaword.norm is doing. This is from ajay/valpola code.
            #  commenting it out and loading glove vectors directly on march 29th2019 until i find out what it does and if we need it.
            #word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga[word_original]])
            word_embed = self.gigaW2vEmbed[self.lookupGiga[word_original]]
        else:
            #word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<unk>"]])
            word_embed = self.gigaW2vEmbed[self.lookupGiga["<unk>"]]

        self.LOG.debug(f"word:[{word_original} \t embedding:{word_embed}")
        return word_embed

    def create_word_vocab_embed(self):
        word_vocab_embed = list()
        # leave last word = "@PADDING"
        counter=0
        all_words=range(0, len(self.word_vocab))
        for word_id in tqdm(all_words,total=len(self.word_vocab)):
            word_embed = self.sanitise_and_lookup_embedding(word_id)
            word_vocab_embed.append(word_embed)
            counter=counter+1

        # NOTE: adding the embed for @PADDING
        #word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
        return np.array(word_vocab_embed).astype('float32')


    # create a mapping from word to id from id to word
    def map_id_to_word(self,word_vocab):
        word_vocab_id_to_word_maping={}
        for word,id in word_vocab.items():
            word_vocab_id_to_word_maping[id]=word

        return word_vocab_id_to_word_maping

    #mithun this is called using:#dataset = datasets.NECDataset(traindir, args, train_transformation)
    def __init__(self, word_vocab,runName,dataset_file, args,emb_file_path,transform=None):
        LOG = logging.getLogger('datasets')
        LOG.setLevel(logging.INFO)

        print("got inside init of RTE data set")

        #if(args.type_of_data=="plain"):
        self.claims, self.evidences, self.labels_str = Datautils.read_data_where_evidences_are_strings(dataset_file,args)
        #else:
           # if (args.type_of_data == "ner_replaced"):
           #     self.claims, self.evidences, self.labels_str = Datautils.read_data_where_evidences_are_strings(dataset_file, args)



        assert len(self.claims)== len(self.evidences)==len(self.labels_str), "claims and evidences are not of equal length"

        #to find the top 10 longest evidences adn remove them. am doing this because GPU was getting memory overloaded because of padding
        list_of_longest_ev_lengths=[]
        list_of_longest_evidences=[]
        max_evidence_len=0
        print(
            f"going to load dataset")

        for each_ev in self.evidences:
            words = [w for w in each_ev.split(" ")]
            if len(words) > max_evidence_len:
                    max_evidence_len = len(words)
                    longest_evidence_words = words
                    list_of_longest_ev_lengths.append(max_evidence_len)
                    list_of_longest_evidences.append(longest_evidence_words)

        '''The dictionary of words in training (vocabulary) should not be updated while reading dev. 
               However, while in dev, if you see a word that exists in the vocabulary, return its index. 
               But if its a new word, don’t update the vocabulary . Instead return the index of dictionary.
               '''


        self.word_vocab, self.max_claims_len, self.max_ev_len, self.word_count= self.get_max_lengths_add_to_vocab(word_vocab,runName,args)
        self.word_vocab_id_to_word={}

        print(f"inside datasets.py . just after  build_word_vocabulary.value of word_vocab.size()={len(self.word_vocab.keys())}")





        # #remove least frequent words
        # for word in self.word_counts:
        #     if self.word_counts[word] >= args.word_frequency:
        #         self.word_vocab.add(word, self.word_counts[word])



        if (RTEDataset.PAD not in self.word_vocab):
            self.word_vocab[RTEDataset.PAD] = 0

        self.pad_id = self.word_vocab[RTEDataset.PAD]

        # create a mapping from word to id from id to word

        self.word_vocab_id_to_word=self.map_id_to_word(self.word_vocab)


        if args.pretrained_wordemb:
            if not runName == "dev": # do not load the word embeddings again in eval
                sys.stdout.write("going to Loading the pretrained embeddings ... ")
                LOG.info("going to Loading the pretrained embeddings ... ")
                sys.stdout.flush()
                self.gigaW2vEmbed, self.lookupGiga, self.embedding_size = Gigaword.load_pretrained_embeddings(emb_file_path,args)
                LOG.info("Done loading embeddings. going to create vocabulary ... ")
                sys.stdout.write("Done loading embeddings. going to create vocabulary ... " )
                sys.stdout.flush()
                self.word_vocab_embed = self.create_word_vocab_embed()

        else:
            LOG.info("Not loading the pretrained embeddings ... ")
            assert args.update_pretrained_wordemb, "Pretrained embeddings should be updated but " \
                                                   "--update-pretrained-wordemb = {}".format(args.update_pretrained_wordemb)
            self.word_vocab_embed = None




        print("1self.word_vocab.size=", len(self.word_vocab.keys()))

        self.categories = sorted(list({l for l in self.labels_str}))
        self.lbl = [self.categories.index(l) for l in self.labels_str]



        LOG.debug(f"inside dataset.py just after Datautils.read_rte_data. size of self.claism is:{len(self.claims)}")


        #write the vocab file to disk so that you can load it later
        #update: vocab file for dev is same as train. no need to write it twice.

        print("2self.word_vocab.size=", len(self.word_vocab.keys()))
        dir = args.output_folder
        if(runName=="train"):
            vocab_file = dir + 'vocabulary_'+ 'train' + '.txt'
            with io.open(vocab_file, 'w+', encoding=DEFAULT_ENCODING) as f:
                f.write(json.dumps(self.word_vocab))



        print("num of types of labels considered =", len(self.categories))

        #write the list of labels to disk
        label_category_file = dir + 'label_category_'  + runName+'.txt'
        with io.open(label_category_file, 'w', encoding='utf8') as f:
            for lbl in self.categories:
                f.write(lbl + '\n')

        gold_labels_file = dir + 'label_gold_' + runName + '.txt'
        with io.open(gold_labels_file, 'w', encoding='utf8') as f:
            for lbl in self.lbl:
                f.write(str(lbl)+"\n")

        self.transform = transform
        print("4self.word_vocab.size=", len(self.word_vocab.keys()))


    def update_word_count(self, dict_wc,word ):
        if(word in dict_wc.keys()):
            old_count=dict_wc[word]
            dict_wc[word]=old_count+1
        else:
            dict_wc[word] =  1


    def __len__(self):
        return len(self.claims)

    def build_word_vocabulary(self,w,word_vocab):
        # if the word is new, get the last id and add it
        w_small=w
            #commenting out .lower() test if that affects performance.lower()
        #if(w.lower() not in word_vocab ):
        if (w not in word_vocab):
            len_dict=len(word_vocab.keys())
            word_vocab[w_small]=len_dict+1
        return word_vocab


    def get_max_lengths_add_to_vocab(self,word_vocab,runName,args):
        #their vocabulary function was giving issues (including having duplicates). creating my own dictionary.
        #word_vocab = Vocabulary()

        word_count={"dummy":0}

        max_claim_len = 0
        max_evidence_len = 0
        max_num_evidences = 0

        max_claim = ""
        longest_evidence_words = ""   

        list_of_longest_ev_lengths=[]
        list_of_longest_evidences=[]
        list_of_longest_claim_lengths = []
        list_of_longest_claims = []


        for each_claim in self.claims:
            words = [w for w in each_claim.split(" ")]
            for word in words:
                #build vocabulary only from training data. In dev, a new word it sees must be returned @UNKNOWN
                if(runName=='train'):
                    word_vocab=self.build_word_vocabulary(word,word_vocab)

                #increase word frequency count
                self.update_word_count(word_count,word)

            if len(words) > max_claim_len:
                max_claim_len = len(words)
                max_claim = words
                list_of_longest_claim_lengths.append(max_claim_len)
                list_of_longest_claims.append(words)

        for each_ev in self.evidences:
            words = [w for w in each_ev.split(" ")]
            for word in words:
                if (runName == 'train'):
                    word_vocab = self.build_word_vocabulary(word, word_vocab)

                # increase word frequency count
                self.update_word_count(word_count, word)
            if len(words) > max_evidence_len:
                max_evidence_len = len(words)
                longest_evidence_words = words
                list_of_longest_ev_lengths.append(max_evidence_len)
                list_of_longest_evidences.append(longest_evidence_words)



        #for debug: find the top 10 longest sentences and their length
        s=sorted(list_of_longest_evidences,key=len,reverse=True)
        top10=s[:10]
        #LOG.debug(f"list_of_longest_evidences.sort(:{top10}")
        s_lengths=sorted(list_of_longest_ev_lengths,reverse=True)
        #LOG.debug(f"list_of_longest_ev_lengths.sort(:{s_lengths[:10]}")

        claim_sorted_len=sorted(list_of_longest_claim_lengths,reverse=True)
        x=claim_sorted_len[:10]


        print(f"just before returning word_vocab. Number of unique words is {len(word_vocab.keys())}")
        return word_vocab, max_claim_len, max_evidence_len,word_count


    def pad_item(self, dataitem,isev=False):
        if(isev):
            dataitem_padded = dataitem + [self.word_vocab[RTEDataset.PAD]] * (self.max_ev_len - len(dataitem))
        #ask becky : right now am padding with the max entity length. that is what fan also is doing .shouldn't i be padding both claim and evidence -with its own max length (eg:20 and 18719)
        # or should i pad upto  the biggest amongst both, i.e 18719 words in evidence
        else:
            dataitem_padded = dataitem + [self.word_vocab[RTEDataset.PAD]] * (self.max_claims_len - len(dataitem))

        return dataitem_padded

    def get_num_classes(self):
        return len(list({l for l in self.lbl}))

    def get_labels(self):
        return self.lbl

    # __getitem__ is a function of pytorch's Dataset class. Which this class inherits. Here he is just overriding it
    # go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and search for __getitem__
    # this is some internal memory saving thing to not load the entire dataset into memory at once.
    #askajay: so if i want to do some data processing on the raw data that i read from disk, is this the point where i do it? for each data point kind of thing?

    def __getitem__(self, idx):

        # for each word in claim (and evidence in turn) get the corresponding unique id



        label = self.lbl[idx]


        #todo: ask becky if we should do lowercase for all words in claims and evidence

        claims_words_str = [[w for w in (self.claims[idx].split(" "))]]
        ev_words_str= [[w for w in (self.evidences[idx].split(" "))]]

        # if the word existsin in word vocabulary return its id. else if it doesn't exist in word vocabulary,
        # return the id of #UNKNOWN
        claims_words_id=[]
        ev_words_id=[]
        claims_word_id=None
        ev_word_id=None
        for w in (self.claims[idx].split(" ")):
            if (w in self.word_vocab):
                claims_word_id = self.word_vocab[w]
            else:
                claims_word_id = self.word_vocab[self.UNKNOWN]
            claims_words_id.append(claims_word_id)


        for w in (self.evidences[idx].split(" ")):
            if (w in self.word_vocab):
                ev_word_id = self.word_vocab[w]
            else:
                ev_word_id = self.word_vocab[self.UNKNOWN]
            ev_words_id.append(ev_word_id)




        len_claims_words=len(claims_words_id)
        len_evidence_words = len(ev_words_id)

        claims_words_id_padded = self.pad_item(claims_words_id)
        ev_words_id_padded = self.pad_item(ev_words_id,True)


        if self.transform is not None:

            #add noise to both claim and evidence anyway. on top of it, if you want to add replacement,
            # or make it
            #mutually exclusive, do it later. Also note that this function will return two strings
            # each for one string given.
            #that is because it assumes different transformation for student and teacher.

            claim_words_dropout_str = self.transform(claims_words_str, RTEDataset.ENTITY)
            ev_words_dropout_str = self.transform(ev_words_str, RTEDataset.ENTITY)

            # 1. Replace word with synonym word in Wordnet / NIL (whichever is enabled)

            if RTEDataset.WORD_NOISE_TYPE == 'replace':
                assert len(claim_words_dropout_str) == 2, "There is some issue with TransformTwice ... " #todo: what if we do not want to use the teacher ?
                new_replaced_words = [w for ctx in claim_words_dropout_str[0] + claim_words_dropout_str[1]
                                        for w in ctx
                                        if not self.word_vocab.contains(w)]

                # 2. Add word to word vocab (expand vocab)
                new_replaced_word_ids = [self.word_vocab.add(w, count=1)
                                         for w in new_replaced_words]

                # 3. Add the replaced words to the word_vocab_embed (if using pre-trained embedding)
                if self.args.pretrained_wordemb:
                    for word_id in new_replaced_word_ids:
                        word_embed = self.sanitise_and_lookup_embedding(word_id)
                        self.word_vocab_embed = np.vstack([self.word_vocab_embed, word_embed])

                # print("Added " + str(len(new_replaced_words)) + " words to the word_vocab... New Size: " + str(self.word_vocab.size()))


            #back to drop world: now pad 4 things separately i.e claim for teacher, claim for student, evidence for teacher, evidence for student
            claim_dropout_word_ids = list()

            #for each word in the claim (note, this is after drop out), find its corresponding ids from the vocabulary dictionary
            claim_dropout_word_ids.append([[self.word_vocab[w]
                                         for w in ctx]
                                        for ctx in claim_words_dropout_str[0]])
            claim_dropout_word_ids.append([[self.word_vocab[w]
                                         for w in ctx]
                                        for ctx in claim_words_dropout_str[1]])

            if len(claim_dropout_word_ids) == 2:  # i.e if its ==2 , it means transform twice (1. student 2. teacher)
                claims_words_padded_0 = self.pad_item(claim_dropout_word_ids[0][0])
                claims_words_padded_1 = self.pad_item(claim_dropout_word_ids[1][0])
                claims_datum = (torch.LongTensor(claims_words_padded_0), torch.LongTensor(claims_words_padded_1))
            else:
                # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(claim_dropout_word_ids)
                claims_datum = torch.LongTensor(context_words_padded)

            #do the same for evidence also
            evidence_words_dropout = list()
            evidence_words_dropout.append([[self.word_vocab[w]
                                         for w in ctx]
                                        for ctx in ev_words_dropout_str[0]])
            evidence_words_dropout.append([[self.word_vocab[w]
                                         for w in ctx]
                                        for ctx in ev_words_dropout_str[1]])

            if len(evidence_words_dropout) == 2:  # transform twice (1. student 2. teacher): DONE

                #if its evidence , and not claim, pad_item requires the second argument to be True
                evidence_words_padded_0 = self.pad_item(evidence_words_dropout[0][0],True)
                evidence_words_padded_1 = self.pad_item(evidence_words_dropout[1][0],True)
                evidence_datum = (torch.LongTensor(evidence_words_padded_0), torch.LongTensor(evidence_words_padded_1))
            else:
                # todo: change this to an assert (if we are always using the student and teacher networks)
                context_words_padded = self.pad_item(evidence_words_dropout)
                evidence_datum = torch.LongTensor(context_words_padded)

        #if we are not doing any transformation or adding any noise, just pad plain claim adn evidence
        else:
            claims_datum = torch.LongTensor(claims_words_id_padded)
            evidence_datum = torch.LongTensor(ev_words_id_padded)



        # transform means, if you want a different noise for student and teacher
        # so if you are transforming (i.e adding noise to both claim and evidence, for both student and teacher
        #  , you will be returning two different types of claim and evidence. else just one.

        if self.transform is not None:
            return (claims_datum[0], evidence_datum[0]), (claims_datum[1], evidence_datum[1]), label, (len_claims_words,len_evidence_words)
        else:
            return (claims_datum, evidence_datum), label,(len_claims_words,len_evidence_words)



