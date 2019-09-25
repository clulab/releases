from tqdm import tqdm
import json,mmap,os,argparse,string,sys
import processors
from processors import *
from cleantext import clean
from os import listdir
from os.path import isfile,join
import traceback
import logging

logging.basicConfig(filename='merging_sstag_smartnertag.log',filemode='w+')
LOG = logging.getLogger('main')

def get_new_name( prev, unique_new_ners, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                 unique_new_tokens, dict_newner_token):
    separator = ""
    #curr_ner = prev[0]
    new_nertag_i = ""
    full_name_c = " ".join(full_name)

    if (full_name_c in unique_new_tokens.keys()):

        new_nertag_i = unique_new_tokens[full_name_c]

    else:

        if (curr_ner in unique_new_ners.keys()):
            old_index = unique_new_ners[curr_ner]
            new_index = old_index + 1
            unique_new_ners[curr_ner] = new_index
            # to try PERSON SPACE C1 instead of PERSON-C1
            new_nertag_i = curr_ner + separator + ev_claim + str(new_index)
            # new_nertag_i = curr_ner + separator + ev_claim + str(new_index)
            unique_new_tokens[full_name_c] = new_nertag_i

        else:
            unique_new_ners[curr_ner] = 1
            new_nertag_i = curr_ner + separator + ev_claim + "1"
            unique_new_tokens[full_name_c] = new_nertag_i

    if not ((full_name_c, prev[0]) in dict_tokenner_newner):
        dict_tokenner_newner[full_name_c, prev[0]] = new_nertag_i
    else:
        dict_tokenner_newner[full_name_c, prev[0]] = new_nertag_i

    dict_newner_token[new_nertag_i] = full_name_c

    new_sent.append(new_nertag_i)

    full_name = []
    prev = []
    if (curr_ner != "O"):
        prev.append(curr_ner)

    return prev, dict_tokenner_newner, new_sent, full_name, unique_new_ners, unique_new_tokens, dict_newner_token

def attach_freq_to_nertag(ner_tag, ner_dictionary,separator,ev_claim):
    new_index = get_frequency_of_tag(ner_tag, ner_dictionary)
    new_nertag_i = ner_tag + separator + ev_claim + str(new_index)
    return new_nertag_i

def collapse_continuous_names(claims_words_list, claims_ner_list, ev_claim):

    #dict_newNerBasedName_lemma:a mapping from newNerBasedName to its old lemma value(called henceforth as token) Eg:{PERSONc1:Michael Schumacher}.
    dict_newNerBasedName_lemma = {}
    # dict_token_ner_newner:a mapping from a tuple (lemma, original NER tag of the word) to its newNerBasedName  Eg:{(Michael Schumacher,PERSON):PERSONc1}
    dict_token_ner_newner = {}
    #dict_lemmas_newNerBasedName. A mapping from LEMMA/token of the word to its newNerBasedName Eg:{Michael Schumacher:PERSONc1}
    dict_lemmas_newNerBasedName = {}
    #dict_newNerBasedName_freq: A mapping from newNerBasedName to the number of times it occurs in a given sentences
    dict_newNerBasedName_freq = {}
    #a stack to hold all the ner tags before current- this is useful for checking if a name is spread across multiple NER tags. Eg: JRR Tolkein= PERSON,PERSON,PERSON
    prev = []
    #the final result of combining all tags and lemmas is stored here
    new_sent = []
    #this full_name is used to store multiple parts of same name.Eg:Michael Schumacher
    full_name = []


    #in this code, triggers happen only when a continuous bunch of nER tags end. Eg: PERSON , PERSON, O.
    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            if (len(prev) == 0):
                # if there were no tags just before this O, it means, its the end of a combined name. just add that O and move on
                new_sent.append(curr_word)
            else:
                # instead if there was something pushed into a stack, this O means, we are just done with those couple of continuous tags. Collapse names, add new name to dictionaries and empty stack
                prev, dict_token_ner_newner, new_sent, full_name, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma \
                    = get_new_name(prev, dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word,
                                                               new_sent, ev_claim, full_name, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)
                new_sent.append(curr_word)
        else:
            #if length of the list called previous is zero, it means, no tag was collapsed until now.
            if (len(prev) == 0):
                prev.append(curr_ner)
                full_name.append(curr_word)
            else:
                #if the previous ner tag and current ner tag is the same, it means its most probably part of same name. Eg: JRR Tolkein. Collapse it into one nER entity
                if (prev[(len(prev) - 1)] == curr_ner):
                    prev.append(curr_ner)
                    full_name.append(curr_word)
                else:


                    # if the previous ner tag and current ner tag are not the same, this O means, we are just done with those couple of continuous tags. No collapsing, but add both names to dictionaries and empty stack
                    prev, dict_token_ner_newner, new_sent, full_name, dict_newNerBasedName_freq, \
                    dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma = \
                        append_count_to_two_consecutive_ner_tags(prev, dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word, new_sent,
                                                                 ev_claim, full_name, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)

    return new_sent, dict_token_ner_newner, dict_newNerBasedName_lemma


def collapse_continuous_names_with_dashes(words_list, ner_list, ev_claim):

    #dict_newNerBasedName_lemma:a mapping from newNerBasedName to its old lemma value(called henceforth as token) Eg:{PERSONc1:Michael Schumacher}.
    dict_newNerBasedName_lemma = {}
    # dict_token_ner_newner:a mapping from a tuple (lemma, original NER tag of the word) to its newNerBasedName  Eg:{(Michael Schumacher,PERSON):PERSONc1}
    dict_token_ner_newner = {}
    #dict_lemmas_newNerBasedName. A mapping from LEMMA/token of the word to its newNerBasedName Eg:{Michael Schumacher:PERSONc1}
    dict_lemmas_newNerBasedName = {}
    #dict_newNerBasedName_freq: A mapping from newNerBasedName to the number of times it occurs in a given sentences
    dict_newNerBasedName_freq = {}
    #a stack to hold all the ner tags before current- this is useful for checking if a name is spread across multiple NER tags. Eg: JRR Tolkein= PERSON,PERSON,PERSON
    prev_ner = ""
    #the final result of combining all tags and lemmas is stored here
    new_sent = []
    #this full_name is used to store multiple parts of same name.Eg:Michael Schumacher
    full_name = []

    list_of_indices_to_collapse=[]
    total_string_length=len(ner_list)
    #in this code, triggers happen only when a continuous bunch of nER tags end. Eg: PERSON , PERSON, O.
    for index, (curr_ner, curr_word) in enumerate(zip(ner_list, words_list)):
        # skip as many indices as we have already collapsed. This is because say if NEW YORK POST is collapsed into one entity, we don't want it to add post again to any of dictionaries or
        # new sentence words
        if index in (list_of_indices_to_collapse):
            continue;
        if (curr_ner == "O"):
                # if there were no tags just before this O, it means, its the end of a combined name. just add that O and move on
                new_sent.append(curr_word)
                prev_ner=curr_ner
        else:
            if (curr_ner=="_" and prev_ner not in ["O","_"]):
                #if the current NER tag is _ and the previous NER tag was something other than a  _ or O it means, this _
                #  was just after a proper NER tag, like FOOD, NUMBER etc. its time to collapse.
                list_of_indices_to_collapse = find_how_many_indices_to_collapse(index, ner_list)
                assert (len(claim_ann.tags) is not 0)
                new_lemma_name=join_indices_to_new_name(words_list, list_of_indices_to_collapse)
                str_new_lemma_name=" ".join(new_lemma_name)

                #add it to all dictionaries where the curr_ner=prev_ner Eg: Michael Schumacher:Person and curr_word=str_new_lemma_name
                dict_token_ner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma \
                    = append_count_to_ner_tags(dict_newNerBasedName_freq, prev_ner, dict_token_ner_newner, str_new_lemma_name,
                                               new_sent, ev_claim,
                                               dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)
                prev_ner = curr_ner


            else:
                #if the curr_ner is _ and the previous one was O or _, it is an anomaly
                # , it really doesn't make sense/doesn't need collapsing. Just add the word as is
                if (curr_ner == "_" and prev_ner in ["O","_"]):
                    new_sent.append(curr_word)
                    prev_ner = curr_ner
                else:
                    #if you reach here, it means, the current_ner is none of O,_ with O before it, or _ with a proper tag before it. So this must be a proper tag, like FOOD, NUMBER etc
                    prev_ner=curr_ner
                    #look ahead, if the next NER value is a dash, don't add to dictionary. else add.
                    if((index+1) < total_string_length):
                            if not (ner_list[index + 1] == "_"):
                                dict_token_ner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma= append_count_to_ner_tags(dict_newNerBasedName_freq, curr_ner, dict_token_ner_newner, curr_word,
                                                     new_sent, ev_claim,
                                                     dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma)

    return new_sent, dict_token_ner_newner, dict_newNerBasedName_lemma


def append_count_to_ner_tags( dict_newNerBasedName_freq, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim,
                                             dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma):
    # if the ner value is stative, don't find new ner based name. eg:stativec1. just add it to all dictionaries
    if(curr_ner=="stative"):
        new_nertag_i=curr_ner
    else:
        new_nertag_i = attach_freq_to_nertag(curr_ner, dict_newNerBasedName_freq, "", ev_claim)
    lemma_ner_tuple = (curr_word, curr_ner)
    new_sent.append(new_nertag_i)
    add_to_dict_if_not_exists(curr_word, new_nertag_i, dict_lemmas_newNerBasedName)
    add_to_dict_if_not_exists(new_nertag_i,curr_word, dict_newNerBasedName_lemma)
    add_to_dict_if_not_exists(lemma_ner_tuple, new_nertag_i, dict_tokenner_newner)

    return dict_tokenner_newner, new_sent, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma


def append_count_to_two_consecutive_ner_tags(prev, dict_newNerBasedName_freq, curr_ner, dict_tokenner_newner, curr_word, new_sent, ev_claim, full_name,
                                             dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma):

    #do same thing twice for both current and previous tags/words
    prev_tag=prev[(len(prev)-1)]
    prev_word="".join(full_name)
    new_nertag_i=attach_freq_to_nertag(prev_tag,dict_newNerBasedName_freq , "", ev_claim)
    add_to_dict_if_not_exists(prev_word, new_nertag_i, dict_lemmas_newNerBasedName)
    add_to_dict_if_not_exists(new_nertag_i,prev_word, dict_newNerBasedName_lemma)
    new_sent.append(new_nertag_i)

    new_nertag_i=attach_freq_to_nertag(curr_ner, dict_newNerBasedName_freq, "", ev_claim)
    add_to_dict_if_not_exists(curr_word, new_nertag_i, dict_lemmas_newNerBasedName)
    new_sent.append(new_nertag_i)

    full_name = []
    prev = []
    if (curr_ner != "O"):
        prev.append(curr_ner)

    return prev, dict_tokenner_newner, new_sent, full_name, dict_newNerBasedName_freq, dict_lemmas_newNerBasedName, dict_newNerBasedName_lemma

def add_to_dict_if_not_exists(key, value, dict):
    if not (key in dict.keys()):
        dict[key]=value

def replace_value(key, value, dict):
        dict[key]=value


def get_frequency_of_tag(curr_ner,dict_newner_token):
    freq=1
    if curr_ner in dict_newner_token.keys():
        old_count= dict_newner_token[curr_ner]
        freq=old_count+1
        dict_newner_token[curr_ner]=freq
    else:
        dict_newner_token[curr_ner] = 1
    return freq

#if there is one NER followed by more than one dashes, collect them all together so that it can be assigned to one name/new_ner_tag etc
def find_how_many_indices_to_collapse(curr_index, list_ner_tags):
    #very first time add the NER tag before _ Eg: Formula in Formula one
    list_indices_to_collapse=[]
    list_indices_to_collapse.append(curr_index-1)
    #then keep adding indices unti you hit a word that is not _

    while (curr_index<len(list_ner_tags)):
        if  (list_ner_tags[curr_index] == "_"):
            list_indices_to_collapse.append(curr_index)
            curr_index = curr_index + 1
        else:
            return list_indices_to_collapse
    return list_indices_to_collapse

def join_indices_to_new_name(all_words,list_indices):
    new_name=[]
    for i in list_indices:
        new_name.append(all_words[i])
    return new_name

def append_tags_with_count(claims_words_list, claims_ner_list, ev_claim):
    dict_ner_freq = {}
    new_sent = []

    for index, (curr_ner, curr_word) in enumerate(zip(claims_ner_list, claims_words_list)):
        if (curr_ner == "O"):
            new_sent.append(curr_word)
        else:
                freq = get_frequency_of_tag( curr_ner,dict_ner_freq)
                new_sent.append(curr_ner+ev_claim+str(freq))
    return new_sent



def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

def read_rte_data(filename,args):
        tr_len=1000
        all_labels = []
        all_claims = []
        all_evidences = []

        with open(filename) as f:
            for index,line in enumerate(tqdm(f, total=get_num_lines(filename))):
                multiple_ev = False
                x = json.loads(line)
                claim = x["claim"]
                evidences = x["evidence"]
                label = x["label"]

                if (args.remove_punctuations == True):
                    claim = claim.translate(str.maketrans('', '', string.punctuation))
                    evidences = evidences.translate(str.maketrans('', '', string.punctuation))

                all_claims.append(claim)
                all_evidences.append(evidences)
                all_labels.append(label)

        return all_claims, all_evidences, all_labels


def write_json_to_disk(claim, evidence,label,outfile):
    total = {'claim': claim,
             'evidence':evidence,
             "label":label}
    json.dump(total, outfile)
    outfile.write('\n')

def annotate(headline, body, API):
    claim_ann = API.fastnlp.annotate(headline)
    ev_ann = API.fastnlp.annotate(body)
    return claim_ann, ev_ann


def check_exists_in_claim(new_ev_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev, dict_tokenner_newner_claims):


        combined_sent=[]


        found_intersection = False



        #for every token (irrespective of NER or not) in evidence_from_lexicalized_data
        for ev_new_ner_value in new_ev_sent_after_collapse:

            found_intersection=False

            #check if its an ner
            if ev_new_ner_value in dict_newner_token_ev.keys():

                #if thats true find its corresponding string/lemma value Eg: "tolkein" from dict_newner_token_ev which maps PERSON-E1 ->tolkein
                token=dict_newner_token_ev[ev_new_ner_value]
                token_split=set(token.split(" "))


                #now go to through the keys in the dictionary that maps tokens in claim to its new ner Eg: tolkein:PERSON
                for tup in dict_tokenner_newner_claims.keys():
                    name_cl = tup[0]
                    ner_cl=tup[1]
                    name_cl_split = set(name_cl.split(" "))


                    #check if any of the names/tokens in claim have an intersection with what you just got from evidence_from_lexicalized_data ev_new_ner_value. Eg: tolkein
                    if (token_split.issubset(name_cl_split) or name_cl_split.issubset(token_split)):
                        found_intersection = True


                        # also confirm that NER value of the thing you found just now in evidence_from_lexicalized_data also matches the corresponding NER value in claim. This is to avoid john amsterdam PER overlapping with AMSTERDAM LOC
                        actual_ner_tag=""
                        for k, v in dict_tokenner_newner_evidence.items():

                            if (ev_new_ner_value == v):
                                actual_ner_tag=k[1]

                                break

                        #now check if this NER tag in evidence_from_lexicalized_data also matches with that in claims
                        if(actual_ner_tag==ner_cl):
                            val_claim = dict_tokenner_newner_claims[tup]
                            combined_sent.append(val_claim)
                        #if it doesn't, just consider it as a regular token and add it as is
                        else:
                            combined_sent.append(token)


                        #now that you found that there is an overlap between your evidence_from_lexicalized_data token and the claim token, no need to go through the claims dictionary which maps tokens to ner
                        break;


                if not (found_intersection):
                    combined_sent.append(ev_new_ner_value)
                    new_ner=""


                    #get the evidence_from_lexicalized_data's PER-E1 like value
                    for k,v in dict_tokenner_newner_evidence.items():
                        #print(k,v)
                        if(ev_new_ner_value==v):
                            new_ner=k[1]

                    dict_tokenner_newner_claims[token, new_ner] = ev_new_ner_value



            else:
                combined_sent.append(ev_new_ner_value)



        return combined_sent,found_intersection

def parse_commandline_args():
    return create_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_parser():
    parser = argparse.ArgumentParser(description='Pg')
    parser.add_argument('--inputFile', type=str, default='fever_train_split_fourlabels.jsonl',
                        help='name of the input file to convert to smart ner format')
    parser.add_argument('--pyproc_port', type=int, default=8888,
                        help='port at which pyprocessors server should run. If you are running'
                             'multiple servers on the same machine, will need different port for each')
    parser.add_argument('--use_docker', default=False, type=str2bool,
                        help='use docker for loading pyproc. useful in machines where you have root access.', metavar='BOOL')
    parser.add_argument('--convert_prepositions', default=False, type=str2bool,
                        help='.',
                        metavar='BOOL')
    parser.add_argument('--create_smart_NERs', default=False, type=str2bool,
                        help='mutually ',
                        metavar='BOOL')
    parser.add_argument('--merge_ner_ss', default=False, type=str2bool,
                        help='once you have output from sstagger, merge them both.',
                        metavar='BOOL')
    parser.add_argument('--run_on_dummy_data', default=False, type=str2bool,
                        help='to test merging on one or two output files. once you have output from sstagger, turn this to false.',
                        metavar='BOOL')
    parser.add_argument('--remove_punctuations', default=True, type=str2bool,
                        help='once you have output from sstagger, merge them both.',
                        metavar='BOOL')
    parser.add_argument('--outputFolder', type=str, default='outputs',
                        help='name of the folder to write output to')
    parser.add_argument('--smart_ner_sstags_output_file_name', type=str, default='smartner_sstags_merged.jsonl',
                        help='name of the folder to write output to')
    parser.add_argument('--input_folder_for_smartnersstagging_merging', type=str, default='sstagged_files',
                        help='name of the folder where sstagged files will be read from')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='name of the folder where sstagged files will be read from')
    print(parser.parse_args())
    return parser




def collapseAndCreateSmartTagsSSNer(claim_words, claim_ner_tags, evidence_words, evidence_ner_tags):
    ev_claim = "c"
    neutered_claim, dict_tokenner_newner_claims, dict_newner_token = collapse_continuous_names_with_dashes(claim_words,
                                                                                               claim_ner_tags,
                                                                                               ev_claim)
    ev_claim = "e"
    ev_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_continuous_names_with_dashes(
        evidence_words, evidence_ner_tags, ev_claim)

    neutered_evidence, found_intersection = check_exists_in_claim(ev_after_collapse,
                                                                  dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                  dict_tokenner_newner_claims)

    claimn = " ".join(neutered_claim)
    evidencen = " ".join(neutered_evidence)

    return claimn, evidencen



def collapseAndReplaceWithNerSmartly(claim_words,claim_ner_tags, evidence_words, evidence_ner_tags):
        ev_claim="c"
        neutered_claim, dict_token_ner_newner_claims, dict_newNerBasedName_lemma = collapse_continuous_names(claim_words,
                                                                                                      claim_ner_tags,
                                                                                                      ev_claim)

        ev_claim = "e"
        new_sent_after_collapse, dict_tokenner_newner_evidence, dict_newner_token_ev = collapse_continuous_names(evidence_words, evidence_ner_tags, ev_claim)

        neutered_evidence, found_intersection = check_exists_in_claim(new_sent_after_collapse,
                                                                       dict_tokenner_newner_evidence, dict_newner_token_ev,
                                                                      dict_token_ner_newner_claims)


        claimn = " ".join(neutered_claim)
        evidencen = " ".join(neutered_evidence)

        return claimn,evidencen

#whenever you see a preposition in this sentence, replace the NER tags of this sentence with PREP. This is
#being done so that when we do neutering, the PREP also gets added in along with the NER tags. Just another
# experiment to check if prepositions have an effect on linguistic domain transfer
def replacePrepositionsWithPOSTags(claim_pos_tags, ev_pos_tags,claim_ner_tags,ev_ner_tags):
    for index,pos in enumerate(claim_pos_tags):
        if (pos=="IN"):
            claim_ner_tags[index]="PREP"
    for index,pos in enumerate(ev_pos_tags):
        if (pos=="IN"):
            ev_ner_tags[index]="PREP"



    return claim_ner_tags, ev_ner_tags



#for every word, if a NER tag exists, give that priority. if not, check if it has a SS tag, if yes, pick that.
# if a sstag exists and the word has no NER tag, pick SStag
def mergeSSandNERTags(ss_tags, ner_tags ):
    # give priority to NER tags when there is a collision,. Except when NER tag is MISC. In that case pick SSTag
    for index,sst in enumerate(ss_tags):
        if not (sst==""):
            #if the sstag is _, , we need it as is for the collapsing process
            #however check if the tag before this wasn't empty- there were cases of "_,_
            if(sst=="_" and not ner_tags[index-1]=="O"):
                ner_tags[index] = sst
            else:
                # if the ss TAG IS NOT empty  #get the corresponding ner tag
                nert=ner_tags[index]
                if not (nert=="O"):
                    # if the NER tag is not O,  there is a collision between NER and SSTag. Check if the NER tag is MISC
                    if(nert=="MISC"):
                        #if its MISC, pick the corresponding SSTag #if not, pick the NER tag itself -i.e dont, do anything.
                        ner_tags[index]=sst
                else:
                    #if the NER tag is 0 and SSTag exists, replace NER tag with SSTag
                    ner_tags[index] = sst
    return ner_tags

def remove_rrb_lsb_etc(sent):
    no_lrb_data=[]
    words=sent.split(" ")
    for word in words:
        if not (word.lower() in ["lrb", "rrb", "lcb", "rcb", "lsb", "rsb"]):
            no_lrb_data.append(word)
    return " ".join(no_lrb_data)


def read_sstagged_data(filename,args):
    sstags = []
    words=[]
    line_counter=0
    with open(filename,"r") as f:
            line=f.readline()
            while(line):
                line_counter=line_counter+1
                split_line=line.split("\t")
                word = split_line[1]
                if (args.remove_punctuations == True):
                    word=remove_punctuations(word)
                    #remove punctuations twice, . this is done because words like "-lrb-" was first stripped, but the stripper was not removing lsb itself.
                    word = remove_punctuations(word)
                #if the word is empty now, it means it was a punctuation, and hence removed. Don't add the word or  its tag
                if not(word == ""):
                    if not (word.lower() in ["lrb","rrb","lcb","rcb","lsb","rsb"]):
                            # if the 6th column has a dash, add it. A dash in sstagger means, this word, with the word just before it was collapsed into one entity. i.e it was I(inside) in BIO notation.
                            sstag6 = split_line[6]
                            sstag7 = split_line[7]
                            if (sstag6 == "_"):
                                sstag = sstag6
                            else:
                                sstag = sstag7
                            sstags.append(sstag)
                            words.append(word)
                            line = f.readline()
                    else:
                        line = f.readline()
                else:
                    line = f.readline()
    return sstags,words

def remove_punctuations(word):
    return clean(word,
          fix_unicode=True,  # fix various unicode errors
          to_ascii=True,  # transliterate to closest ASCII representation
          lower=False,  # lowercase text
          no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
          no_urls=False,  # replace all URLs with a special token
          no_emails=False,  # replace all email addresses with a special token
          no_phone_numbers=False,  # replace all phone numbers with a special token
          no_numbers=False,  # replace all numbers with a special token
          no_digits=False,  # replace all digits with a special token
          no_currency_symbols=False,  # replace all currency symbols with a special token
          no_punct=True,  # fully remove punctuation
          replace_with_url="<URL>",
          replace_with_email="<EMAIL>",
          replace_with_phone_number="<PHONE>",
          replace_with_number="<NUMBER>",
          replace_with_digit="0",
          replace_with_currency_symbol="<CUR>",
          lang="en"  # set to 'de' for German special handling
          )


if __name__ == '__main__':

    args = parse_commandline_args()
    if (args.log_level=="INFO"):
        LOG.setLevel(logging.INFO)
    else:
        if (args.log_level=="DEBUG"):
            LOG.setLevel(logging.DEBUG)
        else:
            if (args.log_level=="ERROR"):
                LOG.setLevel(logging.ERROR)

    if(args.use_docker==True):
        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
    else:
        API = ProcessorsAPI(port=args.pyproc_port)

    filename=args.inputFile
    #if not (args.run_on_dummy_data):
    all_claims, all_evidences, all_labels=read_rte_data(filename,args)
    all_claims_neutered=[]
    all_evidences_neutered = []

    merge_sstag_nertag_output_file = os.path.join(args.outputFolder, args.smart_ner_sstags_output_file_name)
    with open(merge_sstag_nertag_output_file, 'w') as outfile:
        outfile.write('')
    # go through all the files that start with the word claim., split its name, find its unique id, create the name of the evidence_from_lexicalized_data file with this id, and open it.
    ss_claim_file_full_path= ""
    ssfilename_ev=""
    files_skipped=0
    files_read=0
    assert (os.path.isdir(args.input_folder_for_smartnersstagging_merging)is True)
    for index,file in enumerate(listdir(args.input_folder_for_smartnersstagging_merging)):
                LOG.info(f" index: {index}")
                try:
                    file_full_path=join(args.input_folder_for_smartnersstagging_merging,file)
                    if isfile(file_full_path):
                        if file.startswith("claim"):
                            files_read=files_read+1
                            split_file_name=file.split("_")
                            datapoint_id_pred_tags=split_file_name[4]
                            dataPointId_split=datapoint_id_pred_tags.split(".")
                            dataPointId=dataPointId_split[0]
                            ss_claim_file_full_path=file_full_path
                            ssfilename_ev="evidence_words_pos_datapointid_"+str(datapoint_id_pred_tags)
                            ss_evidence_file_full_path=join(args.input_folder_for_smartnersstagging_merging, ssfilename_ev)
                            if not ss_claim_file_full_path:
                                LOG.error("ss_claim_file_full_path is empty.skipping this datapoint")
                                files_skipped = files_skipped + 1
                                continue;
                            if not ssfilename_ev:
                                LOG.error("ssfilename_ev is empty. skipping this datapoint")
                                files_skipped = files_skipped + 1
                                continue;
                            if not isfile(ss_evidence_file_full_path):
                                LOG.error(f"ss_evidence_file_full_path is not found:{ss_evidence_file_full_path}")
                                files_skipped = files_skipped + 1
                                continue;

                            LOG.info(f"*************************")
                            LOG.info(f"value of ss_claim_file_full_path is:{ss_claim_file_full_path}")
                            LOG.info(f"value of ssfilename_ev is:{ssfilename_ev}")


                            if (args.merge_ner_ss):
                                claims_sstags, sstagged_claim_words = read_sstagged_data(ss_claim_file_full_path, args)
                                assert (len(claims_sstags) is len(sstagged_claim_words))
                                LOG.debug(f"done reading read_sstagged_data for ss_claim_file_full_path")
                                ev_sstags, sstagged_ev_words = read_sstagged_data(ss_evidence_file_full_path,args)
                                LOG.debug(f"done reading read_sstagged_data for ss_evidence_file_full_path")
                                LOG.debug(f"value of evidence_from_lexicalized_data from sstagged data:{sstagged_ev_words}")
                                LOG.debug(f"value of ev_sstags:{ev_sstags}")

                                if not (len(ev_sstags)== len(sstagged_ev_words)):
                                    LOG.debug(f"value of len(ev_sstags):{len(ev_sstags)}")
                                    LOG.debug(f"value of len(sstagged_ev_words) :{len(sstagged_ev_words)}")
                                    LOG.error("value of len(ev_sstags) and len(sstagged_ev_words) don't match ")


                                if not (dataPointId):
                                    LOG.error("dataPointId is empty")

                                dataPointId_int=int(dataPointId)
                                claim_before_removing_punctuations = all_claims[dataPointId_int]
                                evidence_from_lexicalized_data = all_evidences[dataPointId_int]
                                l = all_labels[dataPointId_int]
                                LOG.debug(f"value of evidence_from_lexicalized_data from lexicalized data:{evidence_from_lexicalized_data}")
                                if(args.remove_punctuations==True):
                                    evidence_from_lexicalized_data=remove_punctuations(evidence_from_lexicalized_data)
                                    evidence_from_lexicalized_data = remove_rrb_lsb_etc(evidence_from_lexicalized_data)


                                l_ev_lexicalized=len(evidence_from_lexicalized_data.split(" "))
                                LOG.debug(f"value of length of evidence_from_lexicalized_data from lexicalized data:{l_ev_lexicalized }")
                                LOG.debug(f"value of length of evidence_from_lexicalized_data from sstagged data:{len(sstagged_ev_words) }")


                                claim=claim_before_removing_punctuations
                                #remove punctuations and unicode from claims also and make sure its same size as
                                if (args.remove_punctuations == True):
                                    claim=remove_punctuations(claim_before_removing_punctuations)
                                    claim = remove_rrb_lsb_etc(claim)

                                LOG.debug(f"value of claim from lexicalized data:{claim}")
                                LOG.debug(f"value of claim from sstagged data:{sstagged_claim_words}")

                                claim_ann, ev_ann = annotate(claim, evidence_from_lexicalized_data, API)
                                assert (claim_ann is not None)
                                assert (ev_ann is not None)

                                claim_ner_tags = claim_ann._entities
                                ev_ner_tags= ev_ann._entities

                                lcet = len(claims_sstags)
                                lesst = len(claim_ner_tags)
                                LOG.debug(f"value of len(claims_sstags) is :{lcet}")
                                LOG.debug(f"value claim_ner_tags is :{lesst}")
                                if not (lcet == lesst):
                                    LOG.error(
                                        "value of len(claims_sstags) and len(claim_ner_tags) don't match ")
                                    files_skipped = files_skipped + 1
                                    LOG.error(f"total files skipped so far is {files_skipped}")
                                    for x,y,z in zip(claims_sstags, claim_ner_tags,sstagged_claim_words):
                                        LOG.error(f"{x},{y},{z}")
                                    continue


                                lcet = len(ev_sstags)
                                lesst = len(ev_ner_tags)
                                LOG.debug(f"value of len(ev_sstags) is :{lcet}")
                                LOG.debug(f"value ev_ner_tags is :{lesst}")
                                if not (lcet == lesst):
                                    LOG.error(
                                        "value of len(ev_sstags) and len(ev_ner_tags) don't match ")
                                    files_skipped = files_skipped + 1
                                    LOG.error(f"total files skipped so far is {files_skipped}")
                                    for x,y,z in zip(ev_sstags, ev_ner_tags,sstagged_ev_words):
                                        LOG.error(f"{x},{y},{z}")
                                    continue



                                claim_ner_ss_tags_merged = mergeSSandNERTags(claims_sstags, claim_ner_tags)
                                ev_ner_ss_tags_merged = mergeSSandNERTags(ev_sstags, ev_ner_tags)


                                claim_pos_tags = claim_ann.tags
                                ev_pos_tags = ev_ann.tags

                                # LOG.debug(f"value of claim_pos_tags is:{claim_pos_tags}")
                                # LOG.debug(f"value of ev_pos_tags is:{ev_pos_tags}")
                                LOG.debug(f"value of claim_ner_tags is:{claim_ner_tags}")
                                LOG.debug(f"value of ev_ner_tags is:{ev_ner_tags}")
                                LOG.debug(f"value of claims_sstags is:{claims_sstags}")
                                LOG.debug(f"value of ev_sstags is:{ev_sstags}")
                                LOG.debug(f"value of claim_ner_ss_tags_merged is:{claim_ner_ss_tags_merged}")
                                LOG.debug(f"value of ev_ner_ss_tags_merged is:{ev_ner_ss_tags_merged}")

                                LOG.error(f"total files skipped so far is {files_skipped}")
                                if(args.convert_prepositions==True):
                                    claim_ner_ss_tags_merged, ev_ner_ss_tags_merged=replacePrepositionsWithPOSTags(claim_pos_tags, ev_pos_tags, claim_ner_ss_tags_merged, ev_ner_ss_tags_merged)

                                if (args.create_smart_NERs == True):

                                    claim_neutered, ev_neutered =collapseAndReplaceWithNerSmartly(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)



                                if (args.merge_ner_ss == True):
                                    lcet = len(claim_ann.words)
                                    lesst = len(claim_ner_ss_tags_merged)
                                    LOG.debug(f"value of len(claim_ann.words) is :{lcet}")
                                    LOG.debug(f"value len(claim_ner_ss_tags_merged) :{lesst}")
                                    if not (lcet == lesst):
                                        LOG.error(
                                            "value of len(claim_ann.words) and value len(claim_ner_ss_tags_merged) don't match ")


                                        files_skipped = files_skipped + 1
                                        LOG.error(f"total files skipped so far is {files_skipped}")
                                        for x, y in zip(claim_ann.words, claim_ner_ss_tags_merged):
                                            LOG.error(f"{x},{y}")
                                        continue

                                    lcet = len(ev_ann.words)
                                    lesst = len(ev_ner_ss_tags_merged)
                                    LOG.debug(f"value of len(ev_ann.words) is :{lcet}")
                                    LOG.debug(f"value len(ev_ner_ss_tags_merged) is :{lesst}")
                                    if not (lcet == lesst):
                                        LOG.error(
                                            "value of len(ev_sstags) and len(ev_ner_tags) don't match ")

                                        files_skipped = files_skipped + 1
                                        LOG.error(f"total files skipped so far is {files_skipped}")
                                        for x,y in zip(ev_ann.words, ev_ner_ss_tags_merged):
                                            LOG.error(f"{x},{y}")
                                        continue



                                    claim_neutered, ev_neutered =collapseAndCreateSmartTagsSSNer(claim_ann.words, claim_ner_ss_tags_merged, ev_ann.words, ev_ner_ss_tags_merged)


                                with open(merge_sstag_nertag_output_file, 'a+') as outfile:
                                    write_json_to_disk(claim_neutered, ev_neutered,l.upper(),outfile)



                except IOError:
                    LOG.error('An error occured trying to read the file.')
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    traceback.print_exc()
                    continue


                except ValueError:
                    LOG.error('Non-numeric data found in the file.')
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    traceback.print_exc()
                    continue


                except ImportError:
                    LOG.error("NO module found")
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    traceback.print_exc()
                    continue

                except EOFError:
                    LOG.error('Why did you do an EOF on me?')
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    traceback.print_exc()
                    continue


                except KeyboardInterrupt:
                    LOG.error('You cancelled the operation.')
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    traceback.print_exc()
                    continue


                except:
                    LOG.error('An error which wasnt explicity caught occured.')
                    LOG.error(f"value of current datapoint is {dataPointId}")
                    LOG.error(f"value of index  is {index}")
                    traceback.print_exc()
                    continue


