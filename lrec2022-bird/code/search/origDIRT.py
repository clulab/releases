'''
Reads several tsv files which are the results of path extraction 
phase. Then for a given path, finds the top 40 most similar paths 
according to the original DIRT method. More specifically:
    
    1- finds the candidate paths (all the paths that share at 
       least one feature with the input path)
    
    2- filters the candidate paths if the number of their common 
       features with the input path is less than a fixed percent 
       of the total number of features for the input path and the 
       candidate path.
       
    3- calculates a similary score between the input path and the 
       remaining candidate paths according to equations (1), (2), 
       and (3) on page 4 of DIRT paper.
       
    4- outputs the top 40 candidate paths in descending order of their 
       similarity to the input path.
       
Usage: 
python3 origDIRT.py {input query path}
'''

from ast import literal_eval
import math
import sys

paths_features = {}
paths_slotfreq = {}
xfeatures_totalfreqs = {}
yfeatures_totalfreqs = {}
xfeatures_totalfreqs_sum = 0
yfeatures_totalfreqs_sum = 0
sentenceids_to_sentences = {}
paths_sentences = {}

if len(sys.argv) != 2:
    print("Usage: python3 origDIRT.py {input query path}")
    exit()

def main():
    global paths_features
    global paths_slotfreq
    global xfeatures_totalfreqs
    global yfeatures_totalfreqs
    global xfeatures_totalfreqs_sum
    global yfeatures_totalfreqs_sum
    global sentenceids_to_sentences
    global paths_sentences

    # read 'pathids_to_paths.tsv' and load it into pathids_to_paths
    # dictionary. This dictionary will contain a mapping from path ids
    # to paths.
    files_dir = "../data/"
    pathids_to_paths_file = files_dir + "pathids_to_paths.tsv"
    f = open(pathids_to_paths_file, mode="r", encoding="utf_8")
    pathids_to_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        path = literal_eval(fields[1])
        pathids_to_paths[path_id] = get_path_textual_string(path)
    f.close()

    # read 'paths_features.tsv' and load it into paths_features
    # dictionary. This dictionary will contain a mapping from 
    # path ids to a tuple (slotX and slotY). slotX is a dictionary
    # that contains the slot X features of a paths (slot-filler
    # words and their frequencies). Similarly, slotY contains
    # the slot Y features of a path.
    paths_features_file = files_dir + "paths_features.tsv"
    f = open(paths_features_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        slotX = literal_eval(fields[1])
        slotY = literal_eval(fields[2])
        paths_features[path_id] = (slotX,slotY)
    f.close()

    # read 'xfeatures_paths.tsv' and load it into xfeatures_paths
    # dictionary. This dictionary will contain a mapping from a word
    # to the list of the path ids of all the paths that the word is
    # a slot X slot-filler. This dictionary is required for speeding up
    # calculation of similary scores.
    xfeatures_paths_file = files_dir + "xfeatures_paths.tsv"
    f = open(xfeatures_paths_file, mode="r", encoding="utf_8")
    xfeatures_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        xfeatures_paths[word] = path_ids
    f.close()
    
    # read 'yfeatures_paths.tsv' and load it into yfeatures_paths
    # dictionary. This dictionary will contain a mapping from a word
    # to the list of the path ids of all the paths that the word is
    # a slot Y slot-filler. This dictionary is required for speeding up
    # calculation of similary scores.
    yfeatures_paths_file = files_dir + "yfeatures_paths.tsv"
    f = open(yfeatures_paths_file, mode="r", encoding="utf_8")
    yfeatures_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        yfeatures_paths[word] = path_ids
    f.close()
    
    # read 'paths_slotfreq.tsv' and load it into paths_slotfreq
    # dictionary. This dictionary will contain a mapping from
    # a path id to tuple (slotX_freq , slotY_freq). slotX_freq
    # is the sum of all of the word frequencies for slot X of the
    # path. Similarly, slotY_freq is the sum of all word frequencies
    # for slot Y of the path. The paths_slotfreq dictionary is required
    # for speeding up calculation of the similarity scores.
    paths_slotfreq_file = files_dir + "paths_slotfreq.tsv"
    f = open(paths_slotfreq_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        t = literal_eval(fields[1])
        paths_slotfreq[path_id] = t
    f.close()
    
    # read 'xfeatures_totalfreqs.tsv' and load it into xfeatures_totalfreqs
    # dictionary. This dictionary will contain a mapping from words
    # to the total number of frequencies the word was ever used in slot X 
    # of any path in the corpus. This dictionary is required for speeding up
    # calculation of the similarity scores.
    xfeatures_totalfreqs_file = files_dir + "xfeatures_totalfreqs.tsv"
    f = open(xfeatures_totalfreqs_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        freq = literal_eval(fields[1])
        xfeatures_totalfreqs[word] = freq
    f.close()

    # read 'yfeatures_totalfreqs.tsv' and load it into yfeatures_totalfreqs
    # dictionary. This dictionary will contain a mapping from words
    # to the total number of frequencies the word was ever used in slot Y 
    # of any path in the corpus. This dictionary is required for speeding up
    # calculation of the similarity scores.    
    yfeatures_totalfreqs_file = files_dir + "yfeatures_totalfreqs.tsv"
    f = open(yfeatures_totalfreqs_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        freq = literal_eval(fields[1])
        yfeatures_totalfreqs[word] = freq
    f.close()
    
    xfeatures_totalfreqs_sum = sum(xfeatures_totalfreqs.values())
    yfeatures_totalfreqs_sum = sum(yfeatures_totalfreqs.values())

    sentenceids_to_sentences_file = files_dir + "sentenceids_to_sentences.tsv"
    f = open(sentenceids_to_sentences_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        sen_id = literal_eval(fields[0])
        sentence = literal_eval(fields[1])
        sentenceids_to_sentences[sen_id] = sentence
    f.close()
    
    paths_sentences_file = files_dir + "paths_sentences.tsv"
    f = open(paths_sentences_file, mode="r", encoding="utf_8")
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        sentences_info = literal_eval(fields[1])
        paths_sentences[path_id] = sentences_info
    f.close()

    # read an input path from command-line
    input_path_string = sys.argv[1]
    p = None
    for path_id, path_str in pathids_to_paths.items():
        if path_str == input_path_string:
            p = path_id
            break
    if p is None:
        print("The given path not was found in the paths database")
        exit()
    
    # find all the candidate paths
    print("\nFinding the candidate path...")
    candidate_paths = get_candidate_paths(p, paths_features, xfeatures_paths, yfeatures_paths)
    
    print("\nNumber of candidate paths:", len(candidate_paths))

    # filter the candidate paths if the number of their common 
    # features with the input path is less than a fixed percent 
    # (filtering_threshold_precentage) of the total number of 
    # features for the input path and the candidate path.
    print("\nFiltering the candidate paths...")
    filtered_candidate_paths = []
    filtering_threshold_precentage = 0.01
    (slotX , slotY) = paths_features[p]
    i = 0
    progressMileStone = 0.05
    total_candidate_paths = len(candidate_paths)
    for c in candidate_paths:
        if ((i/total_candidate_paths) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
    
        common_features_count = 0
        
        for word in slotX:
            if (c in xfeatures_paths[word]):
                common_features_count += 1

        for word in slotY:
            if (c in yfeatures_paths[word]):
                common_features_count += 1

        total_features_count = len(slotX) + len(slotY) + len(paths_features[c][0]) + len(paths_features[c][1])

        if ((common_features_count * 2.0) / total_features_count) >= filtering_threshold_precentage:
            filtered_candidate_paths.append(c)
        
        i += 1
    print("100%")
    
    print("\nNumber of remaining candidate paths:", len(filtered_candidate_paths))
    
    # calculate similarity scores between the input path and the candidate paths
    # according to equations (1), (2), and (3) on page 4 of DIRT paper.
    print("\nCalculating similarity scores between the candidate paths and the input path...")
    results = []
    i = 0
    progressMileStone = 0.05
    total_filtered_candidate_paths = len(filtered_candidate_paths)
    for c in filtered_candidate_paths:
        if ((i/total_filtered_candidate_paths) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
    
        similarity_score = S(p,c)
        if (similarity_score>0):
            results.append((c , similarity_score))
            
        i += 1
    print("100%")
        
    if (len(results)==0):
        print("\nNo results found in the corpus.")
        exit()
    
    # sort the resutls based on similarity score
    results.sort(key = lambda x: x[1], reverse=True)

    # print the top k results
    print_top_k = 40
    print("\nThe top " + str(print_top_k) + " results are:")
    for i in range(min(print_top_k,len(results))):
        pid = results[i][0]
        print(pathids_to_paths[pid])
        print("   Example sentences:")
        # also print two sentences of the path
        for i in [0 , -1]:
            print("   " , sentenceids_to_sentences[paths_sentences[pid][i][-1]])
        print()


# returns the textual representation of a given path
def get_path_textual_string(path):
    
    ret_val = ""
    last_printed_element = ""
    first_element_to_print = ""
    
    for t in path:
        if (t[3] == ">"):
            first_element_to_print = t[0]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                       
            ret_val += first_element_to_print + "->" + t[2] + "->" + t[1]
            last_printed_element = t[1]
        else:
            first_element_to_print = t[1]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                    
            ret_val += first_element_to_print + "<-" + t[2] + "<-" + t[0]
            last_printed_element = t[0]
            
    return ret_val[1:-1]


# calculates similarity between two given paths according to
# equation (3) of DIRT paper
def S(p_id1, p_id2):
    return math.sqrt( sim(p_id1,p_id2,0) * sim(p_id1,p_id2,1) )


# calculates similarity between the given slot of the two given
# paths according to equation (2) of DIRT paper
def sim(p_id1, p_id2, slot_idx):

    T_p1_s = set(paths_features[p_id1][slot_idx].keys())
    T_p2_s = set(paths_features[p_id2][slot_idx].keys())

    numerator = 0
    for w in (T_p1_s & T_p2_s):
        numerator += mi(p_id1,slot_idx,w) + mi(p_id2,slot_idx,w)

    denominator_term1 = 0
    for w in T_p1_s:
        denominator_term1 += mi(p_id1,slot_idx,w)
    
    denominator_term2 = 0
    for w in T_p2_s:
        denominator_term2 += mi(p_id2,slot_idx,w)

    ret_val = numerator / (denominator_term1 + denominator_term2)
    if (ret_val < 0):
        ret_val = 0

    return ret_val


# calculates the mutual information between the given slot of the 
# given path and its filler word according to equation (1) of DIRT paper
def mi(p_id, slot_idx, word):

    frequency_count_psw = paths_features[p_id][slot_idx][word]
    
    if slot_idx==0:
        frequency_count_NsN = xfeatures_totalfreqs_sum
    else:
        frequency_count_NsN = yfeatures_totalfreqs_sum
    
    frequency_count_psN = paths_slotfreq[p_id][slot_idx]
    
    if slot_idx==0:
        frequency_count_Nsw = xfeatures_totalfreqs[word]
    else:
        frequency_count_Nsw = yfeatures_totalfreqs[word]

    return math.log((frequency_count_psw * frequency_count_NsN)
                    /
                    (frequency_count_psN * frequency_count_Nsw))

 
# returns the list of candidate paths for a given path
def get_candidate_paths(p, paths_features, xfeatures_paths, yfeatures_paths):
    
    combined_path_ids = []
    
    (slotX , slotY) = paths_features[p]

    for word in slotX:
        combined_path_ids = combined_path_ids + xfeatures_paths[word]

    for word in slotY:
        combined_path_ids = combined_path_ids + yfeatures_paths[word]

    ret_val = list(set(combined_path_ids))
    
    ret_val.remove(p)
    
    return ret_val


if __name__ == "__main__":
    main()
