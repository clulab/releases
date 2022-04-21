import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torchtext
import torch
from ast import literal_eval
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Pool
import scipy.cluster.hierarchy as hac
import tqdm 
import matplotlib.pyplot as plt
import json

"""
Simple script to produce break a dataset into multiple clusters
Ideally, each cluster should be learnable
"""


def append_highlighted_part(x):
    if x['reversed'] == 0:
        return x['subj_type'].lower() + ' ' + ' '.join(x['highlighted']).lower() + ' ' + x['obj_type'].lower()
    else:
        return x['obj_type'].lower() + ' ' + ' '.join(x['highlighted']).lower() + ' ' + x['subj_type'].lower()




# stop_words from nltk
stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", 
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", 
    "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", 
    "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", 
    "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", 
    "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", 
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", 
    "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", 
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", 
    "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", 
    "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", 
    "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according", "accordingly", "across", "act", "actually", 
    "added", "adj", "affected", "affecting", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always", 
    "among", "amongst", "announce", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently", 
    "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b", "back", "became", "become", 
    "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "believe", "beside", "besides", "beyond", "biol", 
    "brief", "briefly", "c", "ca", "came", "cannot", "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", 
    "containing", "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight", "eighty", 
    "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", 
    "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "fix", "followed", "following", "follows", "former", 
    "formerly", "forth", "found", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", 
    "gone", "got", "gotten", "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", 
    "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance", "important", "inc", "indeed", 
    "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k", "keep", "keeps", "kept", "kg", "km", "know", "known", 
    "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", 
    "line", "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", 
    "meantime", "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug", "must", "n", "na", 
    "name", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", 
    "nine", "ninety", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", 
    "obviously", "often", "oh", "ok", "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", 
    "p", "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus", "poorly", "possible", 
    "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily", "probably", "promptly", "proud", "provides", "put", "q", 
    "que", "quickly", "quite", "qv", "r", "ran", "rather", "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", 
    "regardless", "regards", "related", "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", 
    "say", "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", 
    "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows", "significant", "significantly", "similar", "similarly", 
    "since", "six", "slightly", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", 
    "sorry", "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially", "successfully", 
    "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", 
    "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", 
    "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", 
    "together", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", 
    "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "v", "value", 
    "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", 
    "what'll", "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim", 
    "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish", "within", "without", "wont", "words", 
    "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", 
    "appropriate", "associated", "best", "better", "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", 
    "considering", "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", 
    "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", 
    "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious", "seriously", "sure", "t's", "third", "thorough", 
    "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", 
    "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", 
    "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", 
    "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", 
    "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", 
    "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", 
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", 
    "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", 
    "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", 
    "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", 
    "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", 
    "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", 
    "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", 
    "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", 
    "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", 
    "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", 
    "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", 
    "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", 
    "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", 
    "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", 
    "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", 
    "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", 
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op", "research-articl", "pagecount", "cit", "ibid", 
    "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", 
    "6b", "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", 
    "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", 
    "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", 
    "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", 
    "ep", "eq", "er", "es", "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", 
    "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", 
    "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", 
    "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", 
    "nc", "nd", "ne", "ng", "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", 
    "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", 
    "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", 
    "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", 
    "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", 
    "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", 
    "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]


      
def save_to_csv(df, where_to_save):
    if df[df['highlighted'].apply(lambda x: x == [])].shape[0] == df.shape[0]:
        df.to_csv(where_to_save, sep='\t', quoting=3, header=True, index=True)        
    elif df[df['highlighted'].apply(lambda x: x == [])].shape[0] > 0:
        empty_df    = df[df['highlighted'].apply(lambda x: x == [])]
        nonempty_df = df[df['highlighted'].apply(lambda x: x != [])]
        empty_df.to_csv(f'{where_to_save}_empty', sep='\t', quoting=3, header=True, index=True)
        nonempty_df.to_csv(f'{where_to_save}_nonempty', sep='\t', quoting=3, header=True, index=True)
    else:
        df.to_csv(where_to_save, sep='\t', quoting=3, header=True, index=True)        

def cluster(rel_emb, threshold=0.1):
    linkage = hac.linkage(rel_emb, metric='cosine', method='complete')
    labels = np.array(hac.fcluster(linkage, threshold, criterion="distance"))
    return labels


def dump_to_file(df, dfe, current_path, suffix='', max_size=10, rec_level=0, threshold=0.1):
    # Cannot call cluster when there is only one entry, as there will be no distance matrix
    # In that case, just save it to file. It is a valid cluster of size 1
    if df.shape[0] == 1:
        save_to_csv(df, f'{current_path}{suffix}_i')
    else:
        labels = cluster(dfe, threshold=threshold)
        number_of_labels = np.unique(labels).shape[0]

        for i in range(1, number_of_labels+1):
            if(rec_level < 10):
                if(df[labels==i].shape[0] > max_size):
                    dump_to_file(df[labels==i], dfe[labels==i], current_path, suffix + '_' + str(i), max_size, rec_level + 1, threshold*0.5)
                else:
                    save_to_csv(df[labels==i], f'{current_path}{suffix}_{i}')
                    # df[labels==i].to_csv(f'{current_path}{suffix}_{i}', sep='\t', quoting=3, header=True, index=True)
            else:
                save_to_csv(df[labels==i], f'{current_path}{suffix}_{i}')
                # df[labels==i].to_csv(f'{current_path}{suffix}_{i}', sep='\t', quoting=3, header=True, index=True)


def agglomerative_clustering(filename: str = '/data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/train_processed.json', base_path = '/data/nlp/corpora/odinsynth/data/TACRED/odinsynth_tacred102', max_size = 5, threshold = 0.1):
    data = pd.read_csv(filename, sep='\t', quoting=3, converters={"highlighted": literal_eval})

    # print(data[data['relation']=='org:founded_by']['highlighted'])
    
    data['highlighted_string'] = data.apply(lambda x: append_highlighted_part(x), axis=1)
    
    # used for computing the embedding
    glove = torchtext.vocab.GloVe(name='840B', dim=300)
    
    relations = sorted(list(set(data['relation'].tolist())))
    # print(relations)
    relno = list(zip(relations, range(len(relations))))

    for is_reversed in [0, 1]:
        for relation, relation_number in relno[1:]:
            p = relation.replace(':','_').replace('/', '_')
            max_size = 1000
            if not os.path.exists(f'{base_path}/{relation_number}.{p}'):
                os.mkdir(f'{base_path}/{relation_number}.{p}')
            current_path = f'{base_path}/{relation_number}.{p}/cluster_r{is_reversed}'

            # Some filtering (by relation and by reverse)
            data_rel = data[data['relation'] == relations[relation_number]] # per:date_of_death
            data_rel = data_rel[data_rel['reversed']==is_reversed]

            # Cluster only if there is something to cluster
            if data_rel.shape[0] != 0:
                data_rel_list = [] # holds the text
                data_rel_emb  = [] # holds the embeddings

                for i in range(data_rel.shape[0]):
                    d = data_rel.iloc[i]
                    text = d['highlighted_string']
                    # text = [t.lower() for t in text]
                    data_rel_list.append(text)
                    emb = torch.cat([glove[x.lower()].unsqueeze(0) for x in text], dim=0).mean(0)
                    data_rel_emb.append(emb.unsqueeze(0))

                data_rel_emb = torch.cat(data_rel_emb, dim=0)

                drll = np.array([len(x) for x in data_rel_list]) # data_rel_list lengths
                data_rel = data_rel[drll<max_size]

                drl = np.array(data_rel_list)[drll<max_size] # np array of data_rel_list
                data_rel_emb = data_rel_emb[drll<max_size]

                dump_to_file(data_rel, data_rel_emb, current_path, suffix='', max_size=max_size, threshold=threshold)

# 2, 3!, 14, 16!?, 17!?,  25, 29, 30, 31!!, 40!!, 41!, 

"""
# """
# Cluster things that appear multiple times together
# For the rest, do an agglomerative clustering
"""
def cluster_identicals():
    for is_reversed in [0, 1]:
        for relation, relation_number in relno[1:][-2:][-1:]:
            p = relation.replace(':','_').replace('/', '_')
            max_size = 1000
            if not os.path.exists(f'{base_path}/{relation_number}.{p}'):
                os.mkdir(f'{base_path}/{relation_number}.{p}')
            current_path = f'{base_path}/{relation_number}.{p}/cluster_r{is_reversed}'

            data_rel = data[data['relation'] == relations[relation_number]] # per:date_of_death
            data_rel = data_rel[data_rel['reversed']==is_reversed]
            grouped = data_rel.groupby('highlighted_string')
            idx = 0
            counts  = data_rel['highlighted_string'].value_counts()
            # counts_mean = data_rel['highlighted_string'].value_counts().mean()
            # counts_std  = data_rel['highlighted_string'].value_counts().std()

            df_under_thresholds = []

            for text, df in grouped:
                if df.shape[0] > 4:
                    df.to_csv(f'{current_path}_{idx}', sep='\t', quoting=3, header=True, index=True)
                else:
                    df_under_thresholds.append(df)
                idx += 1
            # exit()
            
def stats():
    def cos(x, y):
        return torch.dot(x, y)/torch.sqrt(torch.dot(x,x) * torch.dot(y,y))


    data['highlighted_string_embedding'] = data.apply(lambda z: torch.cat([glove[x.lower()].unsqueeze(0) for x in z['highlighted_string'].split(' ')], dim=0).mean(0), axis=1)
    embeddings = torch.cat([x.unsqueeze(dim=0) for x in data['highlighted_string_embedding']], dim=0)
    lengths = np.array([len(x.split(' ')) for x in data['highlighted_string']])
    print(np.mean(lengths))
    print(np.median(lengths))
    # exit()
    # embeddings_len_l10 = embeddings[lengths<10]

    embeddings_len_g10 = embeddings[lengths>10]
    embeddings_len_g20 = embeddings[lengths>20]
    # embeddings_len_g30 = embeddings[lengths>30]
    # embeddings_len_g40 = embeddings[lengths>40]
    # embeddings_len_g50 = embeddings[lengths>50]

    # random_numbers_l10 = np.random.choice(range(embeddings_len_l10.shape[0]), (10000, 2))
    # random_numbers_l10 = random_numbers_l10[random_numbers_l10[:,0] != random_numbers_l10[:,1]]
    
    random_numbers_g10 = np.random.choice(range(embeddings_len_g10.shape[0]), (10000, 2))
    random_numbers_g10 = random_numbers_g10[random_numbers_g10[:,0] != random_numbers_g10[:,1]]
    
    random_numbers_g20 = np.random.choice(range(embeddings_len_g20.shape[0]), (10000, 2))
    random_numbers_g20 = random_numbers_g20[random_numbers_g20[:,0] != random_numbers_g20[:,1]]
    
    # random_numbers_g30 = np.random.choice(range(embeddings_len_g30.shape[0]), (10000, 2))
    # random_numbers_g30 = random_numbers_g30[random_numbers_g30[:,0] != random_numbers_g30[:,1]]
    
    # random_numbers_g40 = np.random.choice(range(embeddings_len_g40.shape[0]), (10000, 2))
    # random_numbers_g40 = random_numbers_g40[random_numbers_g40[:,0] != random_numbers_g40[:,1]]

    # random_numbers_g50 = np.random.choice(range(embeddings_len_g50.shape[0]), (10000, 2))
    # random_numbers_g50 = random_numbers_g50[random_numbers_g50[:,0] != random_numbers_g50[:,1]]

    random_numbers_all = np.random.choice(range(embeddings.shape[0]), (100000, 2))
    random_numbers_all = random_numbers_all[random_numbers_all[:,0] != random_numbers_all[:,1]]

    # mean_l10 = np.mean([cos(embeddings_len_l10[x[0]], embeddings_len_l10[x[1]]) for x in random_numbers_l10])

    mean_g10 = np.mean([cos(embeddings_len_g10[x[0]], embeddings_len_g10[x[1]]) for x in random_numbers_g10])
    mean_g20 = np.mean([cos(embeddings_len_g20[x[0]], embeddings_len_g20[x[1]]) for x in random_numbers_g20])
    # mean_g30 = np.mean([cos(embeddings_len_g30[x[0]], embeddings_len_g30[x[1]]) for x in random_numbers_g30])
    # mean_g40 = np.mean([cos(embeddings_len_g40[x[0]], embeddings_len_g40[x[1]]) for x in random_numbers_g40])
    # mean_g50 = np.mean([cos(embeddings_len_g50[x[0]], embeddings_len_g50[x[1]]) for x in random_numbers_g50])

    mean_all = np.mean([cos(embeddings[x[0]], embeddings[x[1]]) for x in random_numbers_all])
    random_numbers1 = np.random.choice(range(embeddings.shape[0]), (100000, 2))

    coerrcoef_all_cos      = [cos(embeddings[x[0]], embeddings[x[1]]) for x in random_numbers_all]
    coerrcoef_all_lengths1 = [min(lengths[x[0]], lengths[x[1]]) for x in random_numbers_all]
    coerrcoef_all_lengths2 = [max(lengths[x[0]], lengths[x[1]]) for x in random_numbers_all]
    print(np.corrcoef(coerrcoef_all_lengths1, coerrcoef_all_cos))
    print(np.corrcoef(coerrcoef_all_lengths2, coerrcoef_all_cos))

    # print(mean_l10)

    print(mean_g10)
    print(mean_g20)
    # print(mean_g30)
    # print(mean_g40)
    # print(mean_g50)
    print(mean_all)
"""

"""
Reads each episode into a dataframe which will contain its supporting sentences (with all the relations that are available in the episode)
Then, saves the dataframe like "df_ep{episode_number}"

After this function is applied, the supporting sentences of the episode "i" can be read as they are in agglomerative_clustering function:
    data = pd.read_csv(filename, sep='\t', quoting=3, converters={"highlighted": literal_eval})
"""
def from_fewshot_tacred_to_processed_json(path: str, savepath: str): 
    with open(path) as fin:
        data = json.load(fin)

    
    episodes_data  = data[0]
    relations_data = data[2]

    dataframes = []
    for (i, (episodes, relations)) in enumerate(zip(episodes_data, relations_data)):
        train_relations = relations[0]
        test_relations  = relations[1]
        train_data      = episodes['meta_train']
        test_data       = episodes['meta_test']

        data_for_episode = []
        for (train_rel_eps, train_rel) in zip(train_data, train_relations):
            for tre in train_rel_eps:
                is_reversed = 1 if tre['subj_start'] > tre['obj_start'] else 0
                if is_reversed == 1:
                    highlighted_tokens = tre['token'][(tre['obj_end'] + 1):tre['subj_start']]
                else:
                    highlighted_tokens = tre['token'][(tre['subj_end'] + 1):tre['obj_start']]

            # subj_start, subj_end, subj_type, obj_start, obj_end, obj_type, highlighted, tokens, reversed, relation
                data_for_episode.append([
                    tre['subj_start'],
                    tre['subj_end'],
                    tre['subj_type'],
                    tre['obj_start'],
                    tre['obj_end'],
                    tre['obj_type'],
                    highlighted_tokens,
                    tre['token'],
                    is_reversed,
                    train_rel
                ])
        
        df = pd.DataFrame(data_for_episode, columns = ['subj_start', 'subj_end', 'subj_type', 'obj_start', 'obj_end', 'obj_type', 'highlighted', 'tokens', 'reversed', 'relation'])
        dataframes.append([df, i])
        
    for (df, i) in dataframes:
        df.to_csv(f'{savepath}/df_ep{i}', sep='\t', quoting=3, header=True, index=False)

"""
Helper function to be used inside fewshot_tacred_data_prep with Pool
"""
def multi_processing_function(path):
    last_part = path.split('/')[-1]
    base_path = '/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0/'
    agglomerative_clustering(path, f'{base_path}/{last_part}', max_size=2, threshold = 0.005)

"""
Read the data in .json format, as they are after applying from_fewshot_tacred_to_processed_json
on the original data generated using the Few-Shot TACRED scripts
Then, attempts to cluster everything with the agglomerative_clustering
In this scenario, the support data in an episode represents the "data" to be "clustered"
Compared to when using the full data where the whole train partition of TACRED was clustered
"""
def fewshot_tacred_data_prep():
    # Read each episode support sentences and save the resulting dataframes to their own files
    from_fewshot_tacred_to_processed_json('/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json', '/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/episodes_0/')
    import glob
    dataframes_paths = glob.glob('/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/episodes_0/*')
    base_path = '/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0/'
    # Create folders for the clusters
    # For easier access, we create them like clusters/df_ep{ep_number}/{relation}/cluster_r{reversed}_{number}
    for dp in dataframes_paths:
        last_part = dp.split('/')[-1]
        if not os.path.exists(f'{base_path}/{last_part}'):
            os.mkdir(f'{base_path}/{last_part}')
        
    pool = Pool(40)
    pool.map(multi_processing_function, dataframes_paths)
    unique_clusters('/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0/', '/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0_unique/')

def df_to_set_of_tuples(filename): 
    data = pd.read_csv(filename, sep='\t', index_col=0, quoting=3, converters={"highlighted": literal_eval, "tokens": literal_eval})
    list_of_tuples = []
    for l in data.values.tolist():
        list_of_tuples.append(tuple([tuple(x) if type(x)==list else x for x in l]))
    result = frozenset(list_of_tuples)
    
    return result

def set_of_tuples_to_list_of_lists(tuples):
    result = []
    for t in list(tuples):
        result.append(list([list(x) if type(x) == tuple else x for x in t]))
        
    return result


def unique_clusters(basepath: str, savepath: str):
    import glob
    from collections import defaultdict
    clusters = glob.glob(f'{basepath}/*/*/*')

    # Store the initial header; Will be used when re-creating the dataframes
    header = pd.read_csv(clusters[0], sep='\t', index_col=0, quoting=3, converters={"highlighted": literal_eval, "tokens": literal_eval}).columns

    uniques = set()
    identical_clusters = defaultdict(list)
    
    for c in tqdm.tqdm(clusters):
        df = df_to_set_of_tuples(c)
        identical_clusters[df].append(c)
        uniques.add(df)

    print("We have", len(uniques), "unique clusters. The relation between original cluster and its representative (the one that is saved) is at identical_clusters_paths")
    uniques_list = list(uniques)
    reversed_dictionary = {}
    for i, c in tqdm.tqdm(enumerate(uniques_list)):
        for cluster_path in identical_clusters[c]:
            reversed_dictionary[cluster_path] = f'cluster_{i}'
        ll = set_of_tuples_to_list_of_lists(c)
        df = pd.DataFrame(ll, columns=header)
        df.to_csv(f'{savepath}/cluster_{i}', sep='\t', quoting=3, header=True, index=True)
    
    with open(f'{savepath}/identical_clusters_paths', 'w+') as fout:
        json.dump(reversed_dictionary, fout, indent=4, sort_keys=True)


if __name__ == "__main__":
    # df_to_tuple_of_tuples('/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0/df_ep7092/1.org_parents/cluster_r1_1')
    fewshot_tacred_data_prep()
    # agglomerative_clustering()
