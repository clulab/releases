import numpy as np
import time
import sys
import string
from sklearn import preprocessing as sklearnpreprocessing

class Gigaword:
    @classmethod
    def load_pretrained_embeddings(cls, path_to_file, args,take=None):
        sys.stdout.write("Loading the gigaword embeddings from file : " + path_to_file + "\n")

        np.random.seed(args.np_rand_seed)
        sys.stdout.flush()
        lookup = {}
        c = 0
        delimiter = " "
        embedding_size=0
        time_start_loading = time.clock()
        with open(path_to_file, "r") as f:
                    embedding_vectors = list()
                    for line in f:
                        if (take and c <= take) or not take:
                            # split line
                            line_split = line.rstrip().split(delimiter)
                            embedding_size = len(line_split)-1
                            # extract word and vector
                            word = line_split[0]
                            vector = np.array([float(i) for i in line_split[1:]])
                            # get dimension of vector
                            # add to lookup
                            lookup[word] = c
                            # add to embedding vectors
                            embedding_vectors.append(vector)
                            c += 1

                        if c % 100000 == 0:
                            sys.stdout.write("Completed loading %d lines \r" % (c))
                            sys.stdout.flush()
                    sys.stdout.write("Writing the <unk> at " + str(c) + "\n")
                    embedding_vectors.append(np.zeros((embedding_size)))
                    lookup["<unk>"] = c


                    sys.stdout.write("Writing the <pad> at " + str(c+1) + "\n")
                    embedding_vectors.append(np.ones((embedding_size))*(-1))
                    lookup["<pad>"] = c + 1

                    # assign embeddings to tags like SETe1, PERSONc1 etc: take PERSON as mean+add perturbation
                    emb_of_all_ner_neutered_tags = cls.create_perturbed_embeddings(lookup,embedding_size,embedding_vectors)
                    # import jsonlines
                    # with open('custom_emb.jsonl', 'w') as writer:
                    #     writer.write(emb_of_all_ner_neutered_tags)

                    import csv
                    with open('custom_emb.txt', 'w+') as fp:
                        counter=c+2
                        for k,v in emb_of_all_ner_neutered_tags.items():
                            embedding_vectors.append(v)
                            lookup[k] = counter
                            counter=counter+1
                            str_v = " ".join(format(x, "1.7f") for x in v)
                            fp.write(k+" "+str(str_v)+"\n")

                            #
                            # str_json=k+""+str_v+"\n"
                            # fp.write(str_json)

                    # with io.open(vocab_file, 'w+', encoding=DEFAULT_ENCODING) as f:
                    #     f.write(json.dumps(self.word_vocab))
        sys.stdout.write("[done] Completed loading " + str(c) + " lines\n")

        # sys.stdout.write("Time taken : " + str((time.clock() - time_start_loading)) + "\n")
        sys.stdout.flush()

        # time_start_vectorizing = time.clock()
        embedding_matrix = np.vstack(embedding_vectors)
        # sys.stdout.write("Converting to a 2-D numpy vector ; Time Taken : " + str((time.clock() - time_start_vectorizing)) + "\n")

        sys.stdout.write("Total time taken : " + str((time.clock()-time_start_loading)) + "\n")
        sys.stdout.flush()

        return embedding_matrix, lookup,embedding_size

    @classmethod
    def load_pretrained_dep_embeddings(cls, path_to_file, take=None):
        sys.stdout.write("Loading the gigaword embeddings from file : " + path_to_file + "\n")
        sys.stdout.write("Writing the <unk> at the end\n")
        sys.stdout.flush()
        lookup = {}
        c = 0
        delimiter = " "
        time_start_loading = time.clock()
        with open(path_to_file, "r") as f:
            embedding_vectors = list()
            for line in f:
                if (take and c <= take) or not take:
                    # split line
                    line_split = line.rstrip().split(delimiter)
                    # extract word and vector
                    word = line_split[0]
                    vector = np.array([float(i) for i in line_split[1:]])
                    # get dimension of vector
                    embedding_size = vector.shape[0]
                    # add to lookup
                    lookup[word] = c
                    # add to embedding vectors
                    embedding_vectors.append(vector)
                    c += 1

                if c % 10000 == 0:
                    sys.stdout.write("Completed loading %d lines \r" % (c))
                    sys.stdout.flush()
            embedding_vectors.append(np.zeros((embedding_size)))
            lookup["<unk>"] = c


        sys.stdout.write("[done] Completed loading " + str(c) + " lines\n")
        # sys.stdout.write("Time taken : " + str((time.clock() - time_start_loading)) + "\n")
        sys.stdout.flush()

        # time_start_vectorizing = time.clock()
        embedding_matrix = np.array(embedding_vectors)
        # sys.stdout.write("Converting to a 2-D numpy vector ; Time Taken : " + str((time.clock() - time_start_vectorizing)) + "\n")

        sys.stdout.write("Total time taken : " + str((time.clock()-time_start_loading)) + "\n")
        sys.stdout.flush()

        return embedding_matrix, lookup

    #############################################################
    #### Ported from `org.clulab.embeddings.word2vec.Word2Vec`
    #############################################################
    @classmethod
    def sanitiseWord(cls, word):
        w = word
        # if w == "-lrb-" or w == "-rrb-" or w == "-lsb-" or w == "-rsb-" :
        #     return ""
        #
        # if w.startswith("http") or ".com" in w or ".org" in w:
        #     return ""

        # if any(char.isdigit() for char in w):
        #     return "xnumx"

        # ## remove punctuations from a string: https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
        # translator = str.maketrans('', '', string.punctuation)

        # return w.translate(translator)
        return w

    @classmethod
    def create_random_embedding_for_NER_c1_kind_of_tags(cls):
        all_NER_replacements=[]
        NER_tags=["PERSON","LOCATION","PERSON","MISC","MONEY","NUMBER","DATE","PERCENT","TIME","ORGANIZATION","ORDINAL"
        , "DURATION","SET"]
        claim_evidence=["c","e"]
        nums=range(1,20)

        for ner in NER_tags:
            for ce in claim_evidence:
                for num in nums:
                    rep=ner+ce+str(num)
                    all_NER_replacements.append(rep)
        return all_NER_replacements



    #get the embedding value of PERSON and perturb it based on a guassian mean and variance for PERSONc1, PERSONc2 etc
    @classmethod
    def create_perturbed_embeddings(cls,lookup,embedding_size,embedding_vectors):
        tags_emb = {}
        NER_tags = ["PERSON", "LOCATION", "PERSON", "MISC", "MONEY", "NUMBER", "DATE", "PERCENT", "TIME",
                    "ORGANIZATION", "ORDINAL", "DURATION", "SET"]
        claim_evidence = ["c", "e"]
        nums = range(1, 20)
        for ner in NER_tags:
            if(ner in lookup.keys()):
                word_index=lookup[ner]
            else:
                word_index = lookup["<unk>"]
            ner_emb=embedding_vectors[word_index]
            for ce in claim_evidence:
                for num in nums:
                    ner_tag_claim_ev_count = ner + ce + str(num)
                    emb_normal=np.random.normal(0,0.1,embedding_size)
                    emb_tag=ner_emb+emb_normal
                    tags_emb[ner_tag_claim_ev_count]=emb_tag.tolist()
        return tags_emb



    @classmethod
    def norm(cls, embeddings):
        # if embeddings.ndim == 1:
        #     norm = np.sqrt(np.sum(np.square(embeddings)))
        # elif embeddings.ndim == 2:
        #     norm = np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))
        # else:
        #     raise ValueError('wrong number of dimensions')
        #
        # return embeddings / norm

        #### NOTE: Commenting this above code as it fails for embeddings which are zero vectors. Then this results in a NaN or Inf
        #### Instead calling the sklearn.preprocessing.normalize as it handles this gracefully.
        if embeddings.ndim == 1: ## to avoid the warning:sklearn/utils/validation.py:395: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample. DeprecationWarning)
            embeddings = np.expand_dims(embeddings, axis=0)

        normalized_vector = sklearnpreprocessing.normalize(embeddings, copy=False)
        normalized_vector_squeezed = np.squeeze(normalized_vector) ## remove the extra dimension for single dimension input vectors
        return normalized_vector_squeezed


if __name__ == '__main__':
    embedding_matrix, lookup = Gigaword.load_pretrained_embeddings("./data/vectors.txt")
    print(embedding_matrix)
    print(embedding_matrix.shape)
    # print (lookup)
