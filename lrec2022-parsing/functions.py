"""helper functions for getting corpus comparison"""

from collections import Counter
import pymagnitude as pm

def load_corpus(language):
    """
    Loads corpus files depending on selected language.
    """
    print(f"Loading {language} corpora...")
    if language=="english":
        # GUM (Georgetown University Multilayer) corpus from UD website
        corpus_a_dir = "data/ud-en-gum/"
        train_a = corpus_a_dir+"en_gum-ud-train.conllu"
        # dev_a = corpus_a_dir+"en_gum-ud-dev.conllu"

        # wsj corpus with different conventions
        # converted from Stanford dependencies (?)
        corpus_b_dir = "data/wsj-DIFF-CONVENTIONS/"
        train_b = corpus_b_dir+"train.conllu"
        # dev_b = corpus_b_dir+"dev.conllu"
        # test_B = corpus_b_dir+"test.conllu"
    else:
        print("Please select a language!")
    print("Corpora loaded!")

    return train_a, train_b


def load_vectors(language):
    """
    Loads vector file depending on selected language.
    """
    if language == "english":
        vector_file = "./glove.840B.300d.magnitude"
        vector_type = "GloVe"
    if len(vector_file) > 0:
        return vector_file, vector_type
    print(f"No vector file found for {language}!")
    raise ValueError("No vector file found!")


def process_training_data(filename):
    """
    Processes the training data to get head-dependent-relation triples.
    """
    with open(filename) as infile:
        file_text = infile.read()
    sents = file_text.rstrip().split("\n\n")
    num_tokens = 0
    num_sents = len(sents)
    pair_to_relations = {}
    pair_relation_to_sentences = {}
    for sent in sents:
        sentence_text = []
        for line in sent.split("\n"):
            word = line.split("\t")[1]
            sentence_text.append(word)
        for line in sent.split("\n"):
            line = line.split("\t")
            word = line[1]
            num_tokens += 1
            head_idx = int(line[6])
            head_word = sent.split("\n")[head_idx-1].split("\t")[1] if head_idx != 0 else "#ROOT#"
            relation = line[7]
            word_pair = (head_word, word)
            if word_pair in pair_to_relations:
                pair_to_relations[word_pair].append(relation)
            else:
                pair_to_relations[word_pair] = [relation]
            pair_relation = (word_pair, relation)
            if pair_relation in pair_relation_to_sentences:
                pair_relation_to_sentences[pair_relation].append(" ".join(sentence_text))
            else:
                pair_relation_to_sentences[pair_relation] = [" ".join(sentence_text)]
    print(f"{num_sents:,} sentences processed")
    print(f"{num_tokens:,} tokens processed")
    print(f"Average sentence length:\t{num_tokens/num_sents}")
    print(f"{len(pair_to_relations):,} head-dependent:relation pairs")
    print(f"{len(pair_relation_to_sentences):,} head-dependent-relation:sentence triples")

    return pair_to_relations, pair_relation_to_sentences


def generate_human_readable_output(filename, mismatches, sentences, this_data, other_data):
    """
    Generates human-readable output file for evaluation.

    Args:
        filename: the output filename

        mismatches: the dictionary of word pairs with a relation in one file but not the other

        sentences: the dictionary of word pairs, their relations, and the sentences they're found in
    """
    conversion_dict = {}
    with open(filename, "w") as output:
        header = ("SENTENCE" + "\t" +
                  "HEAD WORD" + "\t" +
                  "DEPENDENT WORD" + "\t" +
                  "RELATION" + "\t" +
                  "TOP RELATION IN THIS DATA" + "\t" +
                  "COUNT OF TOP RELATION IN THIS DATA" + "\t" +
                  "PROPORTION OF TOP RELATION IN THIS DATA" + "\t"
                  "TOP RELATION IN OTHER DATA" + "\t" +
                  "COUNT OF TOP RELATION IN OTHER DATA" + "\t" +
                  "PROPORTION OF TOP RELATION IN OTHER DATA" + "\n")
        output.write(header)
        for pair in mismatches:
            head_word = pair[0]
            dependent_word = pair[1]
            # relations for this pair in this partition / corpus
            these_relations = Counter(this_data[pair])
            # relations for this pair in other partition / corpus
            other_relations = Counter(other_data[pair])
            for relation in list(set(mismatches[pair])):
                triple = (pair, relation)
                sentence_texts = []
                for sentence in sentences[triple]:
                    sentence_texts.append(sentence)
                # get most common label/count for this data and other data
                most_common_label = these_relations.most_common(1)[0][0]
                most_common_count = these_relations.most_common(1)[0][1]
                sum_these_relations = sum(these_relations.values())
                most_common_label_other = other_relations.most_common(1)[0][0]
                most_common_count_other = other_relations.most_common(1)[0][1]
                sum_other_relations = sum(other_relations.values())
                line = ("('"+("', '").join(sentence_texts)+"')" + "\t" +
                        head_word + "\t" +
                        dependent_word + "\t" +
                        relation + "\t" +
                        most_common_label + "\t" +
                        str(most_common_count) + "\t" +
                        str(most_common_count/sum_these_relations) + "\t" +
                        most_common_label_other + "\t" +
                        str(most_common_count_other) + "\t" +
                        str(most_common_count_other/sum_other_relations)+ "\n"
                        )
                key = (head_word, dependent_word, relation)
                value = {"most_common_label": most_common_label,
                         "most_common_count": most_common_count,
                         "proportion_these": most_common_count/sum_these_relations,
                         "most_common_label_other": most_common_label_other,
                         "most_common_count_other": most_common_count_other,
                         "proportion_other": most_common_count_other/sum_other_relations}
                conversion_dict[key] = value
                output.write(line)
    return conversion_dict


def get_conversions_simple(mismatches, sentences, this_data, other_data):
    """
    Generates conversion table for this_data into that_data

    Args:
        mismatches: the dictionary of word pairs with a relation in one file but not the other

        sentences: the dictionary of word pairs, their relations, and the sentences they're found in
    """
    conversion_dict = {}
    for pair in mismatches:
        head_word = pair[0]
        dependent_word = pair[1]
        # relations for this pair in this partition / corpus
        these_relations = Counter(this_data[pair])
        # relations for this pair in other partition / corpus
        other_relations = Counter(other_data[pair])
        for relation in list(set(mismatches[pair])):
            triple = (pair, relation)
            sentence_texts = []
            for sentence in sentences[triple]:
                sentence_texts.append(sentence)
            # get most common label/count for this data and other data
            most_common_label = these_relations.most_common(1)[0][0]
            most_common_count = these_relations.most_common(1)[0][1]
            sum_these_relations = sum(these_relations.values())
            most_common_label_other = other_relations.most_common(1)[0][0]
            most_common_count_other = other_relations.most_common(1)[0][1]
            sum_other_relations = sum(other_relations.values())
            key = (head_word, dependent_word, relation)
            value = {"most_common_label": most_common_label,
                     "most_common_count": most_common_count,
                     "proportion_these": most_common_count/sum_these_relations,
                     "most_common_label_other": most_common_label_other,
                     "most_common_count_other": most_common_count_other,
                     "proportion_other": most_common_count_other/sum_other_relations}
            conversion_dict[key] = value

    return conversion_dict


def get_conversions_pretrained(mismatches,
                               this_data,
                               other_data,
                               vector_file,
                               top_n,
                               threshold):
    """
    Generates conversion table for this_data into that_data using pretrained word embeddings.

    Args:
        mismatches: the dictionary of word pairs with a relation in one file but not the other
    """
    vectors = pm.Magnitude(vector_file)

    conversion_dict = {}

    total = 0
    diff = 0
    pairs = 0
    total_top_n = 0
    above_threshold = 0
    potential_pairs = 0
    actual_pairs = 0

    for pair in mismatches:
        pairs += 1
        head_word = pair[0]
        dependent_word = pair[1]

        head_word_replacements = [x[0] for x in vectors.most_similar(head_word, topn=top_n)
                                  if x[0] != head_word and x[1] >= threshold]
        dependent_word_replacements = [x[0] for x in vectors.most_similar(dependent_word, topn=top_n)
                                       if x[0] != dependent_word and x[1] >= threshold]

        total_top_n += top_n*2

        above_threshold += len(head_word_replacements)+len(dependent_word_replacements)

        potential_new_pairs = []
        new_pairs = []
        for head in [head_word]+head_word_replacements:
            for dependent in [dependent_word]+dependent_word_replacements:
                potential_new_pairs.append((head,dependent))
                potential_pairs += 1
                if (head, dependent) in list(other_data.keys()) and (head, dependent) != pair:
                    actual_pairs += 1
                    new_pairs.append((head,dependent))

        # relations for this pair in this corpus
        these_relations = Counter(this_data[pair])
        # relations for this pair in other corpus
        other_relations = Counter(other_data[pair])
        # relations for new pairs in other corpus
        new_relations = sum([Counter(other_data[new_pair]) for new_pair in new_pairs], Counter())

        # get most common label/count for this data and other data
        most_common_label = these_relations.most_common(1)[0][0]
        most_common_count = these_relations.most_common(1)[0][1]
        sum_these_relations = sum(these_relations.values())

        # don't weigh exact pair relation different
        no_weigh = other_relations + new_relations
        most_common_label_other = no_weigh.most_common(1)[0][0]
        most_common_count_other = no_weigh.most_common(1)[0][1]
        sum_other_relations = sum(no_weigh.values())
        if other_relations.most_common(1)[0][0]==most_common_label_other:
            total += 1
        else:
            total += 1
            diff += 1

        for relation in list(set(mismatches[pair])):
            key = (head_word, dependent_word, relation)
            value = {"most_common_label": most_common_label,
                     "most_common_count": most_common_count,
                     "proportion_these": most_common_count/sum_these_relations,
                     "most_common_label_other": most_common_label_other,
                     "most_common_count_other": most_common_count_other,
                     "proportion_other": most_common_count_other/sum_other_relations}
            conversion_dict[key] = value

    return conversion_dict


def apply_conversions(input_file, output_file, conversion_dictionary):
    """
    Applies the conversions and creates a new conllu file.
    """
    with open(input_file) as infile:
        lines = infile.readlines()
    sentences = []
    sentence = []
    for line in lines:
        if len(line.strip()) == 0:
            sentences.append(sentence)
            sentence = []
        else:
            split = line.split("\t")
            sentence.append(split)
    for sentence in sentences:
        for sent in sentence:
            word = sent[1]
            head_idx = int(sent[6])
            head_word = sentence[head_idx-1][1] if head_idx != 0 else "#ROOT#"
            relation = sent[7]
            triple = (head_word, word, relation)
            if conversion_dictionary.get(triple):
                new_relation = conversion_dictionary[triple]["most_common_label_other"]
                sent[7] = new_relation
            else:
                continue
        with open(output_file, "a") as outfile:
            for sent in sentence:
                outfile.write("\t".join(sent))
            outfile.write("\n")
            