import json
import re
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel
import os

import sys
from pathlib import Path

parent_folder_path = str(Path('.').absolute().parent)
sys.path+=[parent_folder_path]

def print_paragraph(paragraph):
    print("="*20)
    print("paragraph")
    for sent in paragraph.split(". "):
        print("\t"+sent)

def simple_check(sentences_dev_json):
    # sentences_dev_json has two keys: data and version
    # data has 48 elements
    # each data element has two keys: title and paragraphs
    # data 1 has 54 paragraphs
    # each paragraph has two keys: context and qas, context is a string and qas is a list
    # qas has fields "answers",

    # print_paragraph(sentences_dev_json["data"][0]["paragraphs"][0]["context"])
    # print(sentences_dev_json["data"][0]["paragraphs"][0]["qas"])
    question_0_context = sentences_dev_json["data"][0]["paragraphs"][0]["context"]
    question_0_ques = sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["question"]
    question_0_answer_0_start = int(sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"])
    question_0_answer_0_text = sentences_dev_json["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    print("question 0 question:", question_0_ques)
    print("answer accessed:", question_0_context[question_0_answer_0_start:question_0_answer_0_start+50])
    print("true answer:", question_0_answer_0_text)

    return 0



# See this repo about how to convert the squad and nq data to retrieval dataset.
# https://github.com/google/retrieval-qa-eval/blob/master/sb_sed.py

def _infer_sentence_breaks(uni_text):
  """Generates (start, end) pairs demarking sentences in the text.
  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.
  Yields:
    (start, end) tuples that demarcate sentences in the input text. Normal
    Python slicing applies: the start index points at the first character of
    the sentence, and the end index is one past the last character of the
    sentence.
  """
  # Treat the text as a single line that starts out with no internal newline
  # characters and, after regexp-governed substitutions, contains internal
  # newlines representing cuts between sentences.
  uni_text = re.sub(r'\n', r' ', uni_text)  # Remove pre-existing newlines.
  text_with_breaks = _sed_do_sentence_breaks(uni_text)
  starts = [m.end() for m in re.finditer(r'^\s*', text_with_breaks, re.M)]
  sentences = [s.strip() for s in text_with_breaks.split('\n')]
  assert len(starts) == len(sentences)
  for i in range(len(sentences)):
    start = starts[i]
    end = start + len(sentences[i])
    yield start, end


def _sed_do_sentence_breaks(uni_text):
  """Uses regexp substitution rules to insert newlines as sentence breaks.
  Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.
  Returns:
    A Unicode string with internal newlines representing the inferred sentence
    breaks.
  """

  # The main split, looks for sequence of:
  #   - sentence-ending punctuation: [.?!]
  #   - optional quotes, parens, spaces: [)'" \u201D]*
  #   - whitespace: \s
  #   - optional whitespace: \s*
  #   - optional opening quotes, bracket, paren: [['"(\u201C]?
  #   - upper case letter or digit
  txt = re.sub(r'''([.?!][)'" %s]*)\s(\s*[['"(%s]?[A-Z0-9])''' % ('\u201D', '\u201C'),
               r'\1\n\2',
               uni_text)

  # Wiki-specific split, for sentence-final editorial scraps (which can stack):
  #  - ".[citation needed]", ".[note 1] ", ".[c] ", ".[n 8] "
  txt = re.sub(r'''([.?!]['"]?)((\[[a-zA-Z0-9 ?]+\])+)\s(\s*['"(]?[A-Z0-9])''',
               r'\1\2\n\4', txt)

  # Wiki-specific split, for ellipses in multi-sentence quotes:
  # "need such things [...] But"
  txt = re.sub(r'(\[\.\.\.\]\s*)\s(\[?[A-Z])', r'\1\n\2', txt)

  # Rejoin for:
  #   - social, military, religious, and professional titles
  #   - common literary abbreviations
  #   - month name abbreviations
  #   - geographical abbreviations
  #
  txt = re.sub(r'\b(Mrs?|Ms|Dr|Prof|Fr|Rev|Msgr|Sta?)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Lt|Gen|Col|Maj|Adm|Capt|Sgt|Rep|Gov|Sen|Pres)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(e\.g|i\.?e|vs?|pp?|cf|a\.k\.a|approx|app|es[pt]|tr)\.\n',
               r'\1. ',
               txt)
  txt = re.sub(r'\b(Jan|Aug|Oct|Nov|Dec)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b(Mt|Ft)\.\n', r'\1. ', txt)
  txt = re.sub(r'\b([ap]\.m)\.\n(Eastern|EST)\b', r'\1. \2', txt)

  # Rejoin for personal names with 3,2, or 1 initials preceding the last name.
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3 \4',
               txt)
  txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3',
               txt)
  txt = re.sub(r'\b([A-Z]\.[A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)
  txt = re.sub(r'\b([A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)

  # Resplit for common sentence starts:
  #   - The, This, That, ...
  #   - Meanwhile, However,
  #   - In, On, By, During, After, ...
  txt = re.sub(r'([.!?][\'")]*) (The|This|That|These|It) ', r'\1\n\2 ', txt)
  txt = re.sub(r'(\.) (Meanwhile|However)', r'\1\n\2', txt)
  txt = re.sub(r'(\.) (In|On|By|During|After|Under|Although|Yet|As |Several'
               r'|According to) ',
               r'\1\n\2 ',
               txt)

  # Rejoin for:
  #   - numbered parts of documents.
  #   - born, died, ruled, circa, flourished ...
  #   - et al (2005), ...
  #   - H.R. 2000
  txt = re.sub(r'\b([Aa]rt|[Nn]o|Opp?|ch|Sec|cl|Rec|Ecl|Cor|Lk|Jn|Vol)\.\n'
               r'([0-9IVX]+)\b',
               r'\1. \2',
               txt)
  txt = re.sub(r'\b([bdrc]|ca|fl)\.\n([A-Z0-9])', r'\1. \2', txt)
  txt = re.sub(r'\b(et al)\.\n(\(?[0-9]{4}\b)', r'\1. \2', txt)
  txt = re.sub(r'\b(H\.R\.)\n([0-9])', r'\1 \2', txt)

  # SQuAD-specific joins.
  txt = re.sub(r'(I Am\.\.\.)\n(Sasha Fierce|World Tour)', r'\1 \2', txt)
  txt = re.sub(r'(Warner Bros\.)\n(Records|Entertainment)', r'\1 \2', txt)
  txt = re.sub(r'(U\.S\.)\n(\(?\d\d+)', r'\1 \2', txt)
  txt = re.sub(r'\b(Rs\.)\n(\d)', r'\1 \2', txt)

  # SQuAD-specific splits.
  txt = re.sub(r'\b(Jay Z\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Washington, D\.C\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(for 4\.\)) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\b(Wii U\.) ([A-Z])', r'\1\n\2', txt)
  txt = re.sub(r'\. (iPod|iTunes)', r'.\n\1', txt)
  txt = re.sub(r' (\[\.\.\.\]\n)', r'\n\1', txt)
  txt = re.sub(r'(\.Sc\.)\n', r'\1 ', txt)
  txt = re.sub(r' (%s [A-Z])' % '\u2022', r'\n\1', txt)
  return txt

#simple_check(sentences_dev_json)

# The following script comes from eval squad

def _generate_examples(data):
    '''
    This function is used for get the question, answer sentence and the context for the whole dataset.
    :param data: json converted to dictionary
    :return:
    '''
    counts = [0, 0, 0]
    for passage in data["data"]:
        counts[0] += 1
        for paragraph in passage["paragraphs"]:
            counts[1] += 1

            paragraph_text = paragraph["context"]
            sentence_breaks = list(_infer_sentence_breaks(paragraph_text))

            para = [str(paragraph_text[start:end]) for (start, end) in sentence_breaks]

            for qas in paragraph["qas"]:
                # The answer sentences that have been output for the current question.
                answer_sentences = set()  # type: Set[str]

                counts[2] += 1
                for answer in qas["answers"]:
                    answer_start = answer["answer_start"]
                    # Map the answer fragment back to its enclosing sentence.
                    sentence = None
                    for start, end in sentence_breaks:
                        if start <= answer_start < end:
                            sentence = paragraph_text[start:end]
                            break

                    # Avoid generating duplicate answer sentences.
                    if sentence not in answer_sentences:
                        answer_sentences.add(str(sentence))
                        yield (qas["question"], qas["id"], str(sentence), paragraph_text, para, answer["text"])


def convert_nq_to_retrieval(nq_json_train_path = parent_folder_path+"/data_raw/NQ/nq_train.json", pickle_save_folder_path = parent_folder_path+"/data_generated/nq_retrieval_raw", pickle_save_file_path = "nq_retrieval_data"):
    '''
    This function is use for converting the original squad data to retrieval data.
    Note that the sentences and contexts are shared by the train set and the test set.
    :param squad_json_train: original json file of squad train
    :param squad_json_test: original json file of squad dev
    :return: squad_retrieval_train, a list of dicts. [{id:str, question:str, answer_sent:int, context:int, label:int, gold_answer_text:str}, {...}, ...]
    :return: squad_retrieval_test, a list of dicts. [{id:str, question:str, answer_sent:int, context:int, label:int}, {...}, ...]
    :return: sentences: list of sentences, obtained by breaking down the context using the infer_sentence_breaks function
    :return: contexts: list of contexts, obtained by reading the context of each paragraph.
    :return: response: list of [sent_num, context_num]. This reponse list is the final list to generate the embeddings.
    '''

    pickle_save_path_complete = pickle_save_folder_path+'/'+pickle_save_file_path+".pickle"

    # If the retrieval file already exists, load from pcikle.
    if os.path.exists(pickle_save_path_complete):
        with open(pickle_save_path_complete, "rb") as handle:
            nq_retrieval_pickle = pickle.load(handle)

        return nq_retrieval_pickle

    # Otherwise generate from scratch and save to pickle
    else:
        with open(nq_json_train_path, "r") as handle:
            nq_json_train = json.load(handle)

        seen_sentences_dict = {}
        seen_documents_dict = {}
        seen_responses_dict ={}
        sent_doc_resp_count = {"sent":0, "doc":0, "resp":0}


        nq_retrieval_train = []
        for question, q_id, answer_sent, document, para, gold_answer_text in _generate_examples(nq_json_train):
            if document in seen_documents_dict:
                document_id = seen_documents_dict[document]
            else:
                document_id = sent_doc_resp_count["doc"]
                seen_documents_dict[document] = document_id
                sent_doc_resp_count["doc"]+=1

            for sentence in para:
                if sentence in seen_sentences_dict:
                    sentence_id = seen_sentences_dict[sentence]
                else:
                    sentence_id = sent_doc_resp_count["sent"]
                    seen_sentences_dict[sentence] = sentence_id
                    sent_doc_resp_count["sent"] += 1

                response_key = str(sentence_id)+","+str(document_id)
                if response_key not in seen_responses_dict:
                    seen_responses_dict[response_key] = sent_doc_resp_count["resp"]
                    sent_doc_resp_count["resp"]+=1

            assert(answer_sent in seen_sentences_dict)
            answer_sent_id = seen_sentences_dict[answer_sent]
            answer_doc_id = document_id
            response_id = seen_responses_dict[str(answer_sent_id)+","+str(answer_doc_id)]

            question_dict_to_append= {"id":q_id,"question":question,"answer":answer_sent_id,"document":answer_doc_id,"response":response_id, "gold_answer_text":gold_answer_text}
            nq_retrieval_train.append(question_dict_to_append)


        # convert sent dict, doc dict and reponse dict to lists:
        sent_list_raw = [k for k,v in  sorted(seen_sentences_dict.items(), key=lambda x: x[1])]  # sentences in raw text
        doc_list_raw = [k for k,v in  sorted(seen_documents_dict.items(), key=lambda x: x[1])]   # documents in raw text
        response_list = [(int(k.split(",")[0]), int(k.split(",")[1])) for k,v in  sorted(seen_responses_dict.items(), key=lambda x: x[1])]

        # tokenize the sentences and documents beforehand to save time at training.
        # sent_list = []
        # doc_list = []
        # for raw_sent in sent_list_raw:
        #     tokens_ = tokenizer.tokenize(raw_sent)
        #     sent_list.append(tokenizer.convert_tokens_to_ids(tokens_))
        #
        # n_long_docs = 0
        # for raw_doc in doc_list_raw:
        #     tokens_ = tokenizer.tokenize(raw_doc)
        #     if len(tokens_)>512:
        #         n_long_docs+=1
        #     doc_list.append(tokenizer.convert_tokens_to_ids(tokens_))

        # print("data generation finished, splitting train/dev/test using seed", random_seed)
        # print("n long docs / n total docs:", n_long_docs, len(doc_list))
        # random.seed(random_seed)
        # random.shuffle(squad_retrieval_train)
        # squad_retrieval_dev = squad_retrieval_train[-num_dev:]
        # squad_retrieval_train = squad_retrieval_train[:-num_dev]

        if not os.path.exists(pickle_save_folder_path):
            os.mkdir(pickle_save_folder_path)

        with open(pickle_save_path_complete, "wb" ) as handle:
            pickle.dump({"train_list": nq_retrieval_train,
                         "sent_list":sent_list_raw,
                         "doc_list":doc_list_raw,
                         "resp_list":response_list}, handle)

    return {"train_list": nq_retrieval_train,
                         "sent_list":sent_list_raw,
                         "doc_list":doc_list_raw,
                         "resp_list":response_list}

# class PadCollateSQuADTrain:
#     def __init__(self):
#         """
#         Nothing to add here
#         """
#
#     def _pad_tensor(self, vec, pad):
#         return vec + [0] * (pad - len(vec))
#
#     def pad_collate(self, batch):
#         # The input here is actually a list of dictionary.
#         # find longest sequence
#         max_len_query = max([len(sample["query_token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
#         # pad according to max_len
#         for sample in batch:
#             sample["query_token_ids"]  = self._pad_tensor(sample["query_token_ids"], pad=max_len_query)
#             sample["query_att_mask_ids"] = self._pad_tensor(sample["query_att_mask_ids"], pad=max_len_query)
#
#         # stack all
#
#         # the output of this function needs to be a already batched function.
#         batch_returned = {}
#         batch_returned["query_token_ids"] = torch.tensor([sample["query_token_ids"] for sample in batch])
#         batch_returned["query_att_mask_ids"] = torch.tensor([sample["query_att_mask_ids"] for sample in batch])
#         batch_returned["query_seg_ids"] = torch.tensor([[0]*max_len_query for sample in batch])
#
#
#         all_facts_ids = []
#         all_facts_att_mask_ids = []
#
#         for sample in batch:
#             all_facts_ids.extend(sample["fact_token_ids"])
#             all_facts_att_mask_ids.extend(sample["fact_att_mask_ids"])
#
#
#         max_len_fact = max([len(fact_token_ids) for fact_token_ids in all_facts_ids])
#
#         for i, fact_ids in enumerate(all_facts_ids):
#             all_facts_ids[i]  = self._pad_tensor(fact_ids, pad=max_len_fact)  # truncate the facts to maximum 512 tokens.
#         for i, fact_att_mask_ids in enumerate(all_facts_att_mask_ids):
#             all_facts_att_mask_ids[i] = self._pad_tensor(fact_att_mask_ids, pad=max_len_fact)
#
#         # stack all
#
#         # the output of this function needs to be a already batched function.
#         batch_returned["fact_token_ids"] = torch.tensor([fact_ids for fact_ids in all_facts_ids])
#         batch_returned["fact_att_mask_ids"] = torch.tensor([fact_att_mask_ids for fact_att_mask_ids in all_facts_att_mask_ids])
#         batch_returned["fact_seg_ids"] = torch.tensor([[0]*max_len_fact for fact_ids in all_facts_ids])
#
#         batch_returned["label_in_distractor"] = torch.tensor([sample["label_in_distractor"] for sample in batch])
#
#         return batch_returned
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)
#
# class SQuADRetrievalDatasetTrain(Dataset):
#     def __init__(self, instance_list, sent_list, doc_list, resp_list, tokenizer, random_seed, n_neg_sample=10):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.sent_list = sent_list
#         self.doc_list = doc_list
#         self.resp_list = resp_list
#         self.tokenizer = tokenizer
#         self.n_neg_sample = n_neg_sample
#
#         self.instance_list=  []
#         random.seed(random_seed)  # set random seed to make the negative sampling reproducible.
#         for i, instance in enumerate(instance_list):
#             # cls_id = 101; sep_id = 102; pad_id = 0;
#             query_tokens = tokenizer.tokenize(instance["question"])   # this is for strip quotes
#             query_token_ids = [101] + tokenizer.convert_tokens_to_ids(query_tokens) + [102]  # this does not include pad, cls or sep
#
#             instance["query_token_ids"] = query_token_ids
#             instance["query_seg_ids"] = [0] * len(query_token_ids)  # use seg id 0 for query.
#             instance["query_att_mask_ids"] = [1] * len(query_token_ids)
#
#             instance["label_in_distractor"] = 0
#
#         self.instance_list = instance_list
#
#     def _random_negative_from_kb(self, target_fact_num, kb_as_list, num_of_negative_facts):
#         candidate_indexes = random.sample(range(len(kb_as_list)), num_of_negative_facts+1)  # the sample should be already set here.
#         if target_fact_num in candidate_indexes:
#             candidate_indexes.remove(target_fact_num)
#             return candidate_indexes
#
#         else:
#             return candidate_indexes[:num_of_negative_facts]
#
#     def __len__(self):
#         return len(self.instance_list)
#
#     def __getitem__(self, idx):
#         fact_token_ids = []
#         fact_seg_ids = []
#         fact_att_mask_ids = []
#
#         gold_resp_index = self.instance_list[idx]["response"]
#         negative_resp_index = self._random_negative_from_kb(gold_resp_index, self.resp_list, self.n_neg_sample)
#         all_resps = [self.resp_list[idx_] for idx_ in [gold_resp_index] + negative_resp_index]
#         for response_tuple in all_resps:
#             single_fact_token_ids = self.sent_list[response_tuple[0]] + [102] + self.doc_list[response_tuple[1]]
#             single_fact_token_ids = single_fact_token_ids[:min(len(single_fact_token_ids), 254)]
#             single_fact_token_ids = [101] + single_fact_token_ids + [102]
#             fact_token_ids.append(single_fact_token_ids)
#             fact_seg_ids.append([0] * len(single_fact_token_ids))  # use seg id 1 for response.
#             fact_att_mask_ids.append([1] * len(single_fact_token_ids))  # use [1] on non-pad token
#
#         self.instance_list[idx]["fact_token_ids"] = fact_token_ids
#         self.instance_list[idx]["fact_seg_ids"] = fact_seg_ids
#         self.instance_list[idx]["fact_att_mask_ids"] = fact_att_mask_ids
#
#         return self.instance_list[idx]
#
# class PadCollateSQuADEvalQuery:
#     def __init__(self):
#         """
#         Nothing to add here
#         """
#     def _pad_tensor(self, vec, pad):
#         return vec + [0] * (pad - len(vec))
#
#     def pad_collate(self, batch):
#         # The input here is actually a list of dictionary.
#         # find longest sequence
#         max_len_query = max([len(sample["query_token_ids"]) for sample in batch]) # this should be equivalent to "for x in batch"
#         # pad according to max_len
#         for sample in batch:
#             sample["query_token_ids"]  = self._pad_tensor(sample["query_token_ids"], pad=max_len_query)
#             sample["query_att_mask_ids"] = self._pad_tensor(sample["query_att_mask_ids"], pad = max_len_query)
#
#         # stack all
#
#         # the output of this function needs to be a already batched function.
#         batch_returned = {}
#         batch_returned["query_token_ids"] = torch.tensor([sample["query_token_ids"] for sample in batch])
#         batch_returned["query_att_mask_ids"] = torch.tensor([sample["query_att_mask_ids"] for sample in batch])
#         batch_returned["query_seg_ids"] = torch.tensor([[0]*max_len_query for sample in batch])
#         batch_returned["response"] = torch.tensor([sample["response"] for sample in batch])
#
#         return batch_returned
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)
#
# class SQuADRetrievalDatasetEvalQuery(Dataset):
#     def __init__(self, instance_list, sent_list, doc_list, resp_list, tokenizer):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.instance_list=  []
#         for instance in instance_list:
#             # cls_id = 101; sep_id = 102; pad_id = 0;
#             query_tokens = tokenizer.tokenize(instance["question"])   # this is for strip quotes
#             query_token_ids = [101]+tokenizer.convert_tokens_to_ids(query_tokens)+[102]   # this does not include pad, cls or sep
#
#             instance["query_token_ids"] = query_token_ids
#             instance["query_seg_ids"] = [0]*len(query_token_ids)  # use seg id 0 for query.
#             instance["query_att_mask_ids"] = [1] * len(query_token_ids)
#
#         self.instance_list = instance_list
#
#     def __len__(self):
#         return len(self.instance_list)
#
#     def __getitem__(self, idx):
#         return self.instance_list[idx]
#
# class PadCollateSQuADEvalFact:
#     def __init__(self):
#         """
#         Nothing to add here
#         """
#
#     def _pad_tensor(self, vec, pad):
#         return vec + [0] * (pad - len(vec))
#
#     def pad_collate(self, batch):
#         # The input here is actually a list of dictionary.
#         # find longest sequence
#         batch_returned = {}
#         all_facts_ids = []
#         all_facts_att_mask_ids = []
#
#         max_len_fact = max([len(sample["fact_token_ids"]) for sample in batch])
#
#         for sample in batch:
#             all_facts_ids.append(self._pad_tensor(sample["fact_token_ids"], pad=max_len_fact))
#             all_facts_att_mask_ids.append(self._pad_tensor(sample["fact_att_mask_ids"], pad=max_len_fact))
#
#         # stack all
#
#         # the output of this function needs to be a already batched function.
#         batch_returned["fact_token_ids"] = torch.tensor([fact_ids for fact_ids in all_facts_ids])
#         batch_returned["fact_seg_ids"] = torch.tensor([[0]*max_len_fact for fact_ids in all_facts_ids])
#         batch_returned["fact_att_mask_ids"] = torch.tensor([fact_att_mask_ids for fact_att_mask_ids in all_facts_att_mask_ids])
#
#         return batch_returned
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)
#
# class SQuADRetrievalDatasetEvalFact(Dataset):
#     def __init__(self, instance_list, sent_list, doc_list, resp_list, tokenizer):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.instance_list=  []
#         for instance in instance_list:
#             # cls_id = 101; sep_id = 102; pad_id = 0;
#             fact_token_ids = sent_list[instance[0]]+[102]+doc_list[instance[1]]
#             fact_token_ids = fact_token_ids[:min(len(fact_token_ids), 254)]
#             fact_token_ids = [101] + fact_token_ids + [102]
#             fact_seg_ids = [0]*len(fact_token_ids)
#
#             instance_new = {}
#             instance_new["fact_token_ids"] = fact_token_ids
#             instance_new["fact_seg_ids"] = fact_seg_ids
#             instance_new["fact_att_mask_ids"] = [1] * len(fact_token_ids)
#
#             self.instance_list.append(instance_new)
#
#     def __len__(self):
#         return len(self.instance_list)
#
#     def __getitem__(self, idx):
#         return self.instance_list[idx]
#
def check_nq_retrieval_pickle(saved_pickle_path = parent_folder_path+"/data_generated/nq_retrieval_raw/nq_retrieval_data.pickle"):
    with open(saved_pickle_path, "rb") as handle:
        nq_retrieval_data = pickle.load(handle)

    print("total number of facts:", len(nq_retrieval_data["sent_list"]))
    print("total number of paragraphs:", len(nq_retrieval_data["doc_list"]))
    print("total number of examples:", len(nq_retrieval_data["train_list"]))

    for data_partition in ["train_list"]:
        non_standard_example = 0
        print("="*20+"\n"+"checking data partition "+data_partition)
        print("number of samples in this partition:", len(nq_retrieval_data[data_partition]))

        samples_to_check = random.sample(list(range(len(nq_retrieval_data[data_partition]))), 10)
        for i, question_dict in enumerate(nq_retrieval_data[data_partition]):
            try:
                assert (question_dict["gold_answer_text"] in nq_retrieval_data["sent_list"][question_dict["answer"]])
                assert (nq_retrieval_data["sent_list"][question_dict["answer"]] in nq_retrieval_data["doc_list"][question_dict["document"]])
            except:
                non_standard_example+=1

            assert(nq_retrieval_data["resp_list"][question_dict["response"]][0]==question_dict["answer"])
            assert(nq_retrieval_data["resp_list"][question_dict["response"]][1]==question_dict["document"])

            if i in samples_to_check:
                print("-"*20)
                print("\tsample index:", i)
                print("\tquestion:", question_dict["question"])
                print("\tanswer sent reconstruct:", nq_retrieval_data["sent_list"][
                    nq_retrieval_data["resp_list"][question_dict["response"]][0]])
                print("\tanswer doc reconstruct:", nq_retrieval_data["doc_list"][
                    nq_retrieval_data["resp_list"][question_dict["response"]][1]])
                print("\tgold answer:", question_dict["gold_answer_text"])

    print("total number of broken sentences:", non_standard_example)

    return 0

def generate_data_for_lucene():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    nq_retrieval_data =  convert_nq_to_retrieval()


    # TODO: later generate dev queries and labels for different seeds.



    print(len(nq_retrieval_data["resp_list"]))
    print(len(nq_retrieval_data["train_list"]))
    with open('nq_test_query.txt', 'a') as the_file:
        for instance in nq_retrieval_data["train_list"]:
            the_file.write(instance["question"]+" QUERY_SEP\n ")

    with open('nq_test_label.txt', 'a') as the_file:
        for instance in nq_retrieval_data["train_list"]:
            the_file.write(str(instance["response"])+"\n")

    with open('nq_test_id.txt', 'a') as the_file:
        for instance in nq_retrieval_data["train_list"]:
            the_file.write(str(instance["id"])+"\n")

    with open('nq_kb.txt', 'a') as the_file:
        for resp in nq_retrieval_data["resp_list"]:
            the_file.write(nq_retrieval_data["sent_list"][resp[0]]+" "+ nq_retrieval_data["doc_list"][resp[1]] +" DOC_SEP\n ")

    return 0

generate_data_for_lucene()

# def check_squad_dataloader(saved_pickle_path = parent_folder_path+"/data_generated/squad_retrieval_data_seed_0_dev_2000.pickle"):
#     with open(saved_pickle_path, "rb") as handle:
#         squad_retrieval_data = pickle.load(handle)
#
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     batch_size = 2
#     n_neg_fact = 3
#
#     check_train = True
#     check_dev = True
#     check_test = True
#     check_fact = True
#
#     if check_train:
#         # examine dataloader train
#         print("="*20+"\nChecking squad train")
#         squad_retrieval_train_dataset = SQuADRetrievalDatasetTrain(instance_list=squad_retrieval_data["train_list"],
#                                                                    sent_list=squad_retrieval_data["sent_list"],
#                                                                    doc_list=squad_retrieval_data["doc_list"],
#                                                                    resp_list=squad_retrieval_data["resp_list"],
#                                                                    tokenizer = tokenizer,
#                                                                    n_neg_sample=n_neg_fact,
#                                                                    random_seed=0)
#
#         squad_retrieval_train_dataloader = DataLoader(squad_retrieval_train_dataset, batch_size=batch_size,
#                                         shuffle=True, num_workers=1, collate_fn=PadCollateSQuADTrain())
#
#         samples_to_check = random.sample(list(range(len(squad_retrieval_train_dataloader))), 3)
#
#
#         for i, batch in enumerate(squad_retrieval_train_dataloader):
#
#             if i in samples_to_check:
#                 print(""+"-"*20)
#                 print("batch size:", batch_size)
#                 print("query token id shape:", batch["query_token_ids"].size())
#                 print("query seg id shape:", batch["query_seg_ids"].size())
#                 print("doc token id shape:", batch["fact_token_ids"].size())
#                 print("doc seg id shape:", batch["fact_seg_ids"].size())
#
#                 print("\n")
#
#                 for j in range(batch["query_token_ids"].size()[0]):
#                     print("query ids:", batch["query_token_ids"][j])
#                     print("query seg ids:", batch["query_seg_ids"][j])
#                     print("query att mask ids:", batch["query_att_mask_ids"][j])
#                     print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
#                     print("\n")
#
#                     for k in range(n_neg_fact + 1):
#                         print("\tfact ids:", batch["fact_token_ids"][j * (n_neg_fact + 1) + k])
#                         print("\tfact seg ids:", batch["fact_seg_ids"][j * (n_neg_fact + 1) + k])
#                         print("\tfact att mask ids:", batch["fact_att_mask_ids"][j * (n_neg_fact + 1) + k])
#                         print("\tfact tokens:", tokenizer.convert_ids_to_tokens(
#                             batch["fact_token_ids"][j * (n_neg_fact + 1) + k].tolist()))
#                         print("\n")
#
#                 input("AAA")
#
#         del(squad_retrieval_train_dataset)
#         del(squad_retrieval_train_dataloader)
#         input("\nwait for the next check\n")
#
#
#     if check_dev:
#         # examine dataloader dev query
#         print("=" * 20 + "\nChecking squad dev")
#         squad_retrieval_dev_dataset = SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["dev_list"],
#                                                                    sent_list=squad_retrieval_data["sent_list"],
#                                                                    doc_list=squad_retrieval_data["doc_list"],
#                                                                    resp_list=squad_retrieval_data["resp_list"],
#                                                                    tokenizer=tokenizer)
#
#         squad_retrieval_dev_dataloader = DataLoader(squad_retrieval_dev_dataset, batch_size=batch_size,
#                                                       shuffle=False, num_workers=1, collate_fn=PadCollateSQuADEvalQuery())
#
#         samples_to_check = random.sample(list(range(len(squad_retrieval_dev_dataloader))), 3)
#
#         for i, batch in enumerate(squad_retrieval_dev_dataloader):
#
#             if i in samples_to_check:
#                 print("\t" + "-" * 20)
#                 print("\tbatch size:", batch_size)
#                 print("\tquery token id shape:", batch["query_token_ids"].size())
#                 print("\tquery seg id shape:", batch["query_seg_ids"].size())
#
#                 print("\n")
#
#                 for j in range(batch["query_token_ids"].size()[0]):
#                     print("query ids:", batch["query_token_ids"][j])
#                     print("query seg ids:", batch["query_seg_ids"][j])
#                     print("query att mask ids:", batch["query_att_mask_ids"][j])
#                     print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
#                     print("\n")
#
#                 input("AAA")
#
#         del (squad_retrieval_dev_dataset)
#         del (squad_retrieval_dev_dataloader)
#         input("\nwait for the next check\n")
#
#     if check_test:
#         # examine dataloader test query
#         print("=" * 20 + "\nChecking squad test")
#         squad_retrieval_test_dataset = SQuADRetrievalDatasetEvalQuery(instance_list=squad_retrieval_data["test_list"],
#                                                                  sent_list=squad_retrieval_data["sent_list"],
#                                                                  doc_list=squad_retrieval_data["doc_list"],
#                                                                  resp_list=squad_retrieval_data["resp_list"],
#                                                                  tokenizer=tokenizer)
#
#         squad_retrieval_test_dataloader = DataLoader(squad_retrieval_test_dataset, batch_size=batch_size,
#                                                     shuffle=False, num_workers=1, collate_fn=PadCollateSQuADEvalQuery())
#
#         samples_to_check = random.sample(list(range(len(squad_retrieval_test_dataloader))), 3)
#
#         for i, batch in enumerate(squad_retrieval_test_dataloader):
#
#             if i in samples_to_check:
#                 print("\t" + "-" * 20)
#                 print("\tbatch size:", batch_size)
#                 print("\tquery token id shape:", batch["query_token_ids"].size())
#                 print("\tquery seg id shape:", batch["query_seg_ids"].size())
#
#                 print("\n")
#
#                 for j in range(batch["query_token_ids"].size()[0]):
#                     print("query ids:", batch["query_token_ids"][j])
#                     print("query seg ids:", batch["query_seg_ids"][j])
#                     print("query att mask ids:", batch["query_att_mask_ids"][j])
#                     print("query tokens:", tokenizer.convert_ids_to_tokens(batch["query_token_ids"][j].tolist()))
#                     print("\n")
#
#                 input("AAA")
#
#         del (squad_retrieval_test_dataset)
#         del (squad_retrieval_test_dataloader)
#         input("\nwait for the next check\n")
#
#     if check_fact:
#         # examine dataloader all facts:
#         print("=" * 20 + "\nChecking squad eval fact")
#         squad_retrieval_eval_fact_dataset = SQuADRetrievalDatasetEvalFact(instance_list=squad_retrieval_data["resp_list"],
#                                                                    sent_list=squad_retrieval_data["sent_list"],
#                                                                    doc_list=squad_retrieval_data["doc_list"],
#                                                                    resp_list=squad_retrieval_data["resp_list"],
#                                                                    tokenizer=tokenizer)
#
#         squad_retrieval_eval_fact_dataloader = DataLoader(squad_retrieval_eval_fact_dataset, batch_size=batch_size,
#                                                       shuffle=False, num_workers=1, collate_fn=PadCollateSQuADEvalFact())
#
#         samples_to_check = random.sample(list(range(len(squad_retrieval_eval_fact_dataloader))), 3)
#
#         for i, batch in enumerate(squad_retrieval_eval_fact_dataloader):
#
#             if i in samples_to_check:
#                 print("\t" + "-" * 20)
#                 print("\tbatch size:", batch_size)
#                 print("\tdoc token id shape:", batch["fact_token_ids"].size())
#                 print("\tdoc seg id shape:", batch["fact_seg_ids"].size())
#
#                 print("\n")
#
#                 for j in range(batch["fact_token_ids"].size()[0]):
#                     print("fact ids:", batch["fact_token_ids"][j])
#                     print("fact seg ids:", batch["fact_seg_ids"][j])
#                     print("fact att mask ids:", batch["fact_att_mask_ids"][j])
#                     print("fact tokens:", tokenizer.convert_ids_to_tokens(batch["fact_token_ids"][j].tolist()))
#                     print("\n")
#
#                 input("AAA")
#
#     return 0


#convert_nq_to_retrieval()
#check_nq_retrieval_pickle()