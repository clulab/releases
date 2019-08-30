import csv,sys
from random import shuffle

from typing import Dict
import json,os
import logging

from overrides import overrides
import tqdm
import sys
from tqdm import tqdm as tq
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from common.dataset.reader import JSONLineReader
from common.util.random import SimpleRandom
from retrieval.fever_doc_db import FeverDocDB
from retrieval.read_claims import UOFADataReader
from rte.riedel.data import FEVERPredictions2Formatter, FEVERLabelSchema, FEVERGoldFormatter
from common.dataset.data_set import DataSet as FEVERDataSet
from rte.mithun.trainer import UofaTrainTest

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("fever")
class FEVERReader(DatasetReader):
    """
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis".

    Parameters
    ----------   
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 db: FeverDocDB,
                 sentence_level = False,
                 wiki_tokenizer: Tokenizer = None,
                 claim_tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 filtering: str = None) -> None:
        self._sentence_level = sentence_level
        self._wiki_tokenizer = wiki_tokenizer or WordTokenizer()
        self._claim_tokenizer = claim_tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.db = db

        self.formatter = FEVERGoldFormatter(set(self.db.get_doc_ids()), FEVERLabelSchema(),filtering=filtering)
        self.reader = JSONLineReader()


    def get_doc_line(self,doc,line):
        lines = self.db.get_doc_lines(doc)
        if line > -1:
            return lines.split("\n")[line].split("\t")[1]
        else:
            non_empty_lines = [line.split("\t")[1] for line in lines.split("\n") if len(line.split("\t"))>1 and len(line.split("\t")[1].strip())]
            return non_empty_lines[SimpleRandom.get_instance().next_rand(0,len(non_empty_lines)-1)]

    def annotation_on_the_fly(self, file_path, run_name, objUOFADataReader):
        print("do_annotation_on_the_fly == true")

        # DELETE THE annotated file IF IT EXISTS every time before the loop
        # self.delete_if_exists(head_file)
        # self.delete_if_exists(body_file)
        if (run_name == "train"):
            print("run_name == train")
            head_file = objUOFADataReader.ann_head_tr
            body_file = objUOFADataReader.ann_body_tr
        else:
            if (run_name == "dev"):
                print("run_name == dev")
                head_file = objUOFADataReader.ann_head_dev
                body_file = objUOFADataReader.ann_body_dev

        ds = FEVERDataSet(file_path, reader=self.reader, formatter=self.formatter)
        ds.read()
        instances = []


        for instance in tqdm.tqdm(ds.data):
            counter = counter + 1

            if instance is None:
                continue

            if not self._sentence_level:
                pages = set(ev[0] for ev in instance["evidence"])
                premise = " ".join([self.db.get_doc_text(p) for p in pages])
            else:
                lines = set([self.get_doc_line(d[0], d[1]) for d in instance['evidence']])
                premise = " ".join(lines)

            if len(premise.strip()) == 0:
                premise = ""

            hypothesis = instance["claim"]
            label = instance["label_text"]

            premise_ann, hypothesis_ann = self.uofa_annotate(hypothesis, premise, counter, objUOFADataReader, head_file,
                                                             body_file)
            instances.append(self.text_to_instance(premise_ann, hypothesis_ann, label))
        return instances

    @overrides
    def read(self, mithun_logger,data_folder):
        mithun_logger.info("got inside read")
        mithun_logger.info("got inside read")

        nei_overlap_counter = 0
        nei_counter = 0
        supports_overlap_counter = 0
        supports_counter = 0
        refutes_overlap_counter = 0
        refutes_counter = 0

        instances = []

        counter=0

        objUOFADataReader = UOFADataReader()

        # do annotation on the fly  using pyprocessors. i.e creating NER tags, POS Tags etc.
        # This takes along time. so almost always we do it only once, and load it from disk
        if(do_annotation_on_the_fly):
            instances = self.annotation_on_the_fly(file_path, run_name, objUOFADataReader)

        # replacing hypothesis with the annotated one-either load pre-annotated data
        # from disk
        #else:

        print("(do_annotation=false):going to load annotated data from the disk. ")

        #
        objUofaTrainTest = UofaTrainTest()

        # folders = {"dev": objUofaTrainTest.data_folder_dev, "train": objUofaTrainTest.data_folder_train,
        #            "test": objUofaTrainTest.data_folder_test,  "small": objUofaTrainTest. data_folder_train_small100}
        #
        # data_folder = folders[run_name]
        # print(f"Run name: {run_name}")
        mithun_logger.debug(f"data_folder: {data_folder}")

        bf = data_folder + objUofaTrainTest.annotated_body_split_folder
        bfl = bf + objUofaTrainTest.annotated_only_lemmas
        bfw = bf + objUofaTrainTest.annotated_words
        bfe = bf + objUofaTrainTest.annotated_only_entities

        hf = data_folder + objUofaTrainTest.annotated_head_split_folder
        hft = hf + objUofaTrainTest.annotated_only_tags
        hfd= hf + objUofaTrainTest.annotated_only_dep
        hfl = hf + objUofaTrainTest.annotated_only_lemmas
        hfw = hf + objUofaTrainTest.annotated_words
        hfe = hf + objUofaTrainTest.annotated_only_entities
        hfcomplete = hf + objUofaTrainTest.annotated_whole_data_head

        #print(f"hfl:{hfl}")
        #print(f"bfl:{bfl}")
        #print("going to read annotated data from disk:")

        heads_lemmas = objUofaTrainTest.read_json(hfl)
        bodies_lemmas = objUofaTrainTest.read_json(bfl)
        heads_entities = objUofaTrainTest.read_json(hfe)
        bodies_entities = objUofaTrainTest.read_json(bfe)
        heads_words = objUofaTrainTest.read_json(hfw)
        bodies_words = objUofaTrainTest.read_json(bfw)
        heads_tags= objUofaTrainTest.read_json(hft)
        heads_deps = objUofaTrainTest.read_json_deps(hfd)
        heads_complete_annotation=objUofaTrainTest.read_id_field_json(hfcomplete)

        print(f"length of bodies_words:{len(bodies_words)}")

        counter=0


        for he, be, hl, bl, hw, bw,ht,hd,hfc in\
                tq(zip(heads_entities, bodies_entities, heads_lemmas,
                                                    bodies_lemmas,
                                                      heads_words,
                                                      bodies_words,heads_tags,heads_deps,heads_complete_annotation),
                   total=len(heads_complete_annotation),desc="reading annotated data"):

            counter=counter+1



            he_split=  he.split(" ")
            be_split = be.split(" ")
            hl_split = hl.split(" ")
            bl_split = bl.split(" ")
            hw_split = hw.split(" ")
            bw_split = bw.split(" ")

            # hypothesis == = claim = headline
            # premise == = evidence = body


            #premise_ann, hypothesis_ann,found_intersection = objUofaTrainTest.convert_SMARTNER_form_per_sent(he_split, be_split, hl_split, bl_split, hw_split, bw_split)

            premise_ann, hypothesis_ann = objUofaTrainTest.convert_NER_form_per_sent_plain_NER(he_split, be_split,hl_split, bl_split,hw_split, bw_split)

            # print(f"hypothesis before annotation: {hw}")
            # print(f"premise before annotation: {bw}")
            #
            # print("value of the first premise and hypothesis after  ner replacement is")
            # print(premise_ann)
            # print(hypothesis_ann)
            # sys.exit(1)
            #





            label=str(hfc)


            # # This is for the analysis of the NEI over-predicting
            # if(label=="NOT ENOUGH INFO"):
            #     nei_counter=nei_counter+1
            #     if(found_intersection):
            #
            #         # print("\n")
            #         # print(f"hw: {hw}")
            #         # print(f"bw: {bw}")
            #         # print(f"hypothesis_ann: {hypothesis_ann}")
            #         # print(f"premise_ann: {premise_ann}")
            #         #
            #         # print(f"label: {label}")
            #
            #         nei_overlap_counter=nei_overlap_counter+1
            #
            # if (label == "SUPPORTS"):
            #     supports_counter = supports_counter + 1
            #     if (found_intersection):
            #         supports_overlap_counter=supports_overlap_counter+1
            #
            # if (label == "REFUTES"):
            #     refutes_counter = refutes_counter + 1
            #     if (found_intersection):
            #         refutes_overlap_counter = refutes_overlap_counter + 1



            instances.append(self.text_to_instance(premise_ann.lower(), hypothesis_ann.lower(), label))


        print(f"after reading and converting training data to  ner format. The length of the number of training data is:{len(instances)}")

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        # print(f"nei_overlap_counter: {nei_counter}")
        # print(f"nei_overlap_counter: {nei_overlap_counter}")
        # print(f"supports_counter: {supports_counter}")
        # print(f"supports_overlap_counter: {supports_overlap_counter}")
        # print(f"refutes_counter: {refutes_counter}")
        # print(f"refutes_overlap_counter: {refutes_overlap_counter}")




        return Dataset(instances)

    def read_annotated_fnc_and_do_ner_replacement(self, args, run_name, do_annotation_on_the_fly,path_to_fnc_annotated_data):
        nei_overlap_counter = 0
        nei_counter = 0
        supports_overlap_counter = 0
        supports_counter = 0
        refutes_overlap_counter = 0
        refutes_counter = 0
        instances = []
        #
        # ds = FEVERDataSet(file_path,reader=self.reader, formatter=self.formatter)
        # ds.read()


        mithun_logger.info("(do_annotation=false):going to load annotated data from the disk.")


        objUofaTrainTest = UofaTrainTest()

        params = Params.from_file(args.param_path, args.overrides)
        uofa_params = params.pop('uofa_params', {})

        data_folder = objUofaTrainTest.data_root + str(path_to_fnc_annotated_data)



        mithun_logger.debug(f"value of data_folder is {data_folder}")
        mithun_logger.debug(f"value of use_plain_NER is {use_plain_NER}")


        #load the labels from the disk
        lbl_file= objUofaTrainTest.label_folder+objUofaTrainTest.label_dev_file
        all_labels= objUofaTrainTest.read_csv_list(lbl_file)



        bf = data_folder + objUofaTrainTest.annotated_body_split_folder
        bfl = bf + objUofaTrainTest.annotated_only_lemmas

        bf = data_folder + objUofaTrainTest.annotated_body_split_folder
        bfl = bf + objUofaTrainTest.annotated_only_lemmas
        bfw = bf + objUofaTrainTest.annotated_words
        bfe = bf + objUofaTrainTest.annotated_only_entities

        hf = data_folder + objUofaTrainTest.annotated_head_split_folder
        hfl = hf + objUofaTrainTest.annotated_only_lemmas
        hfw = hf + objUofaTrainTest.annotated_words
        hfe = hf + objUofaTrainTest.annotated_only_entities




        #print(f"hfl:{hfl}")
        #print(f"bfl:{bfl}")
        #print("going to read annotated data from disk:")



        heads_lemmas = objUofaTrainTest.read_json(hfl)
        bodies_lemmas = objUofaTrainTest.read_json(bfl)
        heads_entities = objUofaTrainTest.read_json(hfe)
        bodies_entities = objUofaTrainTest.read_json(bfe)
        heads_words = objUofaTrainTest.read_json(hfw)
        bodies_words = objUofaTrainTest.read_json(bfw)

        mithun_logger.debug(f"length of headline_words:{len(heads_words)}")
        mithun_logger.debug(f"length of bodies_words:{len(bodies_words)}")
        mithun_logger.debug(f"length of all_labels:{len(all_labels)}")



        counter=0
        #h stands for headline and b for body
        for he, be, hl, bl, hw, bw,indiv_label in\
                tq(zip(heads_entities, bodies_entities, heads_lemmas,
                                                    bodies_lemmas,
                                                      heads_words,
                                                      bodies_words,all_labels),
                   total=len(all_labels),desc="reading annotated data"):

            counter=counter+1
            label = indiv_label



            if not (label == "unrelated"):


                if (label == 'discuss'):
                    new_label = "NOT ENOUGH INFO"
                if (label == 'agree'):
                    new_label = "SUPPORTS"
                if (label == 'disagree'):
                    new_label = "REFUTES"


                he_split=  he.split(" ")
                be_split = be.split(" ")
                hl_split = hl.split(" ")
                bl_split = bl.split(" ")
                hw_split = hw.split(" ")
                bw_split = bw.split(" ")

                # note that these words are equivalent
                # hypothesis == = claim = headline
                # premise == = evidence = body
                #


                # premise_ann=bw
                # hypothesis_ann=hw

                # print(f"hypothesis_before_annotation: {hw}")
                # print(f"premise_before_annotation: {bw}")
                if(use_plain_NER):
                    premise_ann, hypothesis_ann = objUofaTrainTest.convert_NER_form_per_sent_plain_NER(he_split, be_split,hl_split, bl_split,hw_split, bw_split)
                else:
                    premise_ann, hypothesis_ann,found_intersection = objUofaTrainTest.convert_SMARTNER_form_per_sent(he_split, be_split, hl_split, bl_split, hw_split, bw_split)


                #if (found_intersection):
                print("\n")
                print(f"hypothesis_before_annotation: {hw}")
                print(f"premise_before_annotation: {bw}")
                print(f"hypothesis_ann: {hypothesis_ann}")
                print(f"premise_ann: {premise_ann}")
                print(f"label: {label}")
                sys.exit(1)

                # This is for the analysis of the NEI over-predicting
                if(new_label == "NOT ENOUGH INFO"):
                    nei_counter = nei_counter + 1
                    if (found_intersection):
                        nei_overlap_counter = nei_overlap_counter + 1


                if (new_label == "SUPPORTS"):
                    supports_counter = supports_counter + 1
                    if (found_intersection):
                        supports_overlap_counter = supports_overlap_counter + 1

                if (new_label == "REFUTES"):
                    refutes_counter = refutes_counter + 1
                    if (found_intersection):
                        refutes_overlap_counter = refutes_overlap_counter + 1


                instances.append(self.text_to_instance(bw, hw, new_label))


        print(f"after reading and converting training data to smart ner format. The length of the number of training data is:{len(instances)}")

        # print(f"nei_overlap_counter: {nei_counter}")
        # print(f"nei_overlap_counter: {nei_overlap_counter}")
        # print(f"supports_counter: {supports_counter}")
        # print(f"supports_overlap_counter: {supports_overlap_counter}")
        # print(f"refutes_counter: {refutes_counter}")
        # print(f"refutes_overlap_counter: {refutes_overlap_counter}")
        # sys.exit(1)

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        return Dataset(instances)


    def read_fnc(self, d):
        instances = []

        for s in tqdm.tqdm(d.stances):

            headline = s['Headline']
            bodyid = s['Body ID']
            actualBody = d.articles[bodyid]
            label = s['Stance']


            if not (label == "unrelated"):

                if(label=='discuss'):
                    new_label="NOT ENOUGH INFO"
                if (label == 'agree'):
                    new_label = "SUPPORTS"
                if (label == 'disagree'):
                        new_label = "REFUTES"

                hypothesis =headline
                premise = actualBody
                instances.append(self.text_to_instance(premise, hypothesis, new_label))

                # print(new_label)
                # print(premise)
                # print(hypothesis)
                # sys.exit(1)



        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?")
        return Dataset(instances)


    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._wiki_tokenizer.tokenize(premise) if premise is not None else None
        hypothesis_tokens = self._claim_tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers) if premise is not None else None
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)



    def uofa_load_ann_disk(self,objUOFADataReader,run_name):


        # print(f'premise:{premise}')
        # print(f'hyp:{hyp}')
        # sys.exit(1)



        # print(premise,hyp)
        return premise, hyp

    def uofa_annotate(self, claim, evidence, index,objUOFADataReader,head_file,body_file):
        doc1,doc2 = objUOFADataReader.annotate_and_save_doc\
            (claim, evidence, index, objUOFADataReader.API,head_file,body_file,logger)

        he=doc1._entities
        hl=doc1.lemmas
        hw=doc1.words
        be = doc2._entities
        bl = doc2.lemmas
        bw = doc2.words
        objUofaTrainTest=UofaTrainTest()
        # print(f'{he}{hl}{hw}{be}{bl}{bw}')
        #premise, hyp= objUofaTrainTest.convert_SMARTNER_form_per_sent(he, be, hl, bl, hw, bw)
        premise, hyp = objUofaTrainTest.convert_NER_form_per_sent_plain_NER(he, be, hl, bl, hw, bw)


        # print(premise,hyp)
        return premise,hyp

    def delete_if_exists(self, name):

        if os.path.exists(name):
            append_write = 'w'  # make a new file if not
            with open(name, append_write) as outfile:
                outfile.write("")

    @classmethod
    def from_params(cls, params: Params) -> 'FEVERReader':
        claim_tokenizer = Tokenizer.from_params(params.pop('claim_tokenizer', {}))
        wiki_tokenizer = Tokenizer.from_params(params.pop('wiki_tokenizer', {}))

        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        sentence_level = params.pop("sentence_level",False)
        db = FeverDocDB(params.pop("db_path","data/fever.db"))
        params.assert_empty(cls.__name__)
        return FEVERReader(db=db,
                           sentence_level=sentence_level,
                           claim_tokenizer=claim_tokenizer,
                           wiki_tokenizer=wiki_tokenizer,
                           token_indexers=token_indexers)

