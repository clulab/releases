# ### The Dataset

import json
from torch.utils.data import Dataset, DataLoader
from  .vectorizer_with_embedding import VectorizerWithEmbedding
import pandas as pd
import random
from mean_teacher.utils.utils_valpola import export
import os
from tqdm import tqdm

# @export
# def fever():
#
#     if RTEDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
#         addNoise = data.RandomPatternWordNoise(RTEDataset.NUM_WORDS_TO_REPLACE, RTEDataset.OOV, RTEDataset.WORD_NOISE_TYPE)
#     else:
#         assert False, "Unknown type of noise {}".format(RTEDataset.WORD_NOISE_TYPE)
#
#     return {
#         'train_transformation': None,
#         'eval_transformation': None,
#     }

class RTEDataset(Dataset):
    def __init__(self, claims_evidences_df, vectorizer):
        """
        Args:
            claims_evidences_df (pandas.DataFrame): the dataset
            vectorizer (VectorizerWithEmbedding): vectorizer instantiated from dataset
        """
        self.claims_ev_df = claims_evidences_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_claim_length = max(map(measure_len, claims_evidences_df.claim)) + 2
        self._max_evidence_length = max(map(measure_len, claims_evidences_df.evidence)) + 2

        self.train_df = self.claims_ev_df[self.claims_ev_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.claims_ev_df[self.claims_ev_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.claims_ev_df[self.claims_ev_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        self._labels=self.train_df.label



    @classmethod
    def truncate_words(cls,sent, tr_len):
        sent_split = sent.split(" ")
        if (len(sent_split) > tr_len):
            sent_tr = sent_split[:1000]
            sent_output = " ".join(sent_tr)
            return sent_output
        else:
            return sent

    @classmethod
    def truncate_data(cls,data_dataframe, tr_len):
        '''
        #the data has lots of junk values. Truncate/cut short evidence/claim sentneces if they are more than tr_length
        :param data_dataframe:
        :param tr_len:
        :param args:
        :return: modified pandas dataframe
        '''
        for i, row in data_dataframe.iterrows():
            row.claim= cls.truncate_words(row.claim, tr_len)
            row.evidence = cls.truncate_words(row.evidence, tr_len)
        return data_dataframe

    @classmethod
    def remove_mnli_dash_labels(self,args,df):
        """
        #in mnli some labels are tagged as -. drop them.
        :param args:
        :param df: modified dataframe
        :return:
        """
        if ("mnli" in args.database_to_train_with) or ("mnli" in args.database_to_test_with):
            if("-" in df.values):
                df = df.set_index("label")
                df= df.drop("-", axis=0)
                df = df.reset_index()
        assert df is not None
        return df


    @classmethod
    def load_dataset(cls, train_file, dev_file, test_file, args):
        """Load dataset and make a new vectorizer from scratch

        Args:
            args (str): all arguments which were create initially.
        Returns:
            an instance of ReviewDataset
        """

        assert os.path.exists(train_file) is True
        assert os.path.exists(dev_file) is True
        assert os.path.exists(test_file) is True

        assert os.path.isfile(train_file) is True
        assert os.path.isfile(dev_file) is True
        assert os.path.isfile(test_file) is True


        train_df = pd.read_json(train_file, lines=True)
        train_df=cls.truncate_data(train_df, args.truncate_words_length)
        train_df['split'] = "train"

        dev_df = pd.read_json(dev_file, lines=True)
        dev_df=cls.remove_mnli_dash_labels(args,dev_df)
        dev_df = cls.truncate_data(dev_df, args.truncate_words_length)
        dev_df['split'] = "val"

        test_df = pd.read_json(test_file, lines=True)
        test_df = cls.remove_mnli_dash_labels(args, test_df)
        test_df = cls.truncate_data(test_df, args.truncate_words_length)
        test_df['split'] = "test"



        frames = [train_df, dev_df,test_df]
        combined_train_dev_test_with_split_column_df = pd.concat(frames)
        cls.labels=train_df.label


        return combined_train_dev_test_with_split_column_df,train_df

    @classmethod
    def create_vocabulary(cls, train_file, dev_file, test_file, args):
        """Load dataset and make a new vectorizer from scratch

        Args:
            args (str): all arguments which were create initially.
        Returns:
            an instance of ReviewDataset
        """
        combined_train_dev_test_with_split_column_df,fever_lex_train_df=cls.load_dataset(train_file, dev_file, test_file, args)

        return cls(combined_train_dev_test_with_split_column_df,
                   VectorizerWithEmbedding.create_vocabulary(fever_lex_train_df, args.frequency_cutoff))
    @classmethod
    def load_dataset_and_load_vectorizer(cls, train_input_file, dev_input_file, test_input_file, args):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            input_file (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of ReviewDataset
        """
         #pd.read_json(input_file, lines=True)

        review_df,fever_lex_train_df = cls.load_dataset(train_input_file, dev_input_file, test_input_file,args)
        vectorizer = cls.load_vectorizer_only(args.vectorizer_file)
        return cls(review_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return VectorizerWithEmbedding.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        self._labels=self._target_df.label


    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        claim_vector = \
            self._vectorizer.vectorize(row.claim,self._max_claim_length)

        evidence_vector = \
            self._vectorizer.vectorize(row.evidence, self._max_evidence_length)

        label_index = \
            self._vectorizer.label_vocab.lookup_token(row.label)

        return {'x_claim': claim_vector,
                'x_evidence': evidence_vector,
                'y_target': label_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

    def get_labels(self):
         return self._labels

    def get_all_label_indices(self,dataset):

        #this command returns all the labels and its corresponding indices eg:[198,2]
        all_labels = list(enumerate(dataset.get_labels()))

        #note that even though the labels are shuffled up, we are keeping track/returning only the shuffled indices. so it all works out fine.
        random.shuffle(all_labels)

        #get all the indices alone
        all_indices=[]
        for idx,_  in all_labels:
            all_indices.append(idx)
        return all_indices

