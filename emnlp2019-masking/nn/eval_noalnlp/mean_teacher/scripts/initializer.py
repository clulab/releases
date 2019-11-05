from argparse import Namespace
import torch
import os
import argparse
import argparse
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.utils.logger import LOG

class Initializer():
    def __init__(self):
        self._args=Namespace()

    def set_default_parameters(self):

        args = Namespace(



            # Data and Path information
            frequency_cutoff=5,
            best_model_file_name='best_model',
            # for laptop
            data_dir_local='data',
            fever_train_local_lex='rte/fever/train/fever_train_lex_4labels.jsonl',
            fever_dev_local='rte/fever/dev/fever_dev_split_fourlabels.jsonl',
            fever_test_local='rte/fever/test/fever_test_lex_fourlabels.jsonl',
            fnc_test_local="rte/fnc/test/fn_test_split_fourlabels.jsonl",
            fever_train_local_delex='rte/fever/train/fever_train_delex_oaner_4labels.jsonl',
            fever_dev_local_delex='rte/fever/dev/fever_dev_delex_oaner_4labels.jsonl',
            mnli_train_lex='rte/mnli/train/mnli_train.jsonl',
            mnli_matched_dev_lex='rte/mnli/dev/mnli_dev.jsonl',
            mnli_mismatched_test_lex='rte/mnli/test/mu_mismatched_lex_test.jsonl',
            mednli_test_lex='rte/mednli/test/mednli_test_lex.jsonl',
            mednli_dev='rte/mednli/dev/mednli_dev.jsonl',




            save_dir='model_storage/',
            vectorizer_file='vectorizer.json',
            glove_filepath='glove/glove.840B.300d.txt',


            # Training hyper parameters
            batch_size=32,
            early_stopping_criteria=5,
            learning_rate=0.005,
            num_epochs=500,
            seed=256,
            random_seed=20,
            weight_decay=5e-5,
            Adagrad_init=0,

            # Runtime options
            expand_filepaths_to_save_dir=True,
            load_vectorizer=False,
            load_model_from_disk=False,
            max_grad_norm=5,



            truncate_words_length=1000,
            embedding_size=300,
            optimizer="adagrad",
            para_init=0.01,
            hidden_sz=200,
            arch='decomp_attention',
            pretrained="false",
            update_pretrained_wordemb=False,
            cuda=True,
            workers=0,

            use_gpu=True
        )
        args.use_glove = True
        if args.expand_filepaths_to_save_dir:
            args.vectorizer_file = os.path.join(args.save_dir,
                                                args.vectorizer_file)

            args.model_state_file = os.path.join(args.save_dir,
                                                 args.best_model_file_name)

            print("Expanded filepaths: ")
            print("\t{}".format(args.vectorizer_file))
            print("\t{}".format(args.model_state_file))

        # Check CUDA
        if not torch.cuda.is_available():
            args.cuda = False

        print("Using CUDA: {}".format(args.cuda))

        args.device = torch.device("cuda" if args.cuda else "cpu")

        handle_dirs(args.save_dir)
        self._args=args





    def parse_commandline_args(self):
        parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
        parser.add_argument('--run_type', default="train", type=str,
                            help='type of run. options are: train (which includes val validation also),val, test')
        parser.add_argument('--database_to_train_with', default="fever", type=str,
                            help='')
        parser.add_argument('--database_to_test_with', default="fnc", type=str,
                            help='')
        parser.add_argument('--trained_model_path', default="model_storage/best_model.pth", type=str,
                            help='')
        parser.add_argument('--log_level', default="INFO", type=str,
                            help='')
        parser.add_argument('--learning_rate', default=0.005, type=float,
                            help='')
        parser.add_argument('--load_vectorizer', default=False, type=self.str2bool, metavar='BOOL',
                            help='usually set to true during testing only. load vectorizer saved during training. if set to false during testing, will create a vectorizer'
                                 'based on the file provided under database_to_train_with ')

        parser.add_argument('--very_first_run', default=False, type=self.str2bool, metavar='BOOL',
                            help='this is for updating graph on comet.ml. If its second time onwards, you can use ExistingExperiment, hence. ')

        return parser.parse_args(namespace=self._args)

    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def join_data_dir_path(self,data_dir,filepath):
        LOG.debug(f"inside join_data_dir_path.")
        LOG.debug(f"value of filepath:{filepath}")
        LOG.debug(f"inside data_dir is:{data_dir}")
        path = os.path.join(data_dir, filepath)
        assert os.path.exists(path) is True
        assert os.path.isfile(path) is True
        return path

    def get_file_paths(self, args_in):
        '''
        decide the path of the local files based on whether we are running on server or laptop.
        #todo: move this to config file
        :return:
        '''
        cwd=os.getcwd()
        LOG.debug(f"inside get_file_paths(). value of cwd is:{cwd}")
        data_dir = os.path.join(cwd, args_in.data_dir_local)
        train_input_file=None
        dev_input_file=None
        test_input_file=None
        assert os.path.exists(data_dir) is True
        train_input_file = self.join_data_dir_path(data_dir, args_in.fever_train_local_lex)
        dev_input_file = self.join_data_dir_path(data_dir, args_in.fever_dev_local)
        test_input_file = self.join_data_dir_path(data_dir, args_in.fever_test_local)
        LOG.debug(f"train_input_file:{train_input_file}")
        LOG.debug(f"dev_input_file:{dev_input_file}")
        assert train_input_file is not None
        assert dev_input_file is not None

        if(args_in.run_type=="train"):
            LOG.debug(f"args_in.run_type==train")
            if (args_in.database_to_train_with == "fever_delex"):
                train_input_file=self.join_data_dir_path(data_dir, args_in.fever_train_local_delex)
                dev_input_file = self.join_data_dir_path(data_dir, args_in.fever_dev_local_delex)
                assert train_input_file is not None
                assert dev_input_file is not None
            elif (args_in.database_to_train_with == "mnli_lex"):
                train_input_file=self.join_data_dir_path(data_dir, args_in.mnli_train_lex)
                dev_input_file = self.join_data_dir_path(data_dir, args_in.mnli_matched_dev_lex)
                test_input_file = self.join_data_dir_path(data_dir, args_in.mnli_mismatched_test_lex)
                assert train_input_file is not None
                assert dev_input_file is not None
                assert test_input_file is not None
        elif(args_in.run_type=="test"):
            LOG.debug(f"args_in.run_type==test")
            #vectorizer needs to load any random dataset to return its class value- bad design choices
            train_input_file = self.join_data_dir_path(data_dir,args_in.fever_train_local_lex)
            LOG.debug(f"train_input_file:{train_input_file}")
            assert train_input_file is not None

            if (args_in.database_to_test_with == "fnc"):
                LOG.debug(f"args_in.database_to_test_with==fnc")
                test_input_file = self.join_data_dir_path(data_dir,args_in.fnc_test_local)
                assert test_input_file is not None
            elif (args_in.database_to_test_with == "fever"):
                LOG.debug(f"args_in.database_to_test_with==fever")
                test_input_file = os.path.join(data_dir, args_in.fever_test_local)
                assert test_input_file is not None
            elif (args_in.database_to_test_with == "mnli_lex"):
                LOG.debug(f"args_in.database_to_test_with==mnli_lex")
                test_input_file = os.path.join(data_dir, args_in.mnli_mismatched_test_lex)
                assert test_input_file is not None
            elif (args_in.database_to_test_with == "mednli_lex"):
                LOG.debug(f"args_in.database_to_test_with==mnli_lex")
                test_input_file = os.path.join(data_dir, args_in.mednli_dev)
                assert test_input_file is not None


        glove_filepath_in=self.join_data_dir_path(data_dir,args_in.glove_filepath)


        assert glove_filepath_in is not None
        assert train_input_file is not None
        assert dev_input_file is not None
        assert test_input_file is not None


        return glove_filepath_in,train_input_file,dev_input_file,test_input_file