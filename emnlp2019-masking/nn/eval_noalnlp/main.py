from comet_ml import Experiment,ExistingExperiment
from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix,create_model,set_seed_everywhere
from mean_teacher.utils.logger import LOG
import time


current_time={time.strftime("%c")}
LOG.info(f"starting the run at {current_time}.")

def initialize_comet(args):
    # for drawing graphs on comet:
    comet_value_updater=None
    if(args.run_type=="train"):
        if(args.very_first_run==True):
            comet_value_updater = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", project_name="rte-decomp-attention")
        else:
            comet_value_updater = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="80c6e42f6d8e417d86906a6423345a05")

    return comet_value_updater


initializer=Initializer()
initializer.set_default_parameters()
args = initializer.parse_commandline_args()
comet_value_updater=initialize_comet(args)
if (comet_value_updater) is not None:
    hyper_params = vars(args)
    comet_value_updater.log_parameters(hyper_params)
set_seed_everywhere(args)
LOG.setLevel(args.log_level)

if args.run_type=="test":
    args.load_vectorizer=True
    args.load_model_from_disk=True


glove_filepath_in, train_input_file, dev_input_file, test_input_file=initializer.get_file_paths(args)


LOG.debug(" glove path:{glove_filepath_in}")
LOG.debug(f"value of train_input_file is :{train_input_file}")
LOG.debug(f"value of dev_input_file is :{dev_input_file}")


if args.load_vectorizer:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(train_input_file, dev_input_file, test_input_file,
                                                          args)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.create_vocabulary(train_input_file, dev_input_file, test_input_file, args)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()


# taking embedding size from user initially, but will get replaced by original embedding size if its loaded
embedding_size=args.embedding_size

# Use GloVe or randomly initialized embeddings
if args.use_glove:
    words = vectorizer.claim_ev_vocab._token_to_idx.keys()
    embeddings,embedding_size = make_embedding_matrix(glove_filepath_in,words)
    LOG.info(f"{current_time:} Using pre-trained embeddings")
else:
    LOG.info(f"{current_time:} Not using pre-trained embeddings")
    embeddings = None

num_features=len(vectorizer.claim_ev_vocab)

train_rte=Trainer()

classifier = create_model(logger_object=LOG,args_in=args,num_classes_in=len(vectorizer.label_vocab)
                              ,word_vocab_embed=embeddings,word_vocab_size=num_features,wordemb_size_in=embedding_size)

if args.run_type == "train":
    train_rte.train(args,classifier,dataset,comet_value_updater)
elif args.run_type=="test":
    train_rte.test(args,classifier,dataset,"test",vectorizer)
elif args.run_type == "val":
    train_rte.test(args,classifier, dataset, "val")
