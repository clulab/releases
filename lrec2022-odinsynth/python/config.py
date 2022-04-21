from pathlib import Path

MODEL_NAME='odinsynth'

DATA_DIR = Path('/data/nlp/corpora/odinsynth/data/toy_unrolled')
# DATA_DIR = Path('/data/nlp/corpora/odinsynth/data/random-25k')

SPLIT_DATASET_ON_LENGTH = True
SPLIT_DATASET_THRESHOLD = 20

SPECS_DIR = DATA_DIR/'specs'
STEPS_DIR = DATA_DIR/'steps'

VOCABULARY_FILENAME = DATA_DIR/'vocabulary.txt'

USE_CUDA = True

RANDOM_SEED = 1

NUM_EPOCHS = 100

LEARNING_RATE = 3e-5

TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
NAX_SENTENCE_LENGTH_TRAIN = 30
NAX_SENTENCE_LENGTH_VALID = 500
NUM_WORKERS = 0

MARGIN = 1.0

WEIGHT_DECAY = 0.001
NO_DECAY = ('bias', 'LayerNorm')

TOKEN_FIELDS = ('word', 'lemma', 'tag')
# TOKEN_FIELDS = ('raw', 'word', 'lemma', 'tag', 'entity', 'chunk')

SELECTION_FIELD = 'selection'
SELECTION_LABEL = 'SEL'
NO_SELECTION_LABEL = 'O'

SPEC_FIELDS = TOKEN_FIELDS + (SELECTION_FIELD,)
NUM_SPEC_FIELDS = len(SPEC_FIELDS)

UNK_TOKEN  = '[UNK]'
PAD_TOKEN  = '[PAD]'
CLS_TOKEN  = '[CLS]'
SEP_TOKEN  = '[SEP]'
MASK_TOKEN = '[MASK]'
SPECIAL_TOKENS = {
    'pad_token' : PAD_TOKEN,
    'mask_token': MASK_TOKEN,
    'unk_token' : UNK_TOKEN,
    'cls_token' : CLS_TOKEN,
    'sep_token' : SEP_TOKEN
    }
TREE_SPECIAL_TOKENS = ['AST-OrConstraint-start', 
    'AST-OrConstraint-sep', 
    'AST-AndConstraint-start', 
    'AST-quantifier-*', 
    'AST-quantifier-?', 
    'AST-OrConstraint-end', 
    'AST-HoleConstraint', 
    'AST-FieldConstraint-end', 
    'AST-TokenQuery-end', 
    'AST-RepeatQuery-start', 
    'AST-OrQuery-sep', 
    'AST-NotConstraint-start', 
    'AST-TokenQuery-start', 
    'AST-AndConstraint-end', 
    'AST-OrQuery-start', 
    'AST-OrQuery-end', 
    'AST-AndConstraint-sep', 
    'AST-NotConstraint-end', 
    'AST-ConcatQuery-start', 
    'AST-quantifier-+', 
    'AST-HoleMatcher', 
    'AST-HoleQuery', 
    'AST-ConcatQuery-sep', 
    'AST-ConcatQuery-end', 
    'AST-FieldConstraint-start', 
    'AST-RepeatQuery-end',
    'fieldname-word',
    'fieldname-lemma',
    'fieldname-tag',
    ]
    
POS_TAGS = [
    'RBR', 
    'CC', 
    'SYM', 
    'WRB', 
    'VBZ', 
    'RBS', 
    'LS', 
    'VBG', 
    'NNP', 
    'PRP$', 
    'JJ', 
    'JJS', 
    'VBD', 
    '-LRB-', 
    'JJR', 
    'PRP', 
    'NN', 
    '#', 
    'WP$', 
    'VBP', 
    'RP', 
    'POS', 
    'IN', 
    'RB', 
    'CD', 
    'NNS', 
    'VBN', 
    'NNPS', 
    'MD', 
    'WDT', 
    'VB', 
    'DT', 
    '-RRB-', 
    'WP', 
    'FW', 
    'UH', 
    'TO', 
    'PDT', 
    'EX', 
]
    
HOLE_GLYPH = '\u25a1'  # WHITE SQUARE

INFINITY = float('inf')

HIDDEN_SIZE = 768
# NUM_HIDDEN_LAYERS = 12
NUM_HIDDEN_LAYERS = 6
# NUM_ATTENTION_HEADS = 12
NUM_ATTENTION_HEADS = 8
INTERMEDIATE_SIZE = 3072
HIDDEN_ACT = 'gelu'
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
MAX_POSITION_EMBEDDINGS = 512
TYPE_VOCAB_SIZE = 2
INITIALIZER_RANGE = 0.02
LAYER_NORM_EPS = 1e-12
GRADIENT_CHECKPOINTING = False
