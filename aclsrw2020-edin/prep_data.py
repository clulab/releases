import argparse
from bio_utils import *
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('datadir2')
    parser.add_argument('dev_datadir')
    parser.add_argument('test_datadir')
    parser.add_argument('event')
    parser.add_argument('outdir')
    args = parser.parse_args()

    input_lang = Lang("input")
    pl1 = Lang("position")
    char = Lang("char")
    rule_lang = Lang("rule")
    raw_train = list()

    input_lang, pl1, char, rule_lang, raw_train = prepare_data(args.datadir, args.event, input_lang, pl1, char, rule_lang, raw_train)
    input_lang, pl1, char, rule_lang, raw_train = prepare_data(args.datadir2, args.event, input_lang, pl1, char, rule_lang, raw_train, "%s/rule_mappings.json"%args.datadir2)
    
    input2_lang, pl2, char2, rule_lang2, raw_dev = prepare_data(args.dev_datadir, args.event, valids="rule_mappings.json")
    input3_lang, pl3, char3, rule_lang3, raw_test = prepare_test_data(args.test_datadir)
    
    os.mkdir(args.outdir)
    with open('%s/train'%args.outdir, "wb") as f:
        pickle.dump((input_lang, pl1, char, rule_lang, raw_train), f)
    with open('%s/dev'%args.outdir, "wb") as f:
        pickle.dump((input2_lang, pl2, char2, rule_lang2, raw_dev), f)
    with open('%s/test'%args.outdir, "wb") as f:
        pickle.dump((input3_lang, pl3, char3, rule_lang3, raw_test), f)