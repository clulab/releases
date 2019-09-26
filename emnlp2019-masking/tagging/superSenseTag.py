from tqdm import tqdm
import json,mmap,os,argparse
import processors
from processors import *
from nltk.tokenize import sent_tokenize

def parse_commandline_args():
    return create_parser().parse_args()

def create_parser():
    parser = argparse.ArgumentParser(description='Pg')
    parser.add_argument('--inputFile', type=str, default='data/fever_train_split_fourlabels.jsonl',
                        help='name of the input file to convert to smart ner format')
    parser.add_argument('--pyproc_port', type=int, default=8888,
                        help='port at which pyprocessors server should run. If you are running'
                             'multiple servers on the same machine, will need different port for each')
    parser.add_argument('--use_docker', default=False, type=str2bool,
                        help='use docker for loading pyproc. useful in machines where you have root access.', metavar='BOOL')
    parser.add_argument('--output_folder', type=str, default='outputs/',
                        help='folder where outputs will be created')
    parser.add_argument('--write_pos_tags', default=False, type=str2bool,
                        help='if you want to write the pos tags to disk or not. This is used as input for the SSTagger.',
                        metavar='BOOL')


    print(parser.parse_args())
    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def read_rte_data(filename):
    tr_len = 1000
    all_labels = []
    all_claims = []
    all_evidences = []

    with open(filename) as f:
        for index, line in enumerate(tqdm(f, total=get_num_lines(filename))):
            multiple_ev = False
            x = json.loads(line)
            claim = x["claim"]
            evidences = x["evidence"]
            label = x["label"]

            all_claims.append(claim)
            all_evidences.append(evidences)
            all_labels.append(label)

    return all_claims, all_evidences, all_labels





def annotate(headline, body, API):
    claim_ann = API.fastnlp.annotate(headline)
    ev_ann = API.fastnlp.annotate(body)
    return claim_ann, ev_ann

def replacePrepositionsWithPOSTags(claim_ann, evidence_ann):

    # claimn=claim_ann.words
    # evidencen = evidence_ann.words

    claim_ner_tags=claim_ann._entities
    ev_ner_tags = evidence_ann._entities

    for index,pos in enumerate(claim_ann.tags):
        if (pos=="IN"):
            claim_ner_tags[index]="PREP"


    for index,pos in enumerate(evidence_ann.tags):
        if (pos=="IN"):
            ev_ner_tags[index]="PREP"



    return claim_ner_tags, ev_ner_tags

def write_json_to_disk(claim, evidence,label,outfile):
    total = {'claim': claim,
             'evidence_from_lexicalized_data':evidence,
             "label":label}
    json.dump(total, outfile)
    outfile.write('\n')

def write_token_POS_disk_as_csv(annotated_sent,full_path_output_file):
    # if the file already exists, leave it. It might have been written in a run before
    if not (os.path.isfile(full_path_output_file)):
        with open(full_path_output_file, 'w') as outfile:
            outfile.write('')
            with open(full_path_output_file, 'a+') as outfile:
                for word,postag in zip(annotated_sent.words, annotated_sent.tags):
                    outfile.write(word+"\t"+postag+"\n")

def write_list_of_evidence_sentences_to_disk(annotated_sentences,full_path_output_file):
        if not (os.path.isfile(full_path_output_file)):
            with open(full_path_output_file, 'w') as outfile:
                outfile.write('')
                with open(full_path_output_file, 'a+') as outfile:
                    for annotated_sent in annotated_sentences:
                        for word, postag in zip(annotated_sent.words, annotated_sent.tags):
                            outfile.write(word + "\t" + postag + "\n")
                        outfile.write("\n")

if __name__ == '__main__':

    args = parse_commandline_args()
    if(args.use_docker==True):
        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)
    else:
        API = ProcessorsAPI(port=args.pyproc_port)

    cwd=os.getcwd()
    filename=cwd+"/"+args.inputFile
    all_claims, all_evidences, all_labels=read_rte_data(filename)
    all_claims_neutered=[]

    output_folder=args.output_folder



    length=len(all_evidences)
    index=0

    for (c, e ,l) in tqdm(zip(all_claims, all_evidences,all_labels),total=length):
        claim_ann = API.fastnlp.annotate(c)
        assert (claim_ann is not None)

        # new requirement. SSTagger wants a newline after every sentence. So will have to find sentence boundaries.
        evidence_sentences=sent_tokenize(e)
        evidence_sentences_annotated=[]
        for evidence_sentence in evidence_sentences:
            ev_ann = API.fastnlp.annotate(evidence_sentence)
            assert (ev_ann is not None)
            evidence_sentences_annotated.append(ev_ann)

        if(args.write_pos_tags):
            # write each token and its pos tag to disk, with one line each-This is used as input for the SSTagger
            out_file_name="claim_words_pos_datapointid_"+str(index)
            full_path_output_file=output_folder+out_file_name
            write_token_POS_disk_as_csv(claim_ann, full_path_output_file)

            out_file_name = "evidence_words_pos_datapointid_" + str(index)
            full_path_output_file = output_folder + out_file_name
            write_list_of_evidence_sentences_to_disk(evidence_sentences_annotated, full_path_output_file)
            index=index+1






