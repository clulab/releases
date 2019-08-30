import csv,os,sys,random
from random import shuffle
from retrieval.read_claims import UOFADataReader
from tqdm import tqdm
from processors import ProcessorsBaseAPI

class load_fever_DataSet():




    def read_parent(self,cwd,bodies,stances):
        path = cwd + "/data/fnc/"

        # read the stances into a dictionary. Note that stances are in the format: Headline,Body ID,Stance
        stances = self.read(path,stances)
        articles = self.read(path,bodies)

        # articles_dict = dict()
        #
        # # make the body ID an integer value
        # for s in self.stances:
        #     s['Body ID'] = int(s['Body ID'])
        #
        # # copy all bodies into a dictionary
        # for article in articles_list:
        #     articles_dict[int(article2['Body ID'])] = article['articleBody']

        return stances,articles

    def read(self,path,filename):

        #this is to make sure that the shuffling happens same way every time. I had to do this, because the annotation process
        #itself was hanging up in between. So if its shuffled same way, i can atleast restart from where i left.
        random.seed(448)

        rows = []
        with open(path  + filename, encoding='utf8') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        shuffle(rows)

        return rows


    def annotate_fnc(self, stances,articles,logger,path_to_pyproc_annotated_data_folder):

        API = ProcessorsBaseAPI(hostname="127.0.0.1", port=8886, keep_alive=True)

        ann_head = "split_head/ann_head.json"
        ann_body = "split_body/ann_body.json"
        labels_file = "label_id.csv"

        try:
            os.remove(ann_head)
            os.remove(ann_body)
            os.remove(labels_file)

        except OSError:
            print("not able to find file")
        objUOFADataReader = UOFADataReader()

        for index,s in enumerate(tqdm(stances,total=len(stances),desc="for_each_stance:")):

            headline = s['Headline']
            bodyid = int(s['Body ID'])
            actualBody = articles[bodyid]
            hypothesis = headline
            premise = actualBody

            # get the label/original class and write it to disk
            label = s['Stance']

            objUOFADataReader.annotate_and_save_doc_with_label_as_id(hypothesis, premise, label, API, ann_head, ann_body, logger, path_to_pyproc_annotated_data_folder)
            objUOFADataReader.write_label_to_disk(bodyid,label,labels_file)



