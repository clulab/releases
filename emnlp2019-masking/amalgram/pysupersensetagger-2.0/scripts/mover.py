#for all files which were already written, remove the corresponding input files. This way we can check if the code is even running at all or not
import os,sys,subprocess
from os import listdir
inputFolder="/work/mithunpaul/neuter_ner_fever_training/amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger/"
files=listdir(inputFolder)
for x in files:
    if x.endswith(".pred.tags"):
        print(x)
        fullpath=os.path.join(inputFolder,x)
        os.unlink(fullpath)
