from csv import DictReader
import os
import sys
import csv
import pandas as pd
import numpy as np



def readAdjInterceptFile(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep=',',header=None,names=['adj','intercept'])
    data['adj'] = data['adj'].map(lambda x: x.lstrip('(').rstrip(')'))
    data['intercept'] = data['intercept'].map(lambda x: x.lstrip('(').rstrip(')'))

    return data;




def readWithSpace(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep='\t')

    #print(data["logrespdev"][0])
    #shuffle the data to make sure we have mixed it evenly
    #data= data.sample(frac=1).reset_index(drop=True)
    #print(data["logrespdev"][0])

    return data;

def readRawTurkDataFile(cwd, inputFile):

    path = cwd+"/data/"
    data =pd.read_csv(path  + inputFile,sep=',')


    #shuffle the data to make sure we have mixed it evenly
    #data= data.sample(frac=1).reset_index(drop=True)
    #print(data["logrespdev"][0])

    return data;

def writeToFileWithPd( trainingData,cwd, inputFile):
    path = cwd + "/data/"
    trainingData.to_csv(path + inputFile, sep=',')

#for turk data, if you have it without header, this code will write the header for you
def writeToFileWithHeader(data, cwd, inputFile):
    with open(cwd+"/data/"+inputFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["turker", "adjective", "mean","onestdev","had_negative","logrespdev"])

        for line in data:
            writer.writerow(line)

#input:dictionary output:csv
def writeCsvToFile(data, cwd, inputFile):
    with open(cwd+"/data/"+inputFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for k,v in data.items():
            writer.writerow(str(k),str(v))

    csvfile.close()

def writeDictToFile(data, cwd, inputFile):
    with open(cwd+"/data/"+inputFile, 'w', newline='') as dictfile:
        for k,v in data.items():
            dictfile.write(str(k)+" ")
            for eachValue in v:
                dictfile.write(str(eachValue)+" ")
            dictfile.write(str("\n"))

    dictfile.close()


#use pytorch to read demarneffe matrix.
def loadEmbeddings(gloveFile):
    print("Loading demarneffe Model:")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
