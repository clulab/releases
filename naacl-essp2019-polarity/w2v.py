import gzip, pickle
import numpy as np
import json



class W2VEmbeddings:

    def __init__(self, raw):
        UNK = np.random.random(100) #TODO: Automatically figure this out
        self.keys = set(raw.keys()) | {"*unknown*"}
        self.keys.remove("1579375")  # Had to remove it manually
        self.voc = {w:ix for ix, w in enumerate(sorted(list(self.keys)))}
        arrays = [raw[k] if k != "*unknown*" else UNK for k in sorted(self.keys)]

        self.matrix = np.vstack(arrays)

    def __contains__(self, item):
        return item in self.keys

    def __getitem__(self, item):
        return self.voc[item]

    def shape(self):
        return self.matrix.shape

    def to_list(self):
        return sorted(list(self.keys))


def load_embeddings(path):

    with gzip.open(path, "r") as f:
        raw = pickle.load(f)

    return W2VEmbeddings(raw)


def save_as_json(path):
    with gzip.open(path, "r") as f:
        raw = pickle.load(f)

    new_file_name = '.'.join(path.split('.')[-2:0])+'.json'

    marshalled = {k: list(v) for k, v in raw.items()}

    with open(new_file_name, 'w') as f:
        json.dump(marshalled, f)


def save_as_txt(path):
    with gzip.open(path, "r") as f:
        raw = pickle.load(f)

    new_file_name = '.'.join(path.split('.'))+'.txt'

    lines = list()
    for k, v in raw.items():
        lines.append('%s\t%s\n' % (k, ','.join(str(i) for i in v)))

    with open(new_file_name, 'w') as f:
        f.writelines(lines)

