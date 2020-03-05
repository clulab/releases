import re
import math



def normalize(word, to_lower = True):
    """returns a normalized version of the given word"""
    if re.fullmatch(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', word):
        return '<num>'
    elif to_lower:
        return word.lower()
    else:
        return word


def chunker(sequence, size):
    """useful for splitting a sequence into minibatches"""
    for i in range(0, len(sequence), size):
        chunk = sequence[i:i+size]
        # sort sentences in batch by length in descending order
        # this is needed for padding
        chunk.sort(key=lambda x: len(x), reverse=True)
        yield chunk



def parse_uas(uas):
    m = re.search(r'(\d+) / (\d+)', uas)
    correct, total = m.groups()
    return float(correct) / float(total)

# Returns the first element in the collection that has predicate(element) True. If none exists, returns if_not
def first(collection, predicate, if_not = None):
    for element in collection:
        if predicate(element):
            return element
    return if_not


# https://arxiv.org/pdf/1801.06146.pdf 3.2 Slanted triangular learning rates
# Notations were kept as in the article, with the sole exception learning rate, which in the article is the greek letter eta
# Their defaults: cut_frac = 0.1, ratio = 32, lr_max = 0.01

# t is the time step
# T is the number of training iterations
# cut_frac is the fraction of iterations we increase the learning rate
# cut is the iteration when we switch from increasing to decreasing the LR
# p is the fraction of the number of iterations we have increased or will decrease the learning rate respectively
# ratio specifies how much smaller the lowest learning rate is from the maximum learning rate
# lr_max is the maximum learning rate
def get_slanted_triangular_lr(t, T, cut_frac=0.1, ratio=32, lr_max=0.01):
    cut = math.floor(T * cut_frac)
    p = t/cut if t < cut else 1 - (t-cut)/(cut*((1/cut_frac) - 1))
    lr = lr_max * (1+p*(ratio-1))/ratio
    return lr
