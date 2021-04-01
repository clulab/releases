import re



class Instance:

    def __init__(self, sen, start, end, trigger, polarity, pred_polarity, rule_name):
        self.original = sen
        self.start = start  # + 1 # Plus one to account for the special start/end of sentence tokens
        self.end = end  # + 1
        self.original_trigger = trigger
        self.trigger = trigger.lower().strip()
        self.polarity = polarity  # True for positive, False for negative
        self.pred_polarity = pred_polarity
        self.tokens = Instance.normalize(sen)
        self.rule_name = rule_name.lower()
        self.rule_polarity = True if self.rule_name.startswith("positive") else False;
        self.neg_count = self.get_negCount()

    def get_tokens(self, k=0):
        start = max(0, self.start - k)
        end = min(len(self.tokens) - 1, self.end + k)
        return self.tokens[start:end]

    @staticmethod
    def _is_number(w):
        i = 0
        found_digit = False
        while i < len(w):
            c = w[i]
            match = re.match("\r", c)
            if match is None and c != '-' and c != '+' and c != ',' and c != '.' and c != '/' and c != '\\':
                return False
            if match:
                found_digit = True
            i += 1

        return found_digit

    @staticmethod
    def _sanitize_word(uw, keepNumbers=True):
        w = uw.lower()

        # skip parens from corenlp
        if w == "-lrb-" or w == "-rrb-" or w == "-lsb-" or w == "-rsb-":
            return ""

        # skip URLS
        if w.startswith("http") or ".com" in w or ".org" in w:  # added .com and .org to cover more urls (becky)
            return ""

        # normalize numbers to a unique token
        if Instance._is_number(w):
            return "xnumx" if keepNumbers else ""

        # remove all non-letters; convert letters to lowercase
        os = ""
        i = 0
        while i < len(w):
            c = w[i]
            # added underscore since it is our delimiter for dependency stuff...
            if re.match(r"[a-z]", c) or c == '_':
                os += c
            i += 1

        return os

    @staticmethod
    def normalize(raw):
        sentence = raw.lower()
        # Replace numbers by "[NUM]"
        #sentence = re.sub(r'(\s+|^)[+-]?\d+\.?(\d+)(\s+|$)?', ' [NUM] ', sentence)
        tokens = [Instance._sanitize_word(t) for t in sentence.split()]

        return tokens
        # return ['[START]'] + tokens + ['[END]']

    @staticmethod
    def from_dict(d):
        return Instance(d['sentence text'],
                        int(d['event interval start']),
                        int(d['event interval end']),
                        d['trigger'],
                        # Remember the polarity is flipped because of SIGNOR
                        True if d['polarity'].startswith('Positive') else False,
                        True if d['pred_polarity'].startswith('Positive') else False,
                        d['rule'])

    def get_segments(self, k=2):
        trigger_tokens = self.trigger.split()

        trigger_ix = self.tokens.index(Instance._sanitize_word(trigger_tokens[0]), self.start, self.end+1)
        tokens_prev = self.tokens[max(0, self.start - k):self.start]
        tokens_in_left = self.tokens[self.start:(trigger_ix+len(trigger_tokens)-1)]
        tokens_in_right = self.tokens[(trigger_ix+len(trigger_tokens)):self.end]
        tokens_last = self.tokens[min(self.end, len(self.tokens)-1):min(self.end+k, len(self.tokens)-1)]

        return tokens_prev, tokens_in_left, tokens_in_right, tokens_last
        
    def get_negCount(self):
        event_text = ' '.join(word for word in self.tokens[self.start:self.end])
        neg_count = len(re.findall(r'(?=attenu|block|deactiv|decreas|degrad|delet|deplet|diminish|disrupt|dominant-negative|impair|imped|inhibit|knockdown|knockout|limit|loss|lower|negat|reduc|reliev|repress|restrict|revers|silenc|shRNA|siRNA|slow|starv|suppress|supress|turnover|off)', event_text))
        return neg_count


class WordEmbeddingIndex(object):

    def __init__(self, w2v_data, w2v_ix):
        self.w2v_data = w2v_data
        self.w2v_index = w2v_ix

    def __getitem__(self, w):
        return self.w2v_data[self.w2v_index[w]] if w in self.w2v_index else self.w2v_data[self.w2v_index["*unknown*"]]

class CharEmbeddingIndex(object):
    def __init__(self, c2v_data, char_embeddings):
    # c2v_data is the dynet lookup parameter, character_dict_path is the path of all characters.
        self.char_dict = char_embeddings
        self.c2v_data = c2v_data
        self.c2v_index = {w:i for i,w in enumerate(sorted(list(self.char_dict.keys())))}
        #print('sorted c2v dict:', self.c2v_index)

    def __getitem__(self, c):
        return self.c2v_data[self.c2v_index[c]] if c in self.char_dict else self.c2v_data[len(self.char_dict)]


def build_vocabulary(words):
    index, reverse_index = dict(), dict()
    for i, w in enumerate(sorted(words)):
        index[w] = i
        reverse_index[i] = w

    return index, reverse_index

def build_char_dict(instances):
    char_dict = {}
    for instance in instances:
        for token in instance.tokens:
            for character in token:
                if character=='':
                    print('have empty token!  ', instance.tokens)
                    input('press enter to continue')
                char_dict[character] = 1
    char_dict['']=1
    return char_dict