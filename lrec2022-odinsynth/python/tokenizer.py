import config

class OdinsynthTokenizer:

    def __init__(self):
        self.cls = config.CLS_TOKEN
        self.sep = config.SEP_TOKEN
        self.pad = config.PAD_TOKEN
        self.unk = config.UNK_TOKEN

    def tokenize_batch(self, trees, sents, tokenize_triple=False, current_position=1):
        tokens_per_sequence = []
        current = trees[current_position]
        # encode each (tree, sentence) pair (or (current, next, sentence) tuple)
        for tree, sent in zip(trees, sents):
            if tokenize_triple:
                toks = self.tokenize_triple(current, tree, sent)
            else:
                toks = self.tokenize_pair(tree, sent)
                
            tokens_per_sequence.append(toks)
        # find length of longest sequence
        max_len = max(len(toks) for toks in tokens_per_sequence)
        # pad sequences
        for toks in tokens_per_sequence:
            pad_len = max_len - len(toks)
            pad = [self.pad] * pad_len
            toks.extend(pad)
        return tokens_per_sequence

    def tokenize_pair(self, tree, sent):
        tree_tokens = self.tokenize_tree(tree)
        sent_tokens = self.tokenize_sent(sent)
        tokens = [*tree_tokens, self.sep, *sent_tokens]
        return tokens

    def tokenize_triple(self, current_tree, tree, sent):
        current_tree = self.tokenize_tree(current_tree)
        tree_tokens  = self.tokenize_tree(tree)
        sent_tokens  = self.tokenize_sent(sent)
        tokens = [*current_tree, self.sep, *tree_tokens, self.sep, *sent_tokens]
        return tokens

    def tokenize_tree(self, tree):
        return tree.get_tokens()

    def tokenize_sent(self, sent):
        n_tokens = sent['numTokens']
        n_fields = config.NUM_SPEC_FIELDS
        len_encoded = n_tokens * n_fields
        tokens = [None] * len_encoded
        for i, name in enumerate(config.SPEC_FIELDS):
            tokens[i::n_fields] = [f'{name}-{token}' for token in sent[name]['tokens']]
        return [x.lower() for x in sent['word']['tokens']]
