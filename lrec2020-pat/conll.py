import re
import tempfile
import subprocess
from utils import normalize, chunker



def read_conll(filename, include_non_projective=True, verbose=True, lower_case=True):
    """Reads dependency annotations from CoNLL-U format"""
    return list(iter_conll(filename, include_non_projective, verbose, lower_case))



def write_conll(filename, sentences):
    """Write sentences to conllx file"""
    with open(filename, 'w') as f:
        for sentence in sentences:
            for entry in sentence:
                if entry.id > 0:
                    print(str(entry), file=f)
            print('', file=f)



def eval_conll(sentences, gold_filename, verbose=True):
    with tempfile.NamedTemporaryFile(mode='w') as f:
        for sentence in sentences:
            for entry in sentence:
                if entry.id > 0:
                    print(str(entry), file=f)
            print('', file=f)
        f.flush()
        p = subprocess.run(['./eval.pl', '-g', gold_filename, '-s', f.name], stdout=subprocess.PIPE)
        o = p.stdout.decode('utf-8')
        if verbose: print(o)
        m1 = re.search(r'Unlabeled attachment score: (.+)', o)
        m2 = re.search(r'Labeled   attachment score: (.+)', o)
        return m1.group(1), m2.group(1)



def parse_conll(parser, sentences, batch_size, clear=True):
    if clear:
        clear_dependencies(sentences)
    for batch in chunker(sentences, batch_size):
        parser.parse_conll(batch)
    if parser.mode == 'evaluation' and parser.print_nr_of_cycles:
        print("Nr of cycles: ", parser.nr_of_cycles, ", ", len(sentences), " ", parser.nr_of_cycles / len(sentences))



def clear_dependencies(sentences):
    for sentence in sentences:
        for entry in sentence:
            entry.head = None
            entry.deprel = None
            entry.pos = None



def iter_conll(filename, include_non_projective=True, verbose=True, lower_case=True):
    """Reads dependency annotations in CoNLL-U format and returns a generator."""
    read = 0
    non_proj = 0
    dropped = 0
    root = ConllEntry(id=0, form='<root>', upos='<root>', xpos='<root>', head=0, deprel='rroot')
    with open(filename) as f:
        sentence = [root]
        for line in f:
            if line.isspace() and len(sentence) > 1:
                if is_projective(sentence):
                    yield sentence
                else:
                    non_proj += 1
                    if include_non_projective:
                        yield sentence
                    else:
                        dropped += 1
                read += 1
                sentence = [root]
                continue
            entry = ConllEntry.from_line(line, lower_case=lower_case)
            sentence.append(entry)
        # we may still have one sentence in memory
        # if the file doesn't end in an empty line
        if len(sentence) > 1:
            if is_projective(sentence):
                yield sentence
            else:
                non_proj += 1
                if include_non_projective:
                    yield sentence
                else:
                    dropped += 1
            read += 1
    if verbose:
        print(f'{read:,} sentences read.')
        print(f'{non_proj:,} non-projective sentences found, {dropped:,} dropped.')
        print(f'{read-dropped:,} sentences remaining.')



def is_projective(sentence):
    """returns true if the sentence is projective"""
    roots = list(sentence)
    # keep track of number of children that haven't been
    # assigned to each entry yet
    unassigned = {
        entry.id: sum(1 for e in sentence if e.head == entry.id)
        for entry in sentence
    }
    # we need to find the parent of each word in the sentence
    for _ in range(len(sentence)):
        # only consider the forest roots
        for i in range(len(roots) - 1):
            # attach entries if:
            #   - they are parent-child
            #   - they are next to each other
            #   - the child has already been assigned all its children
            if roots[i].head == roots[i+1].id and unassigned[roots[i].id] == 0:
                unassigned[roots[i+1].id] -= 1
                del roots[i]
                break
            if roots[i+1].head == roots[i].id and unassigned[roots[i+1].id] == 0:
                unassigned[roots[i].id] -= 1
                del roots[i+1]
                break
    # if more than one root remains then it is not projective
    return len(roots) == 1



class ConllEntry:

    def __init__(self, id=None, form=None, lemma=None, upos=None, xpos=None,
                 feats=None, head=None, deprel=None, deps=None, misc=None, lower_case=True):
        # conll-u fields
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        # normalized token
        self.norm = normalize(form, to_lower=lower_case)
        # relative position of token's head
        self.pos = 0 if self.head == 0 else self.head - self.id

    def __repr__(self):
        return f'<ConllEntry: {self.form}>'

    def __str__(self):
        fields = [
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feats,
            self.head,
            self.deprel,
            self.deps,
            self.misc
        ]
        return '\t'.join('_' if f is None else str(f) for f in fields)

    def get_partofspeech_tag(self, pos_type):
        if pos_type == 'upos':
            return self.upos
        elif pos_type == 'xpos':
            return self.xpos
        else:
            raise ValueError(f"Unknown type; {pos_type}")

    @staticmethod
    def from_line(line, lower_case=True):
        fields = [None if f == '_' else f for f in line.strip().split('\t')]
        if fields[1] is None:
            fields[1] = '_'
        fields[0] = int(fields[0]) # id
        fields[6] = int(fields[6]) # head
        fields[7] = str(fields[7]) # deps
        return ConllEntry(*fields, lower_case=lower_case)
