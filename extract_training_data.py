from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)



def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label)
        elif rel == "right_arc":
            state.right_arc(label)

    return state.deps

class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"


def get_training_instances(dep_structure):

    deprels = dep_structure.deprels

    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1

    seq = []
    while state.buffer:
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy()
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):

    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)
        self.pos_vocab = self.read_vocab(pos_vocab_file)
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))

        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file:
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab

    def get_word(self,index,words, pos):
        find_word = ''
        pos_tag = pos[index]
        #check root
        if index == 0:
            pos_tag = '<ROOT>'

        #differnt format in pos.vocal []
        if pos_tag in ['NNP', 'CD']:
            pos_tag = '<'+pos_tag+'>'

        if pos_tag in ['<CD>', '<NNP>', '<UNK>', '<ROOT>', '<NULL>']:
            find_word = pos_tag
        #do we need to check whther the pos in pos_vocab??
        else:
            find_word = words[index]
        return find_word

    def check_vocab(self, word):
        #if the words not in ['<CD>, <NNP>, <UNK>, <ROOT>, <NULL>'] change the word to lowercase
        if word not in ['<CD>', '<NNP>', '<UNK>', '<ROOT>', '<NULL>']:
            word = word.lower()
        if word not in self.word_vocab.keys():
            return '<UNK>'
        return word

    def generate_features(self,list_type, feature_len, words, pos):
        output = []
        #make a copy of the list
        index_list = list_type.copy()
        #While indicated index_list is not empty
        i = 0
        while index_list and i < feature_len :
            #returen and remove the last item from the list
            index =  index_list.pop()
            #print('index list, ', index_list)
            vocab_word = self.get_word(index,words, pos)
            vocab_word = self.check_vocab(vocab_word)
            output.append(self.word_vocab[vocab_word])
            i += 1


        #Add the rest null
        #print('after remove all item in the index list, output: ', output)
        while len(output) < feature_len:
            output.append(self.word_vocab['<NULL>'])

        #print('final index list output ', output)
        return output


    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2
        #list append adds it to the end of the list.
        input_rep = []
        list_type = state.stack
        feature_len = 4
        input_rep = self.generate_features(list_type,feature_len,words, pos )
        list_type2 = state.buffer
        input_rep2 = self.generate_features(list_type2,feature_len,words, pos )
        input_rep = input_rep2 + input_rep

        #print('final input rep: ', input_rep)

        return np.asarray(input_rep)

    def get_output_representation(self, output_pair):
        # TODO: Write this method for Part 2
        value = self.output_labels[output_pair]
        encoded_value = keras.utils.to_categorical(value, num_classes=91)

        return encoded_value




def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0
    for dtree in conll_reader(in_file):
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)



if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'
    #python extract_training_data.py data/train.conll data/words.vocab data/pos.vocab
    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)


    with open(sys.argv[1],'r') as in_file:

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)
