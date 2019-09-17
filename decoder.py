from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import tensorflow.keras as keras

from extract_training_data import FeatureExtractor, State

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        #print('words ', words)
        #print('pos ', pos)

        while state.buffer:
            pass
            # TODO: Write the body of this loop for part 4

            # deps format 19.95	_	CD	CD	_	12	num	_	_

            #first feature extractor get the representation of current states
            cur_input = self.extractor.get_input_representation(words, pos, state)
            #print('cur_input reshape ', cur_input.reshape(1,-1))
            output = self.model.predict(cur_input.reshape(1,-1))
            #Index in descending order
            sorted_index = np.argsort(output[0])[::-1]
            #print('output ', output)
            #print('sorted_index ', sorted_index)
            #check the illigal
            for high_index in sorted_index:
                #print('high_index ',high_index)
                #check the output label to find the tranisition action
                action = self.output_labels[high_index]
                #print('action ', action)


                #check the validation of the action
                #if action works, update states, break the for loop
                #otherwise continue the loop
                #print('action[0] ',action[0])
                if action[0] in ['right_arc', 'left_arc']:
                    relation = action[1]
                    #print('state.stack ',state.stack)
                    if len(state.stack) != 0 :
                        #form the dep [(parent, child, relation)]
                        #(6, 8, 'dobj')
                        if action[0] == 'left_arc':
                            parent = state.buffer[-1]
                            child = state.stack[-1]
                            #check target is not root
                            #print('check target is not root: child =', child)
                            if child == 0:
                                continue
                            state.stack.pop(-1)
                        else:
                            parent = state.stack[-1]
                            child = state.buffer[-1]
                            state.deps.add((parent,child,relation))
                            state.buffer.pop(-1)
                            state.buffer.append(state.stack.pop(-1))
                        dep = (parent,child,relation)
                        #print('new dep: ', dep)
                        state.deps.add((parent,child,relation))
                        break
                else:
                    if (len(state.stack) != 0 ) and (len(state.buffer) == 1) :
                        #print('shifting when stack is empty')
                        #print('len(state.stack):  ', len(state.stack))
                        #print('len(state.buffer):  ',len(state.buffer))
                        continue
                    else:
                        #state.deps.add('None')
                        state.stack.append(state.buffer.pop(-1))
                        break
        #print('state.deps ',state.deps)

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
