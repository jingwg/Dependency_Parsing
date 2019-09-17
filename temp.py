#stack first
#var1 = list_type
#var2 = feature_len
list_type = state.stack
feature_len = 4
output = generate_features(self, list_type,feature_len,words, pos )
list_type2 = state.buffer
output2 = generate_features(self,list_type2,feature_len,words, pos )
output = output2 + output

def generate_features(list_type, feature_len, words, pos):
    output = []
    #make a copy of the list
    index_list = list_type.copy()
    #While indicated index_list is not empty
    while index_list:
        #returen and remove the last item from the list
        index =  index_list.pop()
        print('index list, ', index_list)
        vocab_word = self.get_word(index,words, pos)
        vocab_word = self.check_vocab(vocab_word)
        output.append(self.word_vocab[vocab_word])

    #Add the rest null
    print('after remove all item in the index list, output: ', output)
    while len(output) < feature_len:
        output.append(self.word_vocab['<NULL>'])

    print('final index list output ', output)
