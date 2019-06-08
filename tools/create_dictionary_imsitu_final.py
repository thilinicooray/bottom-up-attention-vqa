from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []

    #add all collected words from imsitu. contains both overlaps with vqa as well as new words
    imsitu_words_path = os.path.join(dataroot, 'allnverbsall_imsitu_words_nl2glovematching.json')
    imsitu_words = json.load(open(imsitu_words_path))

    for label, eng_name in imsitu_words.items():
        dictionary.tokenize(eng_name, True)

    print(' with words coming from imsitu ' , dictionary.__len__())

    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary('data')
    d.dump_to_file('data/dictionary_imsitu_final.pkl')

    d = Dictionary.load_from_file('data/dictionary_imsitu_final.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_init_imsitu_final_%dd.npy' % emb_dim, weights)
