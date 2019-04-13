from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot):
    dictionary = Dictionary()
    files = [
        'imsitu_questions_prev.json'
    ]

    for path in files:
        question_path = os.path.join(dataroot, path)
        q_data = json.load(open(question_path))

        for verb, values in q_data.items():
            roles = values['roles']
            for role, info in roles.items():
                question = info['question']
                dictionary.tokenize(question, True)

    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    print('word count create dict:', len(idx2word))
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
    d.dump_to_file('data/dictionary_imsitu.pkl')

    d = Dictionary.load_from_file('data/dictionary_imsitu.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_init_imsitu_%dd.npy' % emb_dim, weights)
