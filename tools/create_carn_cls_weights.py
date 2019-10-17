from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_glove_embedding_init(object_list, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(object_list), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, object in enumerate(object_list):

        words = object.split()
        final_embedding = word2emb[words[0]]

        for word in words[1:]:
            final_embedding += word2emb[word]

        weights[idx] = final_embedding
    return weights, word2emb


if __name__ == '__main__':
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim

    with open('data/object_name_list.txt') as f:
        content = f.readlines()
    object_list = [x.strip() for x in content]

    weights, word2emb = create_glove_embedding_init(object_list, glove_file)
    np.save('data/glove6b_init_imsitu_carn.npy' % emb_dim, weights)
