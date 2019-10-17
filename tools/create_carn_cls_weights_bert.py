from __future__ import print_function
import os
import sys
import json
import numpy as np
from bert_serving.client import BertClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_bert_embedding_init(object_list):
    bc = BertClient()
    encoded = bc.encode(object_list)
    print(encoded.shape)

    return encoded


if __name__ == '__main__':
    with open('data/object_name_list.txt') as f:
        content = f.readlines()
    object_list = [x.strip() for x in content]

    weights = create_bert_embedding_init(object_list)
    np.save('data/bert_init_imsitu_carn.npy', weights)
