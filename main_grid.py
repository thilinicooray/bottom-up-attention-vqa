import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

from dataset import Dictionary, VQAGridDataset
import base_model
from train_grid import train
import utils
import utils_imsitu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0grid')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--imgset_dir', type=str, default='./data', help='Location of original images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    image_filenames = json.load(open("data/coco_img_file_mapping.json"))
    train_dset = VQAGridDataset(args.imgset_dir + '/train2014',image_filenames,'train', dictionary)
    eval_dset = VQAGridDataset(args.imgset_dir + '/val2014',image_filenames, 'val', dictionary)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    utils_imsitu.set_trainable(model.conv_net, False)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=3)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=3)
    train(model, train_loader, eval_loader, args.epochs, args.output)
