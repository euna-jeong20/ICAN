import os
import time
import random
import argparse
import numpy as np
import math
import pdb

import torch
import torch.nn as nn


from utils import *
from data_loader import *
from train import trainer

# from test import eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tafeng.pickle')
    parser.add_argument('--model_path')
    # parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--t_decay', default=0.8, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--k',default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--isTrain', default=1, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--max_subseq_len', default=3, type=int)
    
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = DataLoader(args)
    
    if args.isTrain:
        args.pad_id = dataset.numItemsTrain
    else:
        args.pad_id = dataset.numItemsTest
    
    trainer(dataset, args)


'''
model코드 채워넣기, evaluation을 고쳐

'''