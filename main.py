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
    # parser.add_argument('--dataset', default='tafeng.pickle')
    # parser.add_argument('--dataset', default='Dunnhumby.pkl')
    parser.add_argument('--dataset', default='RetailRocket.pkl')
    # parser.add_argument('--dataset', default='ValuedShopper.pkl')
    
    parser.add_argument('--model_path')
    # parser.add_argument('--train_dir', required=True)
    parser.add_argument('--n_times', default=3, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--t_decay', default=0.8, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--k',default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--isTrain', default=0, type=int)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--max_subseq_len', default=2, type=int)
    
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    dataset = DataLoader(args)

    
    repeat_ratio = []
    for user, last_bsk in enumerate(dataset.testList):
        last_bsk = list(set(last_bsk))
        if len(last_bsk) ==1:
            continue
        l = 0
        r = l+1
        cnt = 0
        while l < len(last_bsk) - 1:
            if r == len(last_bsk)-1:
                l +=1
                r = l+1
                continue
            if last_bsk[r] in dataset.co_purchase[last_bsk[l]]:
                cnt +=1
            r += 1

        num_comb = len(last_bsk)*(len(last_bsk)-1)/2
        repeat_ratio.append(cnt/ num_comb)
    
    
    def count_values_in_ranges(values, ranges):
        counts = [0] * len(ranges)
        for value in values:
            for i, r in enumerate(ranges):
                if r[0] <= value < r[1]:
                    counts[i] += 1
                    break
        return counts

    ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    counts = count_values_in_ranges(repeat_ratio, ranges)

    if args.isTrain:
        args.pad_id = dataset.numItemsTrain
    else:
        args.pad_id = dataset.numItemsTest
    
    print('start training')
    trainer(dataset, args)


'''
model코드 채워넣기, evaluation을 고쳐

'''