import os
import time
import random
import argparse
import numpy as np
import math
import pdb
import gc

import torch
import torch.nn as nn
from model import *
from utils import get_batch
from evaluation import *


def trainer(dataloader, args):

    if args.device == 'cuda' :
        device = torch.device('cuda')

    if args.isTrain:
        numUsers = dataloader.numTrain
        numItems = dataloader.numItemsTrain
    else:
        numUsers = dataloader.numTrainVal
        numItems = dataloader.numItemsTest

    if numUsers % args.batch_size == 0:
        num_batches = numUsers // args.batch_size
    else:
        num_batches = numUsers // args.batch_size + 1
    
    idxList = [i for i in range(numUsers)]
    model = Mymodel(numUsers, numItems, args, device)

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0
    for epoch in range(args.num_epochs):
        timeEpStr = time.time()
        loss_val = 0
        loss1_val = 0
        loss2_val = 0
        loss3_val = 0

        for batchID in range(num_batches):
            start = args.batch_size * batchID
            end = min(numUsers, start + args.batch_size)

            batchList = idxList[start:end]

            
            i = 0
            while True:
                log_seqs, batch_len, batch_id, _, repeatList = get_batch(dataloader, args, batchList, i)

                if len(batch_id) < 5:
                    break

                loss1, loss2, loss3 = model(log_seqs, batch_id, batch_len, repeatList)
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
                loss1_val += loss1.item()
                loss2_val += loss2.item()
                # loss3_val += loss3.item()

                i += 1
                del log_seqs
                del batch_len
                del batch_id
                del loss1
                del loss2
                del loss
                torch.cuda.empty_cache()
                           


        loss_val /= num_batches
        loss1_val /= num_batches
        loss2_val /= num_batches
        # loss3_val /= num_batches    
        
        timeEpEnd = time.time()

        #eval
        timeEvStr = time.time()
        print('evaluation')
        ndcg, recall, phr = evaluation(model, dataloader, args, device)
        timeEvEnd = time.time()
        
        print(f'epoch: {epoch}, train loss: {loss1_val, loss2_val, loss3_val}, 내가 사용하는 loss: {loss_val}, time: {timeEpEnd - timeEpStr}')
        # print(f'ndcg@{args.k}: {round(ndcg, 4)}, recall@{args.k}: {round(recall, 4)}, PHR@{args.k}: {round(phr, 4)}, time: {timeEvEnd - timeEvStr}')
        print(f'ndcg@{args.k}: {round(ndcg[1], 4)}, recall@{args.k}: {round(recall[1], 4)}, PHR@{args.k}: {round(phr[1], 4)}, time: {timeEvEnd - timeEvStr}')
        

        metric = ndcg[1] + recall[1] + phr[1]
        if best < metric:
            best = metric
            torch.save(model, 'model.pth')
