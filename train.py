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

    row = []
    col = []
    for key, value in dataloader.co_purchase.items():
        row += [key] * len(value)
        col += value
    data = np.ones(len(col))
    item_H = sp.coo_matrix((data, (row, col)), shape=(dataloader.numItems, dataloader.numItems))
    zero = np.zeros(item_H.shape[1])
    zero = sp.coo_matrix(zero)
    item_H = sp.vstack([item_H, zero])
    # DD, BD = generate_G_from_H(item_H)
    # i_DD = convert_sp_mat_to_sp_tensor(DD).to(device)
    # i_BD = convert_sp_mat_to_sp_tensor(BD).to(device)
    # co_item = i_DD, i_BD

    # random.shuffle(idxList)
    model = Mymodel(numUsers, numItems, args, device)
    # model = torch.load('model.pth')

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0


    
    

    for epoch in range(args.num_epochs):
        # random.shuffle(idxList)
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

                # loss1, loss2, loss3 = model(log_seqs, batch_id, batch_len, repeatList, co_item)
                loss1, loss2, loss3 = model(log_seqs, batch_id, batch_len, repeatList, item_H)
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
                loss1_val += loss1.item()
                loss2_val += loss2.item()
                loss3_val += loss3.item()

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
        recall, ndcg, phr = evaluation(model, dataloader, args, device)
        timeEvEnd = time.time()
        
        print(f'epoch: {epoch}, train loss: {loss1_val, loss2_val, loss3_val}, 내가 사용하는 loss: {loss_val}, time: {timeEpEnd - timeEpStr}')
        # print(f'ndcg@{args.k}: {round(ndcg, 4)}, recall@{args.k}: {round(recall, 4)}, PHR@{args.k}: {round(phr, 4)}, time: {timeEvEnd - timeEvStr}')
        print(f'ndcg@{args.k}: {round(ndcg[1], 4)}, recall@{args.k}: {round(recall[1], 4)}, PHR@{args.k}: {round(phr[1], 4)}, time: {timeEvEnd - timeEvStr}')
        print(f'ndcg@5: {round(ndcg[0], 4)}, recall@5: {round(recall[0], 4)}, PHR@5: {round(phr[0], 4)}, time: {timeEvEnd - timeEvStr}')
        

        metric = ndcg[1] + recall[1] + phr[1]
        # if best < metric:
        #     best = metric
        #     torch.save(model, './model' + f'model_{epoch}_{metric}.pth')












def generate_G_from_H(H):

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.array(H.sum(1)) 
    # the degree of the hyperedge
    DE = np.array(H.sum(0))

    invDE2 = sp.diags(np.power(DE, -0.5).flatten())     #엣지^(-1/2)
    DV2 =  sp.diags(np.power(DV, -0.5).flatten())       #노드^(-1/2)

    # invDE2 = sp.diags(np.power(DE, -1).flatten())     #엣지^(-1)
    # DV2 =  sp.diags(np.power(DV, -1).flatten())       #노드^(-1/2)

    W = sp.diags(W)
    HT = H.T


    invDE_HT_DV2 = invDE2 * HT * DV2            #엣지 업데이트할때 필요 B H.t D
    # G = DV2 * H * W * invDE2 * invDE_HT_DV2     #노드 업데이트할때 필요 D H B B H.t D
    G = DV2 * H * W * invDE2     #노드 업데이트할때 필요 D H B

    # invDE_HT_DV2 = invDE2 * HT            #엣지 업데이트할때 필요
    # G = DV2 * H      #노드 업데이트할때 필요

    del W
    del DV
    del DE
    del invDE2
    del DV2
    del HT
    
    

    return G, invDE_HT_DV2


def convert_sp_mat_to_sp_tensor(X):
    """
    Convert scipy sparse matrix to PyTorch sparse matrix

    Arguments:
    ----------
    X = Adjacency matrix, scipy sparse matrix
    """
    coo = X.tocoo().astype(np.float16)
    i = torch.LongTensor(np.mat([coo.row, coo.col]))
    v = torch.FloatTensor(coo.data)
    res = torch.sparse.FloatTensor(i, v, coo.shape)

    del coo
    del i
    del v
    
    

    return res