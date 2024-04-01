import torch
import numpy as np
import math
from utils import get_batch
import gc 


def evaluation(model, dataLoader, args, device):

    if args.isTrain:
        numUser = dataLoader.numValid
    else:
        numUser = dataLoader.numTest

    if numUser % args.batch_size == 0:
        numBatch = numUser // args.batch_size
    else:
        numBatch = numUser // args.batch_size + 1

    idxList = [i for i in range(numUser)]

    Recall = []
    NDCG = []
    PHR = []

    for batch in range(numBatch):
        start = batch * args.batch_size
        end = min(batch * args.batch_size + args.batch_size, numUser)

        batchList = idxList[start:end]

        _, _, _, val, _ = get_batch(dataLoader, args, batchList, 0)

        with torch.no_grad():
            score = model.predict(batchList)
        
        _, predIdx = torch.topk(score, k=30)

        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList = val
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList += val

    for k in [5, 10]:
        recall = calRecall(targetList, predIdxArray, k)
        Recall.append(recall)
        NDCG.append(calNDCG(targetList, predIdxArray, k))
        PHR.append(calPHR(targetList, predIdxArray, k))
    
    del val
    del idxList
    del score
    del targetList
    del predIdx
    del predIdxArray
    torch.cuda.empty_cache()
    
    

    return Recall, NDCG, PHR


def calRecall(target, pred, k):
    assert len(target) == len(pred)
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        ptar = set(pred[i][:k])

        if len(gt) == 0:
            print('Error')

        sumRecall += len(gt & ptar) / float(len(gt))

    recall = sumRecall / float(len(target))

    return recall

def calPHR(gt, pred, k):
    PHR = []
    for i in range(len(gt)):
        gt_bsk = set(gt[i])
        pred_bsk = set(pred[i][:k])
        if len(gt_bsk & pred_bsk) != 0:
            PHR.append(1)
        else:
            PHR.append(0)

    phr = np.mean(PHR)

    return phr


def calNDCG(target, pred, k):
    assert len(target) == len(pred)
    sumNDCG = 0
    for i in range(len(target)):
        valK = min(k, len(target[i]))
        gt = set(target[i])
        idcg = calIDCG(valK)
        dcg = sum([int(pred[i][j] in gt) / math.log(j + 2, 2) for j in range(k)])
        sumNDCG += dcg / idcg

    return sumNDCG / float(len(target))


# the gain is 1 for every hit, and 0 otherwise
def calIDCG(k):
    return sum([1.0 / math.log(i + 2, 2) for i in range(k)])