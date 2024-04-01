import numpy as np
import scipy.sparse as sp
import random
import torch
import pdb
from collections import Counter

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_batch(dataLoader, args, batch_id, subseq_cnt):
    
    samples, sampleLen, val = dataLoader.batchLoader(batch_id, args.isTrain)
    subseq_idx = subseq_cnt + args.max_subseq_len + 1

    max_bsk_list = []
    # max_seq = max([len(user_len) for user_len in sampleLen[subseq_cnt: args.max_subseq_len]])  # max length of sequence
    max_seq = args.max_subseq_len + 1
    for user_len in sampleLen:
        if len(user_len[subseq_cnt: subseq_idx]) == 0:
            continue
        max_bsk_list.append(max(user_len[subseq_cnt: subseq_idx]))
    max_bsk = max(max_bsk_list)  # max length of basket
    
    '''
    len(userLen)으로 유저의 seq 길이를 재고(몇개의 바구니를 소비했냐) 그걸 list로 만들어서 max를 해서 제일 긴 seq의 길이를 저장
    max(userLen)으로 유저마다 제일 긴 바구니안 아이템 개수를 list로 만들어서 max를 해서 제일 큰 바구니의 크기를 저장
    '''
    paddedSamples = []
    repeatList = []
    lenList = []
    check = 0
    remove_user = []

    batch_len = [user_len[subseq_cnt: subseq_idx] for user_len in sampleLen]

    for user_idx, user in enumerate(samples):
        paddedU = []

        if len(batch_len[user_idx]) < max_seq:
            remove_user.append(user_idx)
            continue

         
        # flattenedList = sum(user, [])
        # counts = Counter(flattenedList)
        # repeat_item = [num for num, count in counts.items() if count >= 2]
        # if len(repeat_item) < 1:
        #     repeatList.append(user[-1])
        # else:
        #     repeatList.append(repeat_item)

        user = user[subseq_cnt: subseq_idx]
        
        for eachBas in user:
            
            paddedBas = eachBas + [args.pad_id] * (max_bsk - len(eachBas))
            paddedBas = np.array(paddedBas)
            paddedU.append(paddedBas)  # [batch, maxLenBas]

        paddedU = paddedU + [[args.pad_id] * max_bsk] * (max_seq - len(paddedU))
        paddedU = np.array(paddedU)
        

        if paddedU.shape != (max_seq, max_bsk):
            print('error')
        paddedSamples.append(paddedU)  # [batch, maxLenSeq]

    if len(remove_user) > 0:
        batch_len = [batch_len[i] for i in range(len(batch_len)) if i not in remove_user]
        batch_id = [batch_id[i] for i in range(len(batch_id)) if i not in remove_user]

    
    # pad_batch.shape == [배치크기, max_seq, max_bsk]
    return np.array(paddedSamples), batch_len, batch_id, val, repeatList