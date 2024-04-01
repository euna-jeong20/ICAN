import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from collections import defaultdict
import random
import pickle

class DataLoader():
    def __init__(self, args):
        root = './data/' + args.dataset
        with open(root, 'rb') as f:
            datadict = pickle.load(f)

        user2item = self.generate_user_list(datadict)

        numRemove, user2idx = self.generate_sets(user2item)
        self.numItems = self.get_num_items() + 1

        print("num_users_removed   %d" % numRemove)
        print("num_valid_users   %d" % len(self.allList))
        print("num_items   %d" % self.numItems)


        self.numTrain, self.numValid, self.numTrainVal, self.numTest = len(self.testList), len(self.testList), len(
            self.testList), len(self.testList)      # 다 똑같은 유저 개수
        self.numItemsTrain, self.numItemsTest = self.numItems, self.numItems
        
        self.valid2train = {}
        self.test2trainVal = {}
        for i in range(len(self.trainList)):
            self.valid2train[i] = i
            self.test2trainVal[i] = i

        if args.isTrain:
            self.lenTrain = self.generateLens(self.trainList)
            # self.lenVal = self.generateLens(self.validList)
        else:  # Test
            self.lenTrainVal = self.generateLens(self.trainValList)
            # self.lenTest = self.generateLens(self.allList)        


    def generate_user_list(self, dataDict):
        all_users = list(dataDict.keys())
        user2item = {}
        for user in all_users:
            user2item[user] = dataDict.get(user, [])        # user라는 key가 없으면 []를 return한다.
        return user2item

    def generate_sets(self, user2item):
        self.trainList = []
        self.validList = []
        self.trainValList = []
        self.testList = []
        self.allList = []
        count = 0
        count_remove = 0
        user2idx = {}

        for user in user2item:
            if len(user2item[user]) <= 5:  # train>=3, valid=1, test=1
                count_remove += 1
                continue
            user2idx[user] = count
            count += 1
            # 원본
            # self.trainList.append(user2item[user][:-2])
            # self.validList.append(user2item[user][:-1])
            # self.trainValList.append(user2item[user][:-1])
            # self.testList.append(user2item[user])
            self.trainList.append(user2item[user][:-2])
            self.validList.append(user2item[user][-2])
            self.trainValList.append(user2item[user][:-1])
            self.testList.append(user2item[user][-1])
            self.allList.append(user2item[user])
        return count_remove, user2idx

    def get_num_items(self):
        numItem = 0
        for baskets in self.allList:
            # all the baskets of users
            for basket in baskets:
                for item in basket:
                    numItem = max(item, numItem)

        return numItem
    

    def generateLens(self, userList):
        # list of list of lens of baskets
        lens = []
        # pre-calculate the len of each sequence and basket
        for user in userList:
            lenUser = []
            # the last bas is the traget to calculate errors
            for bas in user:
                lenUser.append(len(bas))
            lens.append(lenUser)
        # lens의 형태 [[3, 5, 1, 2, 6, 2, 4, 10, 3, ...], [18, 1, 2, 10, 13, 16, 3, 18, 1, ...], [4, 2, 2, 3, 6, 3, 3],...]
        return lens
    

    # def batchLoader(self, batchIdx, isTrain, isEval):
    #     if isTrain and not isEval:      # 1 0
    #         train = [self.trainList[idx] for idx in batchIdx]
    #         trainLen = [self.lenTrain[idx] for idx in batchIdx]

    #     elif isTrain and isEval:        # 1 1
    #         train = [self.validList[idx] for idx in batchIdx]
    #         # trainLen = [self.lenVal[idx] for idx in batchIdx]
    #         trainLen = None

    #     elif not isTrain and not isEval:    # 0 0
    #         train = [self.trainValList[idx] for idx in batchIdx]
    #         trainLen = [self.lenTrainVal[idx] for idx in batchIdx]

    #     else:   # 0 1
    #         train = [self.testList[idx] for idx in batchIdx]
    #         # trainLen = [self.lenTest[idx] for idx in batchIdx]
    #         trainLen = None


    #     return train, trainLen


    def batchLoader(self, batchIdx, isTrain):
        if isTrain:      # 1 0
            train = [self.trainList[idx] for idx in batchIdx]
            trainLen = [self.lenTrain[idx] for idx in batchIdx]
            val = [self.validList[idx] for idx in batchIdx]
            # trainLen = [self.lenVal[idx] for idx in batchIdx]
            

        elif not isTrain:    # 0 
            train = [self.trainValList[idx] for idx in batchIdx]
            trainLen = [self.lenTrainVal[idx] for idx in batchIdx]
            val = [self.testList[idx] for idx in batchIdx]
            # trainLen = [self.lenTest[idx] for idx in batchIdx]
            


        return train, trainLen, val