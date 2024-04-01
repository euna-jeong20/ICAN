import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


import os, sys
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
from collections import defaultdict

def preprocessing():

    df = pd.read_csv('/home/euna/hyperr/data/tafeng.csv')
    df['TRANSACTION_DT'] = pd.to_datetime(df['TRANSACTION_DT'])
    df = df[['TRANSACTION_DT', 'CUSTOMER_ID', 'PRODUCT_ID']]
    while True:
        usernum = len(df['CUSTOMER_ID'].unique())
        itemnum = len(df['PRODUCT_ID'].unique())

        while True:
            count_customer = df['CUSTOMER_ID'].value_counts()
            del_customer = count_customer[count_customer < 10].index
            lc = len(del_customer)
        
            if lc != 0:
                mask_customer = df['CUSTOMER_ID'].isin(del_customer)
                df = df[~mask_customer]
        


            count_product = df['PRODUCT_ID'].value_counts()
            del_product = count_product[count_product < 10].index
            lp = len(del_product)

            if lp != 0:
                mask_product = df['PRODUCT_ID'].isin(del_product)
                df = df[~mask_product] 
            elif lc == 0:
                break
        
        # df2 = df.groupby(['TRANSACTION_DT', 'CUSTOMER_ID'])['PRODUCT_ID'].apply(list).reset_index()
        # df2.rename(columns={'PRODUCT_ID' : 'bsk'}, inplace=True)
        # df2['product_count'] = df2['bsk'].apply(len)

        # filtered_df = df2[df2['product_count'] <= 1]
        # if len(filtered_df) != 0:

        #     date_list = filtered_df['TRANSACTION_DT'].tolist()
        #     user_list = filtered_df['CUSTOMER_ID'].tolist()

        #     df = df[~((df['TRANSACTION_DT'].isin(date_list)) & (df['CUSTOMER_ID'].isin(user_list)))]    

        df2 = df.groupby(['TRANSACTION_DT', 'CUSTOMER_ID'])['PRODUCT_ID'].apply(list).reset_index()
        df2.rename(columns={'PRODUCT_ID' : 'bsk'}, inplace=True)    
        result = []
        for customer_id, group in df2.sort_values(by='TRANSACTION_DT').groupby('CUSTOMER_ID'):
            merged_list = [bsk for bsk in group['bsk']]    
            result.append([customer_id, merged_list])

        df3 = pd.DataFrame(result, columns=['CUSTOMER_ID', 'bsks'])

        df3['bsk_count'] = df3['bsks'].apply(len)
        
        filtered_df2 = df3[df3['bsk_count'] <= 3]
        if len(filtered_df2) != 0:
            user_list = filtered_df2['CUSTOMER_ID'].tolist()

            df = df[~(df['CUSTOMER_ID'].isin(user_list))]

        unum = len(df['CUSTOMER_ID'].unique())
        inum = len(df['PRODUCT_ID'].unique())
            
        if usernum == unum and itemnum == inum:
            break

    id_map = {key: i for i, key in enumerate(df['CUSTOMER_ID'].unique())}
    df['CUSTOMER_ID'] = df['CUSTOMER_ID'].map(id_map)

    id_map = {key: i for i, key in enumerate(df['PRODUCT_ID'].unique())}
    df['PRODUCT_ID'] = df['PRODUCT_ID'].map(id_map)
    
    df2 = df.groupby(['TRANSACTION_DT', 'CUSTOMER_ID'])['PRODUCT_ID'].apply(list).reset_index()
    df2.rename(columns={'PRODUCT_ID' : 'bsk'}, inplace=True)

    result = []
    for customer_id, group in df2.sort_values(by='TRANSACTION_DT').groupby('CUSTOMER_ID'):
        merged_list = [bsk for bsk in group['bsk']]    
        result.append([customer_id, merged_list])

    result_df = pd.DataFrame(result, columns=['user_id', 'bsks'])



            
    return result_df  





if __name__ == "__main__":
    df = preprocessing()
    df.to_csv('my_tafeng.csv')
    data_dict = dict(zip(df['user_id'], df['bsks']))

    with open('tafeng.pickle', 'wb') as f:
        pickle.dump(data_dict, f)
