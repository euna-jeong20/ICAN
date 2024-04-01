import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
import random
import pdb



class HGNN_conv4(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, device, dropout_rate):
        super(HGNN_conv4, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layer = num_layer

        self.device = device
        
        self.weight_dict = self._init_weights()
        # self.activation = nn.PReLU()
        # self.activation = nn.ELU()
        # self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        # self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p = dropout_rate)

    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = nn.init.xavier_uniform_
        
        i = 0
        weight_dict['%d' %i] = nn.Parameter(initializer(torch.empty(self.input_dim, self.output_dim).to(self.device)))
        
        for i in range(1, self.num_layer + 1):
            weight_dict['%d' %i] = nn.Parameter(initializer(torch.empty(self.output_dim, self.output_dim).to(self.device)))
            
        return weight_dict
    
    def forward(self, input, coef_item_rep, coef_basket_rep):
        '''
        input: item emb
        '''

        x = input
        final_item_rep = []
        final_basket_rep = []
        
        final_item_rep.append(x.data)
        for i in range(self.num_layer):
            x = torch.sparse.mm(coef_basket_rep, x)

            basket_rep = x
            final_basket_rep.append(basket_rep)
            
            item_rep = torch.sparse.mm(coef_item_rep, basket_rep)
            final_item_rep.append(item_rep)

            x = item_rep
        
        final_basket_rep = torch.stack(final_basket_rep)
        final_item_rep = torch.stack(final_item_rep)
        
        final_basket_rep  = torch.mean(final_basket_rep, dim=0)
        final_item_rep = torch.mean(final_item_rep, dim=0)

        final_item_rep.to(self.device)
        final_basket_rep.to(self.device)

        return final_item_rep, final_basket_rep
    

class Mymodel(nn.Module):
    def __init__(self, num_user, num_item, args, device):
        super().__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.emb_dim = args.emb_dim
        self.num_layer = args.num_layer
        self.dropout_rate = args.dropout_rate
        self.pad_id = args.pad_id
        self.t_decay = args.t_decay

        self.device = device
        
        initializer = nn.init.xavier_uniform_

        self.conv = HGNN_conv4(self.emb_dim, self.emb_dim, self.num_layer, self.device, self.dropout_rate)

        self.user_emb = torch.empty(self.num_user, self.emb_dim).to(self.device)
        self.item_emb = nn.Embedding(self.num_item + 1, self.emb_dim, padding_idx= args.pad_id).to(self.device)
        self.final_user_emb = nn.Parameter(initializer(torch.empty(self.num_user, self.emb_dim).to(self.device)))
        self.g_user_emb = nn.Parameter(initializer(torch.empty(self.num_user, self.emb_dim).to(self.device)))

    def forward(self, log_seqs, batch_user_list, batch_len, repeatList):
        
        '''
        log_seqs [배치크기, max_seq, max_bsk]
        batch_user_list 배치안의 유저아이디 리스트
        batch_len 유저마다 바구니안 아이템의 개수 리스트 [[3, 5, 1, 2, 6, 2, 4, 10, 3, ...], [18, 1, 2, 10, 13, 16, 3, 18, 1, ...], [4, 2, 2, 3, 6, 3, 3],...]
        
        '''
        # count # of edges(baskets)
        num_edge_list  = [] 

        for num_basket in batch_len:
            num_edge_list.append(len(num_basket))
        
        # num_edge_list = [8, 9, ..., 12] len은 배치 수, 각 값은 총 소비한 바구니 개수
        
        num_edge = sum(num_edge_list)

        # hypergraph 만들기
        col = []
        num_item_in_bas1D = sum(batch_len, [])
        idx = 0
        for cnt in num_item_in_bas1D:
            for _ in range(cnt):
                col.append(idx)
            idx += 1

        row = log_seqs.reshape(-1)
        row = row[row != self.pad_id]
        data = np.ones(len(col))

        H = sp.coo_matrix((data, (row, col)), shape=(self.num_item, num_edge))

        zero = np.zeros(H.shape[1])
        zero = sp.coo_matrix(zero)
        H = sp.vstack([H, zero])

        # H로 부터 D^-1/2~~랑 B^-1/2 만들기 (계수)
        DD, BD = generate_G_from_H(H)
        DD = convert_sp_mat_to_sp_tensor(DD).to(self.device)
        BD = convert_sp_mat_to_sp_tensor(BD).to(self.device)

        # conv계산해서 item_rep와 basket_rep얻기
        item_rep, basket_rep = self.conv(self.item_emb.weight, DD, BD)

        #basket_rep에 미리 유저별로 알맞게 time_decay를 곱해 놓는거
        # start = 0
        # for cnt in num_edge_list:
        #     end = start + cnt
        #     bsk_user = basket_rep[start:end]
        #     t = torch.tensor([self.t_decay**(bsk_user.size(0) - i ) for i in range(1, bsk_user.size(0) + 1)]).to(self.device)
        #     bsk_user = bsk_user * t.unsqueeze(1)
        #     basket_rep[start:end] = bsk_user
        #     start = end


        # 유저가 소비한 basket별로 basket_rep를 평균해서 user_rep로 만들기
        user_rep = []
        start = 0
        for cnt in num_edge_list:
            # [start, end)
            end = start + cnt
            bsk_user = basket_rep[start:end]
            #t = torch.tensor([self.t_decay**(bsk_user.size(0) - i ) for i in range(1, bsk_user.size(0) + 1)]).to(self.device)
            #bsk_user = bsk_user * t.unsqueeze(1)
            avg = torch.mean(bsk_user, dim = 0)
            user_rep.append(avg)
            start = end
        user_rep = torch.stack(user_rep)

        # 유저가 소비한 basket별로 마지막 바구니를 제외하고 basket_rep를 평균해서 user_rep로 만들기
        # 유저가 소비한 마지막 바구니 rep로 배치Xd 행렬 만들기
        user_rep_for_loss = []
        pos_bsk = []
        neg_bsk = []
        all_idx = set(range(0, basket_rep.size(0)))
        start = 0
        for cnt in num_edge_list:
            end = start + cnt
            bsk_user = basket_rep[start:end-1]
            #t = torch.tensor([self.t_decay**(bsk_user.size(0) - i ) for i in range(1, bsk_user.size(0) + 1)]).to(self.device)
            #bsk_user = bsk_user * t.unsqueeze(1)
            bsk = basket_rep[end-1]
            avg = torch.mean(bsk_user, dim=0)

            prev_idx = set(range(start, end))
            neg_idx = random.sample(list(all_idx - prev_idx), 1)[0]
            neg = basket_rep[neg_idx]
             
            user_rep_for_loss.append(avg)
            pos_bsk.append(bsk)
            neg_bsk.append(neg)
            start = end
        user_rep_for_loss = torch.stack(user_rep_for_loss)
        pos_bsk = torch.stack(pos_bsk)
        neg_bsk = torch.stack(neg_bsk)



        # 그냥 대입
        self.user_emb[batch_user_list] = user_rep
        self.final_item_emb = nn.Parameter(item_rep)


        ###이제 loss계산
        
        ##1
        epsilon = 1e-8  # 아주 작은 값
        loss_1 = 0
        y_ui = torch.matmul(user_rep_for_loss, pos_bsk.t()).diag()
        y_uj = torch.matmul(user_rep_for_loss, neg_bsk.t()).diag()

        result = torch.sigmoid(y_ui - y_uj)
        result = torch.where(result == 0.0000e+00, epsilon, result)
        result = -torch.mean(torch.log(result))
        loss_1 += result 

        ##2
        loss_2 = 0
        loss_3 = 0

        all_item = set(row)      #배치 안에 있는 모든 아이템
        
        for user_idx, _ in enumerate(log_seqs):
            userbsk_len = batch_len[user_idx]
            userbsk_num = len(userbsk_len)
            
            previous_item = set(np.unique(log_seqs[user_idx][:userbsk_num]))
            
            pos_item_list = log_seqs[user_idx][userbsk_num-1]
            pos_item_list = pos_item_list[pos_item_list != self.pad_id]
            neg_item_list = random.sample(list(all_item - previous_item), len(pos_item_list))

            pos_item_rep = item_rep[pos_item_list]
            neg_item_rep = item_rep[neg_item_list]

            y_ui = torch.matmul(user_rep_for_loss[user_idx], pos_item_rep.t())
            y_uj = torch.matmul(user_rep_for_loss[user_idx], neg_item_rep.t())


            result = torch.sigmoid(y_ui - y_uj)
            result = torch.where(result == 0.0000, epsilon, result)
            result = -torch.mean(torch.log(result))
            loss_2 += result
            
            if not torch.isfinite(result):
                 pdb.set_trace()


            ####### 3 계산 
            
            # pos_item = list(set(np.unique(log_seqs[user_idx][:userbsk_num - 1])))         # 유저가 소비한 아이템 중 마지막 바구니 제외한 것들 
            # # pos_item_list = random.sample(list(pos_item), 5)
            # # print(len(all_item), len(previous_item), len(pos_item))
            # if len(all_item-previous_item) < len(pos_item):
            #     pos_item = random.sample(pos_item, len(all_item-previous_item))
            #     neg_item_list = list(all_item - previous_item)
            
            # else:
            #     neg_item_list = random.sample(list(all_item - previous_item), len(pos_item))
        

            # pos_item_rep = item_rep[pos_item]
            # neg_item_rep = item_rep[neg_item_list]

            # y_ui = torch.matmul(self.weight_dict['user_emb'][user_idx], pos_item_rep.t())
            # y_uj = torch.matmul(self.weight_dict['user_emb'][user_idx], neg_item_rep.t())

            # result = torch.sigmoid(y_ui - y_uj)
            # result = -torch.mean(torch.log(result))
            # loss_3 += result

        loss_2 = loss_2 / len(log_seqs)
        # loss_3 = loss_3 / len(log_seqs)
        l2norm = torch.sum(self.item_emb.weight**2) /2
        l2reg = 1e-4 * l2norm


        
        return loss_1, loss_2, l2reg        


    def predict(self, user):
        user_emb = self.user_emb[user]
        item_emb = self.final_item_emb[:-1]

        g_user_emb = self.g_user_emb[user]
        score = torch.matmul(user_emb, item_emb.t())
        # score = torch.matmul(user_emb, item_emb.t()) + torch.matmul(g_user_emb, item_emb.t())

        return score

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
    return res