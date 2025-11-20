import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model_GCN import GCNII_lyc

class MM_GCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph='full',n_speakers=2, modals=['a','v','l'], use_speaker=True, use_modal=False):
        super(MM_GCN, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph

        self.graph_net = GCNII_lyc(nfeat=n_dim, nlayers=nlayers, nhidden=nhidden, nclass=nclass,
                               dropout=dropout, lamda=lamda, alpha=alpha, variant=variant,
                               return_feature=return_feature, use_residue=use_residue)
        self.a_fc = nn.Linear(a_dim, n_dim)
        self.v_fc = nn.Linear(v_dim, n_dim)
        self.l_fc = nn.Linear(l_dim, n_dim)
        if self.use_residue:
            self.feature_fc = nn.Linear(n_dim*3+nhidden*3, nhidden)
        else:
            self.feature_fc = nn.Linear(nhidden * 3, nhidden)
        self.final_fc = nn.Linear(nhidden, nclass)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.a_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.v_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.l_spk_embs = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal # false

    def forward(self, a, v, l, dia_len, qmask):

        qmask = torch.cat([qmask[:x,i,:] for i,x in enumerate(dia_len)],dim=0)
        spk_idx = torch.argmax(qmask, dim=-1) 
        spk_emb_vector = self.speaker_embeddings(spk_idx) 

        if self.use_speaker: 
            if len(self.modals) == 1:
                a = a + spk_emb_vector
            else:
                if 'l' in self.modals:
                    l = l + spk_emb_vector
                if 'a' in self.modals:
                    a = a + spk_emb_vector
                if 'v' in self.modals:
                    v = v + spk_emb_vector

        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if len(self.modals) == 1:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            else:
                if 'a' in self.modals:
                    a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
                if 'v' in self.modals:
                    v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
                if 'l' in self.modals:
                    l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        adj = self.create_big_adj(a, v, l, dia_len, self.modals)

        if len(self.modals) == 3:
            features = torch.cat([a, v, l], dim=0).cuda()
        elif 'a' in self.modals and 'v' in self. modals:
            features = torch.cat([a, v], dim=0).cuda()
        elif 'a' in self.modals and 'l' in self.modals:
            features = torch.cat([a, l], dim=0).cuda()
        elif 'v' in self.modals and 'l' in self.modals:
            features = torch.cat([v, l], dim=0).cuda()
        else:
            features = a.cuda()

        features,layer_inners = self.graph_net(features, None, qmask, adj)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]
        if len(self.modals) == 3:
            features = torch.cat([features[:all_length], features[all_length:all_length * 2], features[all_length * 2:all_length * 3]], dim=-1)

        elif len(self.modals) == 2:
            features = torch.cat([features[:all_length], features[all_length:all_length * 2]], dim=-1)
        else:
            features = features[:all_length]
    
        if self.return_feature:
            if len(self.modals)>1:
                return features
            else:
                return features
        else:
            return F.softmax(self.final_fc(features), dim=-1)

    def create_big_adj(self, a, v, l, dia_len, modals): 
        modal_num = len(modals)
        all_length = l.shape[0] if len(l)!=0 else a.shape[0] if len(a) != 0 else v.shape[0]

        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).cuda()
        if len(modals) == 3:
            features = [a, v, l]
        elif 'a' in modals and 'v' in modals:
            features = [a, v]
        elif 'a' in modals and 'l' in modals:
            features = [a, l]
        elif 'v' in modals and 'l' in modals:
            features = [v, l]
        else:
            features = [a]
        start = 0

        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features): 
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i])) 
            for m in range(modal_num):
                for n in range(modal_num): 
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n: 
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m] 
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0] += m_start
                        idx[1] += n_start
                        adj[idx[0], idx[1]] = dia_sim

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj