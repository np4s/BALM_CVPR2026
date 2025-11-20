import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from model_GCN import TextCNN
from model_mm import MM_GCN

from BALM import FCM

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features

class DialogueGCNModel(nn.Module):

    def __init__(self, D_m, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes=7, dropout=0.5, no_cuda=False, alpha=0.2,use_residue=True,
                 D_m_v=512,D_m_a=100,modals='avl',att_type='gated',av_using_lstm=False, dataset='iemocap',
                 use_speaker=True, use_modal=False, multi_modal=True, dim_global=737):
        
        super(DialogueGCNModel, self).__init__()

        self.multi_modal=multi_modal
        self.no_cuda = no_cuda
        self.alpha = alpha
        self.dropout = dropout
        self.use_residue = use_residue
        self.return_feature = True
        self.modals = [x for x in modals]  # a, v, l
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        if self.att_type == 'gated' or self.att_type == 'concat_subsequently':
            self.av_using_lstm = av_using_lstm
        else:
            self.multi_modal = False
        self.use_bert_seq = False
        self.dataset = dataset
       
        if not self.multi_modal:
            if len(self.modals) == 3:
                hidden_ = 250
            elif ''.join(self.modals) == 'al':
                hidden_ = 150
            elif ''.join(self.modals) == 'vl':
                hidden_ = 150
            else:
                hidden_ = 100
            self.linear_ = nn.Linear(D_m, hidden_)
            self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        else:
            if 'a' in self.modals:
                hidden_a = 200
                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.lstm_a = nn.LSTM(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:
                hidden_v = 200
                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.lstm_v = nn.LSTM(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                hidden_l = 200 
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.lstm_l = nn.LSTM(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)


        self.window_past = window_past
        self.window_future = window_future

        self.graph_model = MM_GCN(a_dim=2*D_e, v_dim=2*D_e, l_dim=2*D_e, n_dim=2*D_e, nlayers=4, nhidden=graph_hidden_size, nclass=n_classes, 
                                  dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature, 
                                  use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals, 
                                  use_speaker=self.use_speaker, use_modal=self.use_modal)
            
        edge_type_mapping = {} 
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)

        self.edge_type_mapping = edge_type_mapping
        self.dropout_ = nn.Dropout(self.dropout)
        if self.multi_modal:    
            if self.att_type == 'concat_subsequently':
                self.smax_fc = nn.Linear(300 * len(self.modals), n_classes) if self.use_residue else nn.Linear(100 * len(self.modals), n_classes)
            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    self.smax_fc = nn.Linear(100*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear(100, n_classes)
            else:
                self.smax_fc = nn.Linear(2*D_e+graph_hidden_size*len(self.modals), n_classes)
        else:
            self.smax_fc = nn.Linear(2*D_e+graph_hidden_size*len(self.modals), n_classes)
            
        ##BALM
        self.fcm = FCM(D_m, D_m_a, D_m_v, dim_global=dim_global)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)


    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None, amask=None, vmask=None, lmask=None):
        ## BEGIN FCM
        U, U_a, U_v = self.fcm(U, U_a, U_v, lmask, amask, vmask)
        ## END FCM
        
        ## BEGIN BACKBONE
        if not self.multi_modal:
            U = self.linear_(U)
            emotions, hidden = self.lstm(U)
        else:
            if 'a' in self.modals:
                U_a = self.linear_a(U_a)
                if self.av_using_lstm:
                    emotions_a, hidden_a = self.lstm_a(U_a)
                else:
                    emotions_a = U_a
            if 'v' in self.modals:
                U_v = self.linear_v(U_v)
                if self.av_using_lstm:
                    emotions_v, hidden_v = self.lstm_v(U_v)
                else:
                    emotions_v = U_v
            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1,U.shape[-2],U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0],U.shape[1],-1)
                else:
                    U = self.linear_l(U)
                emotions_l, hidden_l = self.lstm_l(U)

        if not self.multi_modal:
            features = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
        else:
            if 'a' in self.modals:
                features_a = simple_batch_graphify(emotions_a, seq_lengths, self.no_cuda)
            else:
                features_a = []

            if 'v' in self.modals:
                features_v = simple_batch_graphify(emotions_v, seq_lengths, self.no_cuda)
            else:
                features_v = []

            if 'l' in self.modals:
                features_l = simple_batch_graphify(emotions_l, seq_lengths, self.no_cuda)
            else:
                features_l = []

        # MMGCN
        if self.multi_modal:
            emotions_feat = self.graph_model(features_a, features_v, features_l, seq_lengths, qmask)
        else:
            emotions_feat = self.graph_model(features, [], [], seq_lengths, qmask)
            
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        smax_feat = self.smax_fc(emotions_feat)
        if smax_feat.shape[-1] > 1:
            smax_feat = F.softmax(smax_feat, dim=1)
            log_prob = torch.log(smax_feat)
        else:
            log_prob = smax_feat
        ## END BACKBONE
        
        if self.training:
            return log_prob, smax_feat, [features_l, features_a, features_v]
        else:
            return log_prob