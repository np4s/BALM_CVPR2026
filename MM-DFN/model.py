import torch
import torch.nn as nn
import torch.nn.functional as F
from model_GCN import TextCNN
from model_mm import MM_GCN
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from BALM import FCM

def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features


class DialogueGNNModel(nn.Module):

    def __init__(self, D_m, D_e, graph_hidden_size, n_speakers, n_classes=7, dropout=0.5, no_cuda=False, alpha=0.1, lamda=0.5, use_residue=True, modal_weight=1.0,
                 D_m_v=512, D_m_a=100, modals='avl', att_type='gated', av_using_lstm=False, Deep_GCN_nlayers=64, dataset='iemocap',
                 use_speaker=True, use_modal=False, reason_flag=False, multi_modal=True, use_crn_speaker=False, speaker_weights='1-1-1', dim_global=737):

        super(DialogueGNNModel, self).__init__()

        self.no_cuda = no_cuda
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = dropout
        self.use_residue = use_residue
        self.return_feature = True
        self.modals = [x for x in modals]  # [a, v, l]
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.reason_flag = reason_flag
        self.multi_modal = multi_modal
        self.n_speakers = n_speakers
        self.use_crn_speaker = use_crn_speaker
        self.speaker_weights = list(map(float, speaker_weights.split('-')))
        self.modal_weight = modal_weight

        if self.att_type in ['gated', 'concat_subsequently', 'mfn', 'mfn_only', 'tfn_only', 'lmf_only', 'concat_only']:
            # multi_modal = True
            self.av_using_lstm = av_using_lstm
        else:
            # concat
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
            self.lstm = nn.GRU(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.GRU(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        else:   
            hidden_a = 200
            hidden_v = 200
            hidden_l = 200

            if 'a' in self.modals:

                self.linear_a = nn.Linear(D_m_a, hidden_a)
                if self.av_using_lstm:
                    self.lstm_a = nn.GRU(input_size=hidden_a, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'v' in self.modals:

                self.linear_v = nn.Linear(D_m_v, hidden_v)
                if self.av_using_lstm:
                    self.lstm_v = nn.GRU(input_size=hidden_v, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            if 'l' in self.modals:
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=hidden_l)
                else:
                    self.linear_l = nn.Linear(D_m, hidden_l)
                self.lstm_l = nn.GRU(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.GRU(input_size=hidden_l, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        # GDF
        self.graph_model = MM_GCN(a_dim=2 * D_e, v_dim=2 * D_e, l_dim=2 * D_e, n_dim=2 * D_e, nlayers=Deep_GCN_nlayers, nhidden=graph_hidden_size,
                                    nclass=n_classes, dropout=self.dropout, lamda=self.lamda, alpha=self.alpha, variant=True,
                                    return_feature=self.return_feature, use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals,
                                    use_speaker=self.use_speaker, use_modal=self.use_modal, reason_flag=self.reason_flag, modal_weight=self.modal_weight,
                                    )

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
                    self.smax_fc = nn.Linear(100 * len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear(100, n_classes)
            elif self.att_type in ['mfn', 'mfn_only']:
                from model_fusion import MFN
                self.mfn = MFN()
                self.smax_fc = nn.Linear(400, n_classes)
            elif self.att_type in ['tfn_only']:
                from model_fusion import TFN
                self.tfn = TFN()
                self.smax_fc = nn.Linear(300, n_classes)
            elif self.att_type in ['lmf_only']:
                from model_fusion import LMF
                self.lmf = LMF()
                self.smax_fc = nn.Linear(300, n_classes)
            elif self.att_type in ['concat_only']:
                self.smax_fc = nn.Linear(900, n_classes)
            else:
                self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size * len(self.modals), n_classes)
        else:
            self.smax_fc = nn.Linear(2 * D_e + graph_hidden_size * len(self.modals), n_classes)
        
        ## BALM
        self.fcm = FCM(D_m, D_m_a, D_m_v, dim_global=dim_global)

    def _reverse_seq(self, X, mask):
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, seq_lengths, U_a=None, U_v=None, amask=None, vmask=None, lmask=None):
        emotions_a, emotions_v, emotions_l, features_a, features_v, features_l = None, None, None, None, None, None
        ## BEGIN FCM
        U, U_a, U_v = self.fcm(U, U_a, U_v, lmask, amask, vmask)
        ## END FCM
        
        ## BEGIN BACKBONE
        if not self.multi_modal:
                U = self.linear_(U)
                emotions, hidden = self.lstm(U)

                # TODO
                if self.use_crn_speaker:
                    # (32,21,200) (32,21,9)
                    U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
                    U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
                    U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                    for b in range(U_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                    E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]  # lstm
                    for b in range(U_p_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                    # (21,32,200)
                    U_p = U_p_.transpose(0, 1)
                    emotions = emotions + self.speaker_weights[2] * U_p
        else:
            if 'a' in self.modals:
                # (21.32,200)
                U_a = self.linear_a(U_a)
                emotions_a = U_a
                if self.av_using_lstm:
                    emotions_a, hidden_a = self.lstm_a(U_a)

                if self.use_crn_speaker:

                    # (32,21,200) (32,21,9)
                    U_, qmask_ = U_a.transpose(0, 1), qmask.transpose(0, 1)
                    U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_a.type())
                    U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                    for b in range(U_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                    E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                    for b in range(U_p_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                    # (21,32,200)
                    U_p = U_p_.transpose(0, 1)
                    emotions_a = emotions_a + self.speaker_weights[0] * U_p

            if 'v' in self.modals:
                # (21.32,200)
                U_v = self.linear_v(U_v)
                emotions_v = U_v
                if self.av_using_lstm:
                    emotions_v, hidden_v = self.lstm_v(U_v)

                # TODO

                if self.use_crn_speaker:
                    # (32,21,200), (32,21,9)
                    U_, qmask_ = U_v.transpose(0, 1), qmask.transpose(0, 1)
                    U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U_v.type())
                    U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                    for b in range(U_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                    E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                    for b in range(U_p_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                    # (21,32,200)
                    U_p = U_p_.transpose(0, 1)

                    emotions_v = emotions_v + self.speaker_weights[1] * U_p

            if 'l' in self.modals:
                if self.use_bert_seq:
                    U_ = U.reshape(-1, U.shape[-2], U.shape[-1])
                    U = self.txtCNN(U_).reshape(U.shape[0], U.shape[1], -1)
                else:
                    # (21.32,200)
                    U = self.linear_l(U)

                # (21,32,200)
                emotions_l, hidden_l = self.lstm_l(U)

                if self.use_crn_speaker:
                    # (32,21,200), (32,21,9)
                    U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
                    U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
                    U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
                    for b in range(U_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0:
                                U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

                    E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

                    for b in range(U_p_.size(0)):
                        for p in range(len(U_parties_)):
                            index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                            if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                    # (21,32,200)
                    U_p = U_p_.transpose(0, 1)

                    emotions_l = emotions_l + self.speaker_weights[2] * U_p
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

        if self.multi_modal:
            emotions_feat, e_l, e_a= self.graph_model(features_a, features_v, features_l, seq_lengths, qmask)
        else:
            emotions_feat= self.graph_model(features, [], [], seq_lengths, qmask)

        if self.att_type == 'mfn':
            emotions_tmp = emotions_feat
            input_conversation_length = torch.tensor(seq_lengths)
            start_zero = input_conversation_length.data.new(1).zero_()
            if torch.cuda.is_available():
                input_conversation_length = input_conversation_length.cuda()
                start_zero = start_zero.cuda()
            max_len = max(seq_lengths)
            start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
            # (77,32,3*300)
            emotions_tmp = torch.stack(
                [pad(emotions_tmp.narrow(0, s, l), max_len, False)
                    for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())
                    ], 0).transpose(0, 1)
            # (77,32,400) << (77,32,900)
            emotions_feat_ = self.mfn(emotions_tmp)

            emotions_feat = []
            batch_size = emotions_feat_.size(1)
            for j in range(batch_size):
                emotions_feat.append(emotions_feat_[:seq_lengths[j], j, :])
            node_features = torch.cat(emotions_feat, dim=0)
            if torch.cuda.is_available():
                emotions_feat = node_features.cuda()

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