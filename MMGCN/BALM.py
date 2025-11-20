import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCM(nn.Module):
    def __init__(self, D_m, D_m_a, D_m_v, dim_global):
        super(FCM, self).__init__()
        self.cz = dim_global
        self.wg = nn.Linear(D_m + D_m_a + D_m_v, self.cz)
        self.wa = nn.Linear(self.cz, D_m_a)
        self.wv = nn.Linear(self.cz, D_m_v)
        self.wl = nn.Linear(self.cz, D_m)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def global_avg_pool_with_lengths(self, x, mask):
        """
        Global average pooling that ignores padded values using sequence lengths
        x: (seq_length, batch_size, dim)
        mask: (seq_length, batch_size, dim) - binary mask indicating valid positions
        """
        batch_size = x.shape[1]
        dim = x.shape[2]

        avg_x = torch.zeros(batch_size, dim, device=x.device)
        for i in range(batch_size):
            avg_x[i] = (x[:, i, :] * mask[:, i, :]).sum(dim=0) / \
                (mask[:, i, :].sum(dim=0) + 1e-8)

        return avg_x

    def forward(self, l, a, v, lmask, amask, vmask):
        """
        Forward pass with sequence lengths
        l, a, v: (seq_length, batch_size, dim)
        length: list or tensor of actual sequence lengths for each batch
        """
        # Global average pooling using sequence lengths
        sa = self.global_avg_pool_with_lengths(a, amask)  # (batch_size, dim_a)
        sv = self.global_avg_pool_with_lengths(v, vmask)  # (batch_size, dim_v)
        sl = self.global_avg_pool_with_lengths(l, lmask)  # (batch_size, dim_l)

        # Concatenate and pass through FC + ReLU
        z = torch.cat((sl, sa, sv), dim=1)  # (batch_size, cl + ca + cv)
        z = self.wg(z)  # (batch_size, cz)
        z = self.relu(z)

        # Generate weights
        ea = self.wa(z)  # (batch_size, ca)
        ev = self.wv(z)  # (batch_size, cv)
        el = self.wl(z)  # (batch_size, cl)

        # Expand to match input dimensions: (batch_size, dim) -> (seq_length, batch_size, dim)
        ea_expanded = ea.unsqueeze(0).expand(a.shape[0], -1, -1)
        ev_expanded = ev.unsqueeze(0).expand(v.shape[0], -1, -1)
        el_expanded = el.unsqueeze(0).expand(l.shape[0], -1, -1)

        # Apply attention: 2 * sigmoid(e) * original
        a_new = 2 * self.sigmoid(ea_expanded) * a
        v_new = 2 * self.sigmoid(ev_expanded) * v
        l_new = 2 * self.sigmoid(el_expanded) * l

        a_new = a_new * amask
        v_new = v_new * vmask
        l_new = l_new * lmask

        return l_new, a_new, v_new


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier, self).__init__()
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.regression = out_dim == 1  # for regression task, e.g. mosei

    def forward(self, x):
        x = F.relu(self.proj1(x))
        x = self.out_layer(x)
        if not self.regression:
            x = F.log_softmax(x, dim=-1)
        x = x.squeeze(-1)
        # for regression task return unprocessed output of FC, cls return log_softmax
        return x


class GRM(nn.Module):
    def __init__(self, hidden_dim, output_dim, modals, rho=1.3, enc_l=200, enc_a=200, enc_v=200, start_epoch=-1, end_epoch=100, normalize=False):
        """
            enc_l (int, optional): Lexical embedding dim of backbone.
            enc_a (int, optional): Audio embedding dim of backbone.
            enc_v (int, optional): Visual embedding dim of backbone.
            Default set to hidden_a, hidden_v (see inside model.py).
        """
        super(GRM, self).__init__()
        self.modals = modals
        self.rho = rho
        self.regression = output_dim == 1  # for regression task, e.g. mosei
        assert self.modals[0] == 'a' and self.modals[1] == 'v' and self.modals[2] == 'l'

        self.output_dim = output_dim

        self.classifier_l = Classifier(enc_l, hidden_dim, output_dim)
        self.classifier_a = Classifier(enc_a, hidden_dim, output_dim)
        self.classifier_v = Classifier(enc_v, hidden_dim, output_dim)

        self.uni_prob = list()
        self.prev_kl = None
        self.loss_gm = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.normalize = normalize

    def cal_kl(self, smax_feat, uni_prob):
        # if cls -> smax_feat is softmax shape (sum(length), n_classes)
        # if regression -> smax_feat is unprocessed of FC shape (sum(length), 1) -> to distrubution using sigmoid
        # classification KL(log_softmax, softmax), regression KL(log_sigmoid, sigmoid)
        kl_fn = nn.KLDivLoss(reduction='batchmean')
        kl_list = list()
        smax_feat = smax_feat.clone().detach()
        smax_feat = smax_feat.squeeze(-1)

        if self.regression:
            smax_feat = torch.stack(
                [F.sigmoid(-smax_feat), F.sigmoid(smax_feat)], -1)
            assert smax_feat.shape[-1] == 2

        for pred in uni_prob:
            pred = pred.clone().detach()
            if self.regression:
                pred = torch.stack(
                    [F.logsigmoid(-pred), F.logsigmoid(pred)], -1)
            kl = kl_fn(pred, smax_feat).item()
            kl_list.append(kl)

        return np.array(kl_list)  # a, v, l

    def cal_cos(self, cls_grad, fusion_grad):
        fgn = fusion_grad.clone().view(-1)
        loss = list()
        for i in range(len(self.modals)):
            tmp = cls_grad[i].clone().view(-1)
            l = F.cosine_similarity(tmp, fgn, dim=0)
            loss.append(l)
        return loss  # a, v, l

    def forward(self, U, U_a, U_v):
        U_detach = U.clone().detach()
        U_a_detach = U_a.clone().detach()
        U_v_detach = U_v.clone().detach()

        pred_l = self.classifier_l(U_detach)
        pred_a = self.classifier_a(U_a_detach)
        pred_v = self.classifier_v(U_v_detach)
        self.uni_prob = [pred_a, pred_v, pred_l]

        return self.uni_prob

    def get_loss(self, enc, smax_feat, label, criterion, model, epoch):
        # Get gradient of predict model (MMGCN, MM-DFN, etc.)
        fusion_grad = None
        for name, para in model.named_parameters():
            if 'smax_fc.weight' in name:
                fusion_grad = para
                break

        # Classfier Guided
        U, U_a, U_v = enc
        uni_prob = self.forward(U, U_a, U_v)
        cls_loss = torch.stack([criterion(prob, label)
                               for prob in uni_prob]).sum()
        cls_loss.backward()

        cls_grad = []
        for name, para in self.named_parameters():
            if 'out_layer.weight' in name:
                cls_grad.append(para)

        assert all(fusion_grad.shape == g.shape for g in cls_grad)
        llist = self.cal_cos(cls_grad, fusion_grad)

        cur_kl = self.cal_kl(smax_feat, self.uni_prob)
        diff_kl = self.prev_kl - cur_kl if self.prev_kl is not None else cur_kl
        diff_sum_kl = diff_kl.sum() + 1e-8
        coeff_kl = (diff_sum_kl - diff_kl) / diff_sum_kl
        self.prev_kl = cur_kl

        loss_gm = torch.stack([torch.abs(coeff_kl[i]*llist[i]).sum()
                              for i in range(len(self.modals))]).sum()
        loss_gm = loss_gm / len(self.modals)

        if self.start_epoch > epoch or epoch > self.end_epoch:
            # Do not apply classifier guided loss if out of range
            return loss_gm, coeff_kl, cur_kl, llist

        self.loss_gm = loss_gm
        # Modify unimodal encoder gradient
        for i in range(len(self.modals)):
            modal = self.modals[i]
            for name, params in model.named_parameters():
                if f'_{modal}.weight' in name or f'_{modal}.bias' in name:
                    params.grad *= (coeff_kl[i] * self.rho)
                    if self.normalize:
                        params.grad += torch.zeros_like(params.grad).normal_(
                            0, params.grad.std().item() + 1e-8)

        return loss_gm, coeff_kl, cur_kl, llist
