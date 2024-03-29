from __future__ import print_function, division
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from dataprocess import to_sparse_tensor
from datetime import datetime
import time
from sklearn.cluster import KMeans


tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)
class GNNLayer(Module):
    def __init__(self, in_features, out_features,dropout):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        if self.dropout != 0:
            features = sparse_or_dense_dropout(features, p=self.dropout, training=self.training)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.leaky_relu(output, negative_slope=0.2)
        return output


class BernoulliDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss

    def forward_batch(self, emb, idx):
        raise NotImplementedError

    def forward_full(self, emb):
        raise NotImplementedError

    def loss_batch(self, emb, ones_idx, zeros_idx):
        raise NotImplementedError

    def loss_full(self, emb, adj):
        raise NotImplementedError


class BerpoDecoder(BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__(num_nodes, num_edges, balance_loss)
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):

        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):

        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)


from torch.nn import Linear
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 3)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.wl(mlp_in)), dim=1)

        return weight_output

class MLP(nn.Module):
    def __init__(self , in_fea , out_fea):
        super(MLP, self).__init__()
        self.mlp= nn.Linear(in_fea , out_fea)

    def forward(self , x):
        out = F.softmax(F.leaky_relu(self.mlp(x)) ,dim=1)
        return out

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 3)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 3)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 3)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w3(mlp_in)), dim=1)

        return weight_output
class fea(nn.Module):
    def __init__(self , x_input , x_enc1 , x_enc2 , x_enc3 , z_input ,dropout):
        super(fea, self).__init__()
        self.agcn1_fea = GNNLayer(x_input, x_enc1, dropout)
        self.agcn2_fea = GNNLayer(x_enc1, x_enc2, dropout)
        self.agcn3_fea = GNNLayer(x_enc2, x_enc3, dropout)
        self.agcn4_fea = GNNLayer(x_enc3, z_input, dropout)
        hidden_dims = [x_enc1 , x_enc2 , x_enc3 , z_input]
        if True:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x,fadj):
        fea_h1 = self.agcn1_fea(x,fadj)
        fea_h1 = F.relu(fea_h1)
        fea_h1 = self.batch_norm[0](fea_h1)

        fea_h2 = self.agcn2_fea(fea_h1,fadj)
        fea_h2 = F.relu(fea_h2)
        fea_h2 = self.batch_norm[1](fea_h2)

        fea_h3 = self.agcn3_fea(fea_h2,fadj)
        fea_h3 = F.relu(fea_h3)
        fea_h3 = self.batch_norm[2](fea_h3)
        fea_h4 = self.agcn4_fea(fea_h3,fadj)
        fea_h4 = F.relu(fea_h4)
        fea_h4 = self.batch_norm[3](fea_h4)
        return fea_h1,fea_h2,fea_h3,fea_h4

class NOCD_DL(nn.Module):
    @staticmethod
    def normalize_adj(adj: sp.csr_matrix):
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
            return adj_norm
        return to_sparse_tensor(adj_norm)

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    def get_biases(self):
        return [w for n, w in self.named_parameters() if 'bias' in n]
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v,dropout,batch_norm , name):
        super(NOCD_DL, self).__init__()
        self.n_cluster = n_clusters
        self.dropout = dropout
        self.n_z = n_z
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load('./data/pre_model/{}_{}.pkl'.format(name , n_z), map_location='cpu'))
        self.fea_gcn = fea(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, dropout)

        self.agcn_0 = GNNLayer(n_input, n_enc_1,dropout)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2,dropout)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3,dropout)
        self.agcn_3 = GNNLayer(n_enc_3, n_z,dropout)
        self.agcn_z = GNNLayer(n_z, n_clusters,dropout)

        self.mlp = MLP_L(3*n_z)

        self.mlp1 = MLP_1(3 * n_enc_1)
        self.mlp2 = MLP_2(3 * n_enc_2)
        self.mlp3 = MLP_3(3 * n_enc_3)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        hidden_dims = [ n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters]
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None
        self.v = v

    def forward(self, x, adj , fadj):
        x_bar, h1, h2, h3, z = self.ae(x)
        fea_h1, fea_h2, fea_h3, fea_h4 = self.fea_gcn(x, fadj)


        x_array = list(np.shape(x))
        n_x = x_array[0]

        z1 = self.agcn_0(x, adj)
        z1 = F.relu(z1)
        if self.batch_norm is not None:
            z1 = self.batch_norm[0](z1)
        # z2
        m1 = self.mlp1(torch.cat((h1, z1,fea_h1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])
        m12 = torch.reshape(m1[:, 1], [n_x, 1])
        m13 = torch.reshape(m1[:, 2], [n_x, 1])
        m11_broadcast = m11.repeat(1, 500)
        m12_broadcast = m12.repeat(1, 500)
        m13_broadcast = m13.repeat(1, 500)
        z2 = self.agcn_1(m11_broadcast.mul(h1)+m12_broadcast.mul(z1)+m13_broadcast.mul(fea_h1), adj)
        z2 = F.relu(z2)
        if self.batch_norm is not None:
            z2 = self.batch_norm[1](z2)
        m2 = self.mlp2(torch.cat((h2, z2,fea_h2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:, 0], [n_x, 1])
        m22 = torch.reshape(m2[:, 1], [n_x, 1])
        m23 = torch.reshape(m2[:, 2], [n_x, 1])
        m21_broadcast = m21.repeat(1, 500)
        m22_broadcast = m22.repeat(1, 500)
        m23_broadcast = m23.repeat(1, 500)
        z3 = self.agcn_2(m21_broadcast.mul(h2)+m22_broadcast.mul(z2)+m23_broadcast.mul(fea_h2), adj)
        z3 = F.relu(z3)
        if self.batch_norm is not None:
            z3 = self.batch_norm[2](z3)
        # z4
        m3 = self.mlp3(torch.cat((h3, z3,fea_h3), 1))  # self.mlp3(h2)
        m3 = F.normalize(m3, p=2)
        m31 = torch.reshape(m3[:, 0], [n_x, 1])
        m32 = torch.reshape(m3[:, 1], [n_x, 1])
        m33 = torch.reshape(m3[:, 2], [n_x, 1])
        m31_broadcast = m31.repeat(1, 2000)
        m32_broadcast = m32.repeat(1, 2000)
        m33_broadcast = m33.repeat(1, 2000)
        z4 = self.agcn_3(m31_broadcast.mul(h3)+m32_broadcast.mul(z3)+m33_broadcast.mul(fea_h3), adj)
        z4 = F.relu(z4)
        if self.batch_norm is not None:
            z4 = self.batch_norm[3](z4)

        m4 = self.mlp(torch.cat((z, z4, fea_h4), 1))  # self.mlp3(h2)
        m4 = F.normalize(m4, p=2)
        m41 = torch.reshape(m4[:, 0], [n_x, 1])
        m42 = torch.reshape(m4[:, 1], [n_x, 1])
        m43 = torch.reshape(m4[:, 2], [n_x, 1])
        m41_broadcast = m41.repeat(1, self.n_z)
        m42_broadcast = m42.repeat(1, self.n_z)
        m43_broadcast = m43.repeat(1, self.n_z)
        z5 = self.agcn_z(m41_broadcast.mul(z) + m42_broadcast.mul(z4) + m43_broadcast.mul(fea_h4), adj)
        emb = F.relu(z5)

        if self.batch_norm is not None:
            emb = self.batch_norm[4](emb)
        pred = F.softmax(emb, dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        fadj_recon = self.dot_product_decode(fea_h4)


        return x_bar ,  z , q , emb , pred , fea_h4, fadj_recon


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()