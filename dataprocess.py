
import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics
from munkres import Munkres
import torch.utils.data as data_utils
from typing import Union
from kmeans_gpu import kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def cluster_acc(y_true, y_pred):

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def eva(y_true, y_pred, show_details=True, epoch=0):

    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    if show_details:
        print('epoch {}'.format(epoch),':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, f1
def clustering(feature, true_labels, cluster_num, epoch):
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, f1 = eva(true_labels, predict_labels.numpy(), show_details=True, epoch = epoch)
    return acc, nmi ,f1

def l2_reg_loss(model, scale=1e-5):
    loss = 0.0
    for w in model.get_weights():
        loss += (w.pow(2.).sum()).cpu()
    return loss * scale

def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     ) -> Union[torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        return sparse_tensor
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")

def collate_fn(batch):
    edges1,edges2 ,  nonedges1 , nonedges2 = batch[0]
    return (edges1,edges2, nonedges1 , nonedges2)

def get_edge_sampler(A , A_fadj , num_pos=1000, num_neg=1000, num_workers=5):
    data_source = EdgeSampler(A ,A_fadj ,  num_pos, num_neg)
    data_source.__getitem__(4)
    return data_utils.DataLoader(data_source, num_workers=num_workers, collate_fn=collate_fn)
class EdgeSampler(data_utils.Dataset):

    def __init__(self, A, A_fadj , num_pos=1000, num_neg=1000):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A
        self.A_fadj = A_fadj

        self.edges_adj = np.transpose(A.nonzero())
        self.edges_fadj = np.transpose(A_fadj.nonzero())

        self.num_nodes = A.shape[0]
        self.num_edges_adj = self.edges_adj.shape[0]
        self.num_edges_fadj = self.edges_fadj.shape[0]


    def __getitem__(self, key):
        np.random.seed(key)
        edges_idx1 = np.random.randint(0, self.num_edges_adj, size=self.num_pos, dtype=np.int64)
        next_edges_adj = self.edges_adj[edges_idx1, :]

        edges_idx2 = np.random.randint(0, self.num_edges_fadj, size=self.num_pos, dtype=np.int64)
        next_edges_fadj = self.edges_fadj[edges_idx2, :]
        generated1 = False
        generated2 = False


        while not generated1:
            candidate_ne_adj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)

            cne1_a, cne2_a = candidate_ne_adj[:, 0], candidate_ne_adj[:, 1]

            to_keep_adj = (1 - self.A[cne1_a, cne2_a]).astype(bool).A1 * (cne1_a != cne2_a)

            next_nonedges_adj = candidate_ne_adj[to_keep_adj][:self.num_neg]

            generated1= to_keep_adj.sum() >= self.num_neg

        while not generated2:
            candidate_ne_fadj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)

            cne1_a, cne2_a = candidate_ne_fadj[:, 0], candidate_ne_fadj[:, 1]

            to_keep_fadj = (1 - self.A_fadj[cne1_a, cne2_a]).astype(bool).A1 * (cne1_a != cne2_a)

            next_nonedges_fadj = candidate_ne_fadj[to_keep_fadj][:self.num_neg]
            generated2= to_keep_fadj.sum() >= self.num_neg

        return torch.LongTensor(next_edges_adj),torch.LongTensor(next_edges_fadj) ,  torch.LongTensor(next_nonedges_adj) ,  torch.LongTensor(next_nonedges_fadj)

    def __len__(self):
        return 2**32

def load_cora(data1 , data2, N):
    adj = np.genfromtxt(data1, dtype=np.int32)
    adj = sp.coo_matrix(adj)
    adjL = adj
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_lable = A + sp.eye(A.shape[0])
    sadj = normalize_adj(adj_lable)
    feature_edges = np.genfromtxt(data2, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    fadjL = fadj
    A_fea = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj_label = A_fea +sp.eye(A.shape[0])
    fadj = normalize_adj(fadj_label)
    return sadj, fadj, A, A_fea, adjL, fadjL




def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)