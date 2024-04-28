from __future__ import division
import warnings
warnings.filterwarnings('ignore')
from evaluation import  eva
from torch import optim
from net import *
from dataprocess import  get_edge_sampler ,l2_reg_loss, load_cora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def log(x):
    return torch.log(x + 1e-8)

dataName = "citeseer"
data1 = './data/{}/{}_adj.txt'.format(dataName , dataName)
data2 = './data/{}/knn_{}/c9.txt'.format(dataName , dataName)
label_file = './data/{}/{}_label.txt'.format(dataName , dataName)
features_file = './data/{}/{}_fea.txt'.format(dataName , dataName)
labels = np.genfromtxt(label_file, dtype=np.int32)
N, K = labels.shape[0] , labels.max() # cora、citeseer的label从1开始
sadj, fadj , A , A_fea , adjL, fadjL= load_cora(data1 ,data2, N)
adjL = np.array(adjL.todense())
fadj_label = torch.tensor(fadjL.todense(),dtype=torch.float32).to(device)

features = np.genfromtxt(features_file)
X_ori = torch.tensor(features ,dtype=torch.float32).to(device)
y_true = labels -1
weight_decay = 1e-2
dropout = 0.5
batch_norm = True
lr = 0.01
max_epochs = 500
balance_loss = True
batch_size = 20000
x_norm =  torch.tensor(features ,dtype=torch.float32)
x_norm = x_norm.to(device)
feat_dim =x_norm.shape[1]

model = NOCD_DL(500 ,500 , 2000 , 2000 , 500 , 500 ,feat_dim, 10, K , 1 ,  dropout=dropout , batch_norm=batch_norm ,name = dataName)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
adj_norm = sadj.to(device)
fadj_norm  = fadj.to(device)
sampler = get_edge_sampler( A, A_fea , batch_size, batch_size, num_workers=5)
decoder = BerpoDecoder(N, A.nnz, balance_loss=balance_loss)

with torch.no_grad():
    _, _, _, _, z =  model.ae(x_norm)
kmeans = KMeans(n_clusters=K, n_init=20)
y_pred = kmeans.fit_predict(z.data.cpu().numpy())
y_pred_last = y_pred
model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

if __name__ == '__main__':
    NMIS=[]
    F1 = []
    ARI = []
    ACC =[]
    Qs = []

    for epoch, batch in enumerate(sampler):
        if epoch >= max_epochs:
            break
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                x_bar, z, q, emb, pred, fea_h4, fadj_recon = model(x_norm, adj_norm, fadj_norm)
                tmp_q = q.data
                p = target_distribution(tmp_q)
                val_loss = decoder.loss_full(F.relu(pred), A)
                res2 = pred.data.cpu().numpy().argmax(1)
                acc, nmi, ari, f1, m = eva(y_true, res2, adjL)
                NMIS.append(nmi)
                F1.append(f1)
                Qs.append(m)
                ACC.append(acc)
                ARI.append(ari)
                print(
                    f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, acc = {acc:.3f} , nmi = {nmi:.3f} , ari = {ari:.3f} , F1-score = {f1:.3f},  modularity = {m:.3f}')

        model.train()
        optimizer.zero_grad()
        x_bar ,  z , q , emb , pred, fea_h4, fadj_recon = model(x_norm, adj_norm, fadj_norm)
        one1_idx, one2_idx, zero1_idx, zero2_idx = batch
        loss_decoder = decoder.loss_batch(F.relu(pred), one1_idx, zero1_idx)
        loss_decoder2 = decoder.loss_batch(F.relu(pred), one2_idx, zero2_idx)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, x_norm)
        fadj_re_loss = F.binary_cross_entropy(fadj_recon, fadj_label)
        ce_loss = F.kl_div(log(F.relu(pred)), p, reduction='batchmean')
        loss = 1 * loss_decoder +  1000 * kl_loss + 1000*re_loss  + 0.01 * ce_loss  + loss_decoder2+ 1 * fadj_re_loss
        loss += l2_reg_loss(model, scale=1e-2)
        loss.backward()
        optimizer.step()



    max_nmi = (np.array(NMIS)).max()
    max_f1score = (np.array(F1)).max()
    max_Acc = (np.array(ACC)).max()
    max_ari = (np.array(ARI)).max()
    max_q = (np.array(Qs)).max()
    print(" max_acc : {:.3f}".format(max_Acc))
    print("---------------------------------------------------------------------------------")
    print(" max_nmi : {:.3f}".format(max_nmi))
    print("---------------------------------------------------------------------------------")
    print(" max_ariscore : {:.3f}".format(max_ari))
    print("---------------------------------------------------------------------------------")
    print("max_f1score : {:.3f}".format(max_f1score))
    print("---------------------------------------------------------------------------------")
    print("max_q : {:.3f}".format( max_q))

