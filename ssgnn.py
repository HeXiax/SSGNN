import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.optim import Adam
from sklearn.cluster import KMeans
from utils import load_data, load_graph
#from missadj import load_data, load_graph

from layers import target_distribution
from layers import GNNLayer
#from layers import LRR_Z
from evaluation import eva
import opt
from utils import sparse_mx_to_torch_sparse_tensor



def soft_thresholding(x, soft_eta):
    out = F.relu(x - soft_eta, inplace=False) - F.relu(- x - soft_eta, inplace=False)
    return out


class LRR_ADMM(nn.Module):
    def __init__(self, num_nodes, num_features, lambda1=10.0, admm_iter=2, rho=1.1, mu_max=1e+6,mu_0=10.0):
        super(LRR_ADMM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = torch.tensor(rho)
        self.lambda1 = torch.tensor(lambda1)
        # In future, we shall make this nu as a parameters
        self.mu_max = torch.tensor(1e+6)
        self.mu = torch.tensor(mu_0)

    def forward(self, X, init_Z=None, init_delta=None, init_Delta=None):
        """
        Parameters
        ----------
        X : Graph signal to be smoothed, shape [Num_node, Num_features]
        init_Z:   the diagonal should be 0
        Returns:  approximated adjacency Z
        """
        if init_Z == None:
            Z = torch.zeros((self.num_nodes, self.num_nodes)).to(device)
        else:
            Z = init_Z
        if init_Delta == None:
            Delta = torch.zeros((self.num_nodes, self.num_nodes)).to(device)
        else:
            Delta = init_Delta
        if init_delta == None:
            delta = torch.zeros((self.num_nodes, 1)).to(device)
        else:
            delta = init_delta

        XXt = self.lambda1 * torch.mm(X, X.transpose(0, 1))

        for k in range(self.admm_iter):
            mergedX = torch.hstack(
                [self.lambda1.pow(0.5) * X, self.mu.pow(0.5) * torch.ones((self.num_nodes, 1)).to(device)])
            mergedX1 = self.mu * torch.eye(mergedX.shape[1]).to(device) + torch.mm(mergedX.transpose(0, 1), mergedX).to(device)
            try:
                inv_mergedX = mergedX1.inverse().to(device)
            except:
                print("no inverse")

            Q = torch.mm(XXt + self.mu * (torch.ones((self.num_nodes, self.num_nodes)).to(device) + Z) - Delta - delta,\
                         (torch.eye(self.num_nodes).to(device) - torch.mm(mergedX,torch.mm(inv_mergedX,mergedX.transpose(0,1)))) / self.mu)

            Z = soft_thresholding(Q + Delta / self.mu, 1 / self.mu)
            Z.fill_diagonal_(0.0)
            delta = delta + self.mu * (Q.sum(1).unsqueeze(1) - 1.0)
            Delta = Delta + self.mu * (Q - Z)
            self.mu = torch.min(self.rho * self.mu, self.mu_max)

        Q = 0.5 * (Q + Q.transpose(0, 1))

        aa = torch.abs(Z).max()
        Z = soft_thresholding(Z, 0.5 * aa)
        Z = torch.where(Z != 0, 1.0, 0.0)
        Z = 0.5 * (Z + Z.transpose(0, 1))
        Z = Z + torch.eye(self.num_nodes).to(device)
        #rowsum = torch.sum(Z, dim=1) ** (-0.5)
        rowsum = torch.sum(Z, dim=1) ** (-1)
        D = torch.diag(rowsum)
        Z = torch.mm(D, Z)
        #Z = torch.mm(Z, D)
        return Z, Q



class LRR_GCN(nn.Module):
    def __init__(self,hidden1,hidden2,hidden3,n_z,n_nodes,n_input,n_clusters,dropout):
        super(LRR_GCN,self).__init__()
        #GCN
        self.lrr1 = LRR_ADMM(n_nodes, n_input)
        self.gnn1 = GNNLayer(n_input, hidden1)
        self.lrr2 = LRR_ADMM(n_nodes, hidden1)
        self.gnn2 = GNNLayer(hidden1, hidden3)
        self.lrr3 = LRR_ADMM(n_nodes, hidden3)
        self.gnn3 = GNNLayer(hidden3, n_z)
        self.lrr4 = LRR_ADMM(n_nodes, n_z)

        self.gnn5 = GNNLayer(n_z, hidden3)
        self.gnn6 = GNNLayer(hidden3, hidden1)
        self.gnn7 = GNNLayer(hidden1, n_input)

    def forward(self, x, adj):
        sigma = 0.5
        adj1,_ = self.lrr1(x)
        h = self.gnn1(x,(sigma * adj1+(1-sigma) * adj))
        adj1,_ = self.lrr2(h)
        h = self.gnn2(h, (sigma * adj1+(1-sigma) * adj))
        adj1,_ = self.lrr3(h)
        z = self.gnn3(h, (sigma * adj1+(1-sigma) * adj), active=False)
        adj1,_ = self.lrr4(z)

        h = self.gnn5(z, (sigma * adj1+(1-sigma) * adj))
        adj1, _ = self.lrr3(h)
        h = self.gnn6(h, (sigma * adj1+(1-sigma) * adj))
        adj1, _ = self.lrr2(h)
        x_hat = self.gnn7(h, (sigma * adj1+(1-sigma) * adj), active=False)


        adj_hat = dot_product_decode(x_hat)
        return z, adj_hat, x_hat


class self_LRRGCN(nn.Module):
    def __init__(self,hidden1,hidden2,hidden3,n_z,n_nodes,n_input,n_clusters,dropout,v=1.0):
        super(self_LRRGCN,self).__init__()
        self.v = v
        self.pregcn = LRR_GCN(hidden1,hidden2,hidden3,n_z,n_nodes,n_input,n_clusters,dropout)
        self.pregcn.load_state_dict(torch.load(opt.args.pretrain_path, map_location='cpu'))

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self,x,adj):
        z, adj_hat, x_hat = self.pregcn(x,adj)
        return z, adj_hat, x_hat


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred


acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []

def train_ssgnn(dataset):
    model = self_LRRGCN(hidden1=opt.args.hidden1,hidden2 = opt.args.hidden2,hidden3 = opt.args.hidden3,
                        n_z=opt.args.n_z,n_nodes=opt.args.n_nodes,n_input=opt.args.n_input,
                        n_clusters=opt.args.n_clusters,
                        dropout=opt.args.dropout,v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(),lr=opt.args.lr)

    adj = load_graph(opt.args.name,opt.args.k)
    adj = adj.to(device)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        h, _, _ = model(data,adj)

    kmeans = KMeans(n_clusters=opt.args.n_clusters,n_init=20)
    y_pred = kmeans.fit_predict(h.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y,y_pred,'initialization')

    for epoch in range(300):
        model.train()
        z, adj_hat, x_hat = model(data, adj)

        gae_loss = F.mse_loss(x_hat, data)
        gr_loss = F.mse_loss(adj_hat, adj.to_dense())
        loss = gae_loss + 0.01 * gr_loss
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z.data.cpu().numpy())

        acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("ARI: {:.4f}".format(ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("F1: {:.4f}".format(f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]))
    print("Epoch:", np.where(acc_reuslt == np.max(acc_reuslt))[0][0])


if __name__== "__main__":

    print("use cuda: {}".format(opt.args.cuda))
    device = torch.device("cuda:0" if opt.args.cuda else "cpu")
    print("use {}".format(device))
    #device = torch.device(0)

    dataset = load_data(opt.args.name)

    if opt.args.name == 'acm':
        opt.args.lr = 1e-4
        opt.args.k = None
        opt.args.n_clusters = 3
        opt.args.n_input = 1870
        opt.args.n_nodes = 3025
        opt.args.pretrain_path = 'acm3.pkl'

    if opt.args.name == 'dblp':
        opt.args.lr = 1e-4
        opt.args.k = None
        opt.args.n_clusters = 4
        opt.args.n_input = 334
        opt.args.n_nodes = 4057
        opt.args.pretrain_path = 'dblp3.pkl'

    if opt.args.name == 'cite':
        opt.args.lr = 1e-4
        opt.args.k = None
        opt.args.n_clusters = 6
        opt.args.n_input = 3703
        opt.args.n_nodes = 3327
        opt.args.pretrain_path = 'cite3.pkl'

    if opt.args.name == 'IMDB':
        opt.args.lr = 1e-3
        opt.args.k = None
        opt.args.n_clusters = 3
        opt.args.n_input = 1232
        opt.args.n_nodes = 4780
        opt.args.pretrain_path = 'IMDB.pkl'

    print(opt.args)
    train_ssgnn(dataset)





