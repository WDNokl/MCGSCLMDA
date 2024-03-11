import torch as t
from torch.nn import Sequential, Linear, ReLU
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph


class autoEncoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super(autoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden if isinstance(hidden, list) else [hidden]
        self.input_hidden = [input_dim] + self.hidden

        self.linears_encoder = nn.ModuleList(
            [nn.Linear(self.input_hidden[i], self.input_hidden[i + 1]) for i in range(len(self.input_hidden) - 1)]
        )
        self.linears_decoder = nn.ModuleList(
            [nn.Linear(self.input_hidden[i], self.input_hidden[i - 1]) for i in
             range(len(self.input_hidden) - 1, 0, -1)][::-1]
        )

        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm1d(i, momentum=0.1, affine=True) for i in self.input_hidden]
        )

    def encoder(self, x, idx_layer):
        x = self.linears_encoder[idx_layer](x)
        x = self.batchNorms[idx_layer + 1](x)
        x = torch.sigmoid(x)

        return x

    def decoder(self, y, idx_layer):
        y = self.linears_decoder[idx_layer](y)
        y = self.batchNorms[idx_layer](y)
        y = torch.sigmoid(y)

        return y

    def decoder_end(self, y, idx_layer):
        for i in range(idx_layer, -1, -1):
            y = self.decoder(y, i)

        return y

    def forward(self, x, idx_layer, flag='training'):
        encoder_op = self.encoder(x, idx_layer)

        if flag == 'training':
            decoder_op = self.decoder(encoder_op, idx_layer)
        else:
            decoder_op = self.decoder_end(encoder_op, idx_layer)

        return decoder_op, encoder_op


class encoderLoss(nn.Module):
    def __init__(self, mustlinks, cannotliks, batch_size, gamma):
        super(encoderLoss, self).__init__()
        self._mustlinks = mustlinks
        self._cannotlinks = cannotliks
        self._batch_size = batch_size
        self._gamma = gamma
        self._alpha = 1 - gamma
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, y_pred, y_truth, data_indx):
        M = self._mustlinks[data_indx][:, data_indx].float()
        loss_ml = self.get_constraints_loss(y_pred, M)

        M2 = self._cannotlinks[data_indx][:, data_indx].float()
        loss_cl = self.get_constraints_loss(y_pred, M2)
        # loss = F.binary_cross_entropy(y_pred, y_truth)
        loss = F.mse_loss(y_pred, y_truth)

        return loss + self._gamma * loss_ml - self._alpha * loss_cl, loss_ml, loss_cl

    def get_constraints_loss(self, y_pred, constraints_batch):
        D = torch.diag(constraints_batch.sum(axis=1))
        L = D - constraints_batch
        loss = torch.trace(torch.matmul(torch.matmul(y_pred.T, L), y_pred)) * 2

        return loss / (self._batch_size * self._batch_size)


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


class FGP_learner(nn.Module):
    def __init__(self, features, k, knn_metric, i):
        super(FGP_learner, self).__init__()

        self.k = k
        self.knn_metric = knn_metric
        self.i = i

        self.Adj = nn.Parameter(
            torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))

    def forward(self, h):
        Adj = F.elu(self.Adj) + 1
        return Adj


class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj):

        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

        self.gnn_encoder_layers = nn.ModuleList()
        self.gnn_encoder_layers.append(GCNConv_dense(in_dim, hidden_dim))
        for _ in range(nlayers - 2):
            self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.gnn_encoder_layers.append(GCNConv_dense(hidden_dim, emb_dim))

        self.dropout_adj = nn.Dropout(p=dropout_adj)

        self.projector = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                    Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_):

        Adj = self.dropout_adj(Adj_)

        for conv in self.gnn_encoder_layers[:-1]:
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gnn_encoder_layers[-1](x, Adj)
        z = self.projector(x)
        return z, x


class GCL(nn.Module):
    def __init__(self, in_dim, args):
        super(GCL, self).__init__()
        self.m = args.m
        self.nlayers = args.nlayers
        self.in_dim = in_dim
        self.hidden_dim = args.hidden_dim
        self.emb_dim = args.emb_dim
        self.proj_dim = args.proj_dim
        self.dropout = args.dropout
        self.dropout_adj = args.dropedge_rate
        self.encoder_a = GraphEncoder(self.nlayers, in_dim, self.hidden_dim, self.emb_dim, self.proj_dim, self.dropout,
                                      self.dropout_adj)
        self.encoder_l = GraphEncoder(self.nlayers, in_dim, self.hidden_dim, self.emb_dim, self.proj_dim, self.dropout,
                                      self.dropout_adj)
        # self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        self._build_projector_and_predictor_mlps(self.emb_dim, self.hidden_dim)

        for param_a, param_l in zip(self.encoder_a.parameters(), self.encoder_l.parameters()):
            param_a.data.copy_(param_l.data)
            param_l.requires_grad = False

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    # project
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_a, param_l in zip(self.encoder_a.parameters(), self.encoder_l.parameters()):
            param_a.data = param_a.data * self.m + param_l.data * (1.0 - self.m)

    def forward(self, x, Adj_, args, branch):
        if branch == 'learner':
            z, embedding = self.encoder_l(x, Adj_)
        else:
            if args.moco:
                with torch.no_grad():
                    self._momentum_update_key_encoder()
                    z, embedding = self.encoder_a(x, Adj_)
            else:
                z, embedding = self.encoder_a(x, Adj_)
        return z, embedding


class MoCo_MDA(GCL):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.encoder_a.projector[0].weight.shape[1]
        del self.encoder_a.projector, self.encoder_l.projector  # remove original fc layer

        # projectors
        # (3,,256,4096)
        self.encoder_a.projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.encoder_l.projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, last_bn=False)


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, args):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        self.hidden_channels = args.hidden_dim_cls
        self.num_layers = args.nlayers_cls
        self.dropout = args.dropout_cls
        self.dropout_adj = args.dropedge_cls

        self.layers.append(GCNConv_dense(in_channels, self.hidden_channels))
        for i in range(self.num_layers - 2):
            self.layers.append(GCNConv_dense(self.hidden_channels, self.hidden_channels))
        self.layers.append(GCNConv_dense(self.hidden_channels, out_channels))

        self.dropout = self.dropout
        self.dropout_adj_p = self.dropout_adj

        self.dropout_adj = nn.Dropout(p=self.dropout_adj)

    def forward(self, x, Adj):

        Adj = self.dropout_adj(Adj)

        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction, drug_lap, mic_lap, alpha1, alpha2, sizes):
        loss_ls = t.norm((target - prediction), p='fro') ** 2
        drug_reg = t.trace(t.mm(t.mm(alpha1.T, drug_lap), alpha1))
        mic_reg = t.trace(t.mm(t.mm(alpha2.T, mic_lap), alpha2))
        graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * mic_reg
        loss_sum = loss_ls + graph_reg
        return loss_sum.sum()


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def calc_loss(x, x_aug, temperature=0.2, sym=True):  # 公式14 节点级对比损失函数
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1
