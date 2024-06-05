import csv
import random

import dgl
import torch
import torch as t

EOS = 1e-10

# -*- Coding: utf-8 -*-
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import minmax_scale, scale


def original_data(args):
    original_features = dict()
    data_path = args.datapath
    mi_networks, mi_symbols = [], []
    mi_paths = args.miRNA
    for path in mi_paths:
        sim = pd.read_csv(data_path + f"{path}.csv", header=None).values
        mi_networks.append(sim)
        pf = pd.read_csv(data_path + f"miRNA_name.csv")
        sy = pf.values[:, 0]
        mi_symbols.append(sy)
    miRNA = dict()
    miRNA['features'] = mi_networks
    miRNA['symbols'] = mi_symbols
    dis_networks, dis_symbols = [], []
    dis_paths = args.disease
    for path in dis_paths:
        sim = pd.read_csv(data_path + f"{path}.csv", header=None).values
        dis_networks.append(sim)
        pf = pd.read_csv(data_path + f"disease_name.csv")
        sy = pf.values[:, 0]
        dis_symbols.append(sy)
    disease = dict()
    disease['features'] = dis_networks
    disease['symbols'] = dis_symbols
    original_features['miRNA'] = miRNA
    original_features['disease'] = disease
    return original_features


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float) / col[:, None]

    return A


def RWR(A, K=3, alpha=0.98):
    """Random Walk on graph"""
    A = _scaleSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha * np.dot(P, A) + (1. - alpha) * P0
        M = M + P

    return M


def load_networks(path):
    networks, symbols = [], []
    for file in path:
        network = np.load(file)['corr']
        network = RWR(network)
        # network = scale(network, axis=1)
        network = minmax_scale(network)

        networks.append(network)
        symbols.append(np.load(file, allow_pickle=True)['symbol'])

    return networks, symbols


class netsDataset(Dataset):
    def __init__(self, net):
        super(netsDataset, self).__init__()
        self.net = net

    def __len__(self):
        return len(self.net)

    def __getitem__(self, item):
        x = self.net[item]
        y = self.net[item]
        return x, y, item


class datapro:
    def __init__(self):
        super(datapro, self).__init__()

    def read_csv(self, filename):
        with open(filename, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            md_data = []
            md_data += [[float(i) for i in row] for row in reader]
            return t.Tensor(md_data)

    def get_edge_index(self, matrix):
        edge_index = [[], []]
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                if matrix[i][j] != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        return torch.LongTensor(edge_index)

    def get_data(self, data):
        miRNA = data['miRNA']
        disease = data['disease']
        attributes_list = []
        for i in range(len(miRNA)):
            c_row, c_colum = miRNA[i].shape
            d_row, d_colum = disease[i].shape
            attributes_list.append(np.vstack((np.hstack((miRNA[i], np.zeros(shape=(c_row, d_colum), dtype=int))),
                                              np.hstack((np.zeros(shape=(d_row, c_colum), dtype=int), disease[i])))))
        features = np.hstack(attributes_list)
        features = features.astype(float)
        features = t.FloatTensor(features)
        original_adj = data['MDA']['original_adj']
        m_d_matrix = data['MDA']['m_d_matrix']
        return features, original_adj, m_d_matrix

    def load_data(self, args, data):
        if args.fusion:
            features = data['features']['consine features']
        else:
            features = data['features']['features']
        original_adj = data['MDA']['original_adj']
        m_d_matrix = data['MDA']['m_d_matrix']
        return features, original_adj, m_d_matrix

    def random_index(self, index_matrix, args):
        association_num = index_matrix.shape[1]
        random_index = index_matrix.T.tolist()
        random.shuffle(random_index)
        k_folds = args.k_fold
        CV_size = int(association_num / k_folds)
        temp = np.array(random_index[:association_num - association_num % k_folds]).reshape(k_folds, CV_size,
                                                                                            -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + random_index[
                                                association_num - association_num % k_folds:]
        return temp

    def datasplit(self, args, m_d_matrix):
        pos_index_matrix = np.mat(np.where(m_d_matrix == 1))
        neg_index_matrix = np.mat(np.where(m_d_matrix == 0))

        pos_index = self.random_index(neg_index_matrix, args)
        neg_index = self.random_index(pos_index_matrix, args)

        index = [pos_index[i] + neg_index[i] for i in range(args.k_fold)]
        return index

    def normalize(self, adj):
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

    def torch_sparse_to_dgl_graph(self, torch_sparse_mx):
        torch_sparse_mx = torch_sparse_mx.coalesce()  # sparse matrix
        indices = torch_sparse_mx.indices()
        values = torch_sparse_mx.values()
        rows_, cols_ = indices[0, :], indices[1, :]
        dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
        dgl_graph.edata['w'] = values.detach().cuda()
        return dgl_graph

    def get_feat_mask(self, features, mask_rate):
        feat_node = features.shape[1]
        mask = torch.zeros(features.shape)
        samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
        mask[:, samples] = 1
        return mask.cuda(), samples

    def symmetrize(self, adj):  # only for non-sparse
        return (adj + adj.T) / 2

    def getview(self, d_views, c_views):
        d_view1 = t.FloatTensor(d_views[0])
        d_view2 = t.FloatTensor(d_views[1])
        d_view3 = t.FloatTensor(d_views[2])
        c_view1 = t.FloatTensor(c_views[0])
        c_view2 = t.FloatTensor(c_views[1])
        c_view3 = t.FloatTensor(c_views[2])
        return d_view1, d_view2, d_view3, c_view1, c_view2, c_view3
