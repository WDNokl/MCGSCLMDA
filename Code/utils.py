import numpy as np
import torch
from sklearn.decomposition import PCA


def pcc(u, v, eps=1e-8):
    u, v = u - torch.mean(u, dim=-1, keepdims=True), v - torch.mean(v, dim=-1, keepdims=True)
    u, v = torch.unsqueeze(u, 1), torch.unsqueeze(v, 0)
    return torch.sum(u * v, dim=-1) / (
            torch.sqrt(torch.sum(u ** 2, dim=-1)) * torch.sqrt(torch.sum(v ** 2, dim=-1)) + eps)


def extractConstraints(representation):
    # Laplace
    representation = representation + np.eye(representation.shape[0])
    D = representation.sum(axis=1)
    D_ = np.diag(np.power(D, -0.5))
    representation = np.dot(np.dot(D_, representation), D_)

    # PCA
    pca = PCA(n_components=400)
    representation = pca.fit_transform(representation)

    representation = torch.from_numpy(representation).float().detach()
    # representation = torch.from_numpy(representation).float().cuda().detach()

    pcc_mat = np.zeros((representation.shape[0], representation.shape[0]), dtype='float')
    for i in range(0, representation.shape[0], 10):
        pcc_mat[i:i + 10] = pcc(representation[i:i + 10], representation).cpu().numpy()

    pcc_mat = np.abs(pcc_mat)

    return pcc_mat


def obtain_constraints(net_numbs, emb, symbols, top_rate, idx_layer):
    pcc_mats = []
    for idx_net in range(net_numbs):
        pcc_mat = extractConstraints(emb[idx_net])
        pcc_mats.append(pcc_mat)
    cannot_links = []
    must_links = []
    for i, pcc_mat in enumerate(pcc_mats):
        np.fill_diagonal(pcc_mat, 0)
        pcc_order = np.sort(pcc_mat.flatten())
        # threshold_max = pcc_order[-topN[i]]  # original
        threshold_index = int(len(pcc_order) * top_rate)
        threshold_max = pcc_order[-threshold_index]
        # print(f'net {i} top{threshold_index} threshold {threshold_max}')
        must_link = (pcc_mat >= threshold_max)
        must_links.append(must_link)
        # cannot links
        threshold_min = pcc_order[threshold_index]
        cannot_link = (pcc_mat <= threshold_min)
        cannot_links.append(cannot_link)
    for i in range(net_numbs):
        for j in range(i):
            xy, x_indx, y_inds = np.intersect1d(symbols[i], symbols[j], return_indices=True)
            tmp1 = must_links[i][x_indx][:, x_indx] + must_links[j][y_inds][:, y_inds]
            must_links[i][np.ix_(x_indx, x_indx)] = tmp1
            must_links[j][np.ix_(y_inds, y_inds)] = tmp1

            tmp2 = cannot_links[i][x_indx][:, x_indx] + cannot_links[j][y_inds][:, y_inds]
            cannot_links[i][np.ix_(x_indx, x_indx)] = tmp2
            cannot_links[j][np.ix_(y_inds, y_inds)] = tmp2
        # print('### Network {}: Number of Must link: {} Number of Cannot link: {}'.format(i, must_links[i].sum(), cannot_links[i].sum()))
    return must_links, cannot_links
