import numpy as np
import pickle
from universal_divergence import estimate
from sklearn.metrics import silhouette_samples
import glob
import os
from math import log, e
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# sys.path.insert(0, 'KL-divergence-estimators/src/')
# from knn_divergence import naive_estimator

cal_min = 30  # minimum number of cells for estimation

def entropy(labels, base=None):
    """ Computes entropy of label distribution.
    Args:
        labels: list of integers
    Returns:
        ent: entropy
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent


def cal_entropy(code, idx, dataset_labels, k=100):
    """ Calculate entropy of cell types of nearest neighbors
    Args:
        code: num_cells * num_features, embedding for calculating entropy
        idx: binary, index of observations to calculate entropy
        dataset_labels:
        k: number of nearest neighbors
    Returns:
        entropy_list: list of entropy of each cell
    """
    cell_sample = np.where(idx == True)[0]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(code)
    entropy_list = []
    _, indices = nbrs.kneighbors(code[cell_sample, :])
    for i in range(len(cell_sample)):
        entropy_list.append(entropy(dataset_labels[indices[i, :]]))

    return entropy_list


def evaluate_scores(div_ent_code, sil_code, cell_labels, dataset_labels, num_datasets, div_ent_dim, sil_dim, sil_dist):
    """ Calculate three proposed evaluation metrics
    Args:
        div_ent_code: num_cells * num_features, embedding for divergence and entropy calculation, usually with dim of 2
        sil_code: num_cells * num_features, embedding for silhouette score calculation
        cell_labels:
        dataset_labels:
        num_datasets:
        div_ent_dim: if dimension of div_ent_code > div_ent_dim, apply PCA first
        sil_dim: if dimension of sil_code > sil_dim, apply PCA first
        sil_dist: distance metric for silhouette score calculation
    Returns:
        div_score: divergence score
        ent_score: entropy score
        sil_score: silhouette score
    """
    # calculate divergence and entropy
    if div_ent_code.shape[1] > div_ent_dim:
        div_ent_code = PCA(n_components=div_ent_dim).fit_transform(div_ent_code)
    div_pq = []  # divergence dataset p, q
    div_qp = []  # divergence dataset q, p
    ent = []  # entropy
    # pairs of datasets
    for d1 in range(1, num_datasets+1):
        for d2 in range(d1+1, num_datasets+1):
            idx1 = dataset_labels == d1
            idx2 = dataset_labels == d2
            labels = np.intersect1d(np.unique(cell_labels[idx1]), np.unique(cell_labels[idx2]))
            idx1_mutual = np.logical_and(idx1, np.isin(cell_labels, labels))
            idx2_mutual = np.logical_and(idx2, np.isin(cell_labels, labels))
            idx_specific = np.logical_and(np.logical_or(idx1, idx2), np.logical_not(np.isin(cell_labels, labels)))
            # divergence
            if np.sum(idx1_mutual) >= cal_min and np.sum(idx2_mutual) >= cal_min:
                div_pq.append(max(estimate(div_ent_code[idx1_mutual, :], div_ent_code[idx2_mutual, :], cal_min), 0))
                div_qp.append(max(estimate(div_ent_code[idx2_mutual, :], div_ent_code[idx1_mutual, :], cal_min), 0))
            # entropy
            if (sum(idx_specific) > 0):
                ent_tmp = cal_entropy(div_ent_code, idx_specific, dataset_labels)
                ent.append(sum(ent_tmp) / len(ent_tmp))
    if len(ent) == 0:  # if no dataset specific cell types, store entropy as -1
        ent.append(-1)

    # calculate silhouette_score
    if sil_code.shape[1] > sil_dim:
        sil_code = PCA(n_components=sil_dim).fit_transform(sil_code)
    sil_scores = silhouette_samples(sil_code, cell_labels, metric=sil_dist)

    # average for scores
    div_score = (sum(div_pq) / len(div_pq) + sum(div_qp) / len(div_qp)) / 2
    ent_score = sum(ent) / len(ent)
    sil_score = sum(sil_scores) / len(sil_scores)

    return div_score, ent_score, sil_score