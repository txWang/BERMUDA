# !/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale

def read_csv(filename, take_log):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    dataset = {}
    df = pd.read_csv(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['sample_labels'] = dat[0, :].astype(int)
    dataset['cell_labels'] = dat[1, :].astype(int)
    dataset['cluster_labels'] = dat[2, :].astype(int)
    gene_sym = df[df.columns[0]].tolist()[3:]
    gene_exp = dat[3:, :]
    if take_log:
        gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    return dataset


def read_cluster_similarity(filename, thr):
    """ read cluster similarity matrix, convert into the format of pairs and weights
    first line is cluster label, starting with 1
    Args:
        filename: filename of the cluster similarity matrix
        thr: threshold for identifying corresponding clusters
    Returns:
        cluster_pairs: np matrix, num_pairs by 3 matrix
                        [cluster_idx_1, cluster_id_2, weight]
    """
    df = pd.read_csv(filename, header=None)
    cluster_matrix = df[df.columns[:]].values
    cluster_matrix = cluster_matrix[1:, :]
    # use blocks of zeros to determine which clusters belongs to which datasets
    num_cls = cluster_matrix.shape[0]
    dataset_idx = np.zeros(num_cls, dtype=int)
    idx = 0
    for i in range(num_cls - 1):
        dataset_idx[i] = idx
        if cluster_matrix[i, i + 1] != 0:
            idx += 1
    dataset_idx[num_cls - 1] = idx
    num_datasets = idx + 1

    # only retain pairs if cluster i in dataset a is most similar to cluster j in dataset b
    local_max = np.zeros(cluster_matrix.shape, dtype=int)
    for i in range(num_cls):
        for j in range(num_datasets):
            if dataset_idx[i] == j:
                continue
            tmp = cluster_matrix[i, :] * (dataset_idx == j)
            local_max[i, np.argmax(tmp)] = 1
    local_max = local_max + local_max.T
    local_max[local_max > 0] = 1
    cluster_matrix = cluster_matrix * local_max  # only retain dataset local maximal pairs
    cluster_matrix[cluster_matrix < thr] = 0
    cluster_matrix[cluster_matrix > 0] = 1 # binarize

    # construct cluster pairs
    tmp_idx = np.nonzero(cluster_matrix)
    valid_idx = tmp_idx[0] < tmp_idx[1]  # remove duplicate pairs
    cluster_pairs = np.zeros((sum(valid_idx), 3), dtype=float)
    cluster_pairs[:, 0] = tmp_idx[0][valid_idx] + 1
    cluster_pairs[:, 1] = tmp_idx[1][valid_idx] + 1
    for i in range(cluster_pairs.shape[0]):
        cluster_pairs[i, 2] = cluster_matrix[int(cluster_pairs[i, 0] - 1), int(cluster_pairs[i, 1] - 1)]

    return cluster_pairs


def remove_duplicate_genes(gene_exp, gene_sym):
    """ Remove duplicate gene symbols in a dataset
    Chooses the one with highest mean value when there are duplicate genes
    Args:
        gene_exp: np matrix, num_genes by num_cells
        gene_sym: length num_cells
    Returns:
        gene_exp
        gene_sym
    """
    dic = {}  # create a dictionary of gene_sym to identify duplicate
    for i in range(len(gene_sym)):
        if not gene_sym[i] in dic:
            dic[gene_sym[i]] = [i]
        else:
            dic[gene_sym[i]].append(i)
    if (len(dic) == len(gene_sym)):  # no duplicate
        return gene_exp, gene_sym

    remove_idx = []  # idx of gene symbols that will be removed
    for sym, idx in dic.items():
        if len(idx) > 1:  # duplicate exists
            # print('duplicate! ' + sym)
            remain_idx = idx[np.argmax(np.mean(gene_exp[idx, :], axis=1))]
            for i in idx:
                if i != remain_idx:
                    remove_idx.append(i)
    gene_exp = np.delete(gene_exp, remove_idx, 0)
    for idx in sorted(remove_idx, reverse=True):
        del gene_sym[idx]
    # print("Remove duplicate genes, remaining genes: {}".format(len(gene_sym)))

    return gene_exp, gene_sym


def intersection_idx(lists):
    """ intersection of multiple lists. Returns intersection and corresponding indexes
    Args:
        lists: list of lists that need to intersect
    Returns:
        intersect_list: list of intersection result
    """
    idx_dict_list = []  # create index dictionary
    for l in lists:
        idx_dict_list.append(dict((k, i) for i, k in enumerate(l)))
    intersect_set = set(lists[0])  # create intersection result
    for i in range(1, len(lists)):
        intersect_set = set(lists[i]).intersection(intersect_set)
    intersect_list = list(intersect_set)
    # generate corresponding index of intersection
    idx_list = []
    for d in idx_dict_list:
        idx_list.append([d[x] for x in intersect_list])

    return intersect_list, idx_list


def intersect_dataset(dataset_list):
    """ Only retain the intersection of genes among multiple datasets
    Args:
        dataset_list: list of datasets
    Returns:
        intersect_dataset_list: list of after intersection pf gene symbols
    """
    dataset_labels = ['sample_labels', 'cell_labels', 'cluster_labels']  # labels in a dataset
    intersect_dataset_list = []
    gene_sym_lists = []
    for i, dataset in enumerate(dataset_list):
        gene_sym_lists.append(dataset['gene_sym'])
    # intersection of gene symbols
    gene_sym, idx_list = intersection_idx(gene_sym_lists)
    # print("Intersection of genes: {}".format(len(gene_sym)))
    # only retain the intersection of genes in each dataset
    for dataset, idx in zip(dataset_list, idx_list):
        dataset_tmp = {'gene_exp': dataset['gene_exp'][idx,:], 'gene_sym': gene_sym}
        for l in dataset_labels:
            if l in dataset:
                dataset_tmp[l] = dataset[l]
        intersect_dataset_list.append(dataset_tmp)

    return intersect_dataset_list


def pre_processing(dataset_file_list, pre_process_paras):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    standardization = pre_process_paras['standardization']
    scaling = pre_process_paras['scaling']

    dataset_list = []
    for data_file in dataset_file_list:
        dataset = read_csv(data_file, take_log)
        if standardization:
            scale(dataset['gene_exp'], axis=1, with_mean=True, with_std=True, copy=False)
        if scaling:  # scale to [0,1]
            minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)
        dataset_list.append(dataset)
    dataset_list = intersect_dataset(dataset_list)  # retain intersection of gene symbols

    return dataset_list

if __name__ == '__main__':
    dataset_file_list = ['data/muraro_seurat.csv', 'data/baron_human_seurat.csv']
    pre_process_paras = {'take_log': True, 'standardization': True, 'scaling': True}
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    print()