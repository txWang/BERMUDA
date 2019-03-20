import numpy as np
from scipy.stats import ttest_ind
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
stats = importr('stats')
from pre_processing import pre_processing


def de_p_value(gene_exp, cluster_labels, cluster_1, cluster_2):
    ''' Calculate adjusted p values of gene differential expression analysis
    Args:
        gene_exp: num_genes * num_cells, pre-processed using logTPM
        cluster_1: one cell cluster for differential expression analysis
        cluster_2: one cell cluster for differential expression analysis
    Returns:
         adj_p_val: adjusted p-value of each gene using T-test
    '''
    idx_1 = np.isin(cluster_labels, cluster_1)
    idx_2 = np.isin(cluster_labels, cluster_2)
    p_val = np.zeros(len(gene_sym))
    for i in range(len(gene_sym)):
        p_val[i] = ttest_ind(gene_exp[i, idx_1], gene_exp[i, idx_2]).pvalue
    adj_p_val = stats.p_adjust(FloatVector(p_val), method='BH')
    adj_p_val = np.asarray(adj_p_val)
    adj_p_val[np.isnan(p_val)] = 1
    return adj_p_val

if __name__ == '__main__':
    p_thr = 1e-50 # cutoff of p values
    data_folder = 'pancreas/'
    dataset_file_list = ['muraro_seurat.csv', 'baron_seurat.csv']
    dataset_file_list = [data_folder + f for f in dataset_file_list]
    pre_process_paras = {'take_log': True, 'standardization': False, 'scaling': False}

    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    gene_exp_list = []
    cluster_list = []
    for dataset in dataset_list:
        gene_exp_list.append(dataset['gene_exp'])
        cluster_list.append(dataset['cluster_labels'])
    gene_exp = np.concatenate(gene_exp_list, axis=1)
    gene_sym = dataset_list[0]['gene_sym']
    cluster_labels = np.concatenate(cluster_list)

    # alpha
    cluster_11 = [1, 3]
    cluster_12 = [10]
    cluster_21 = [12]
    cluster_22 = [10]
    p_val_1 = de_p_value(gene_exp, cluster_labels, cluster_11, cluster_12)
    p_val_2 = de_p_value(gene_exp, cluster_labels, cluster_21, cluster_22)
    idx = np.argsort(p_val_1)
    print('\nAlpha cells\n')
    for i in range(len(p_val_1)):
        if p_val_1[idx[i]] <= p_thr and p_val_2[idx[i]] <= p_thr:
            print('{:}\t{:.3E}\t{:.3E}'.format(gene_sym[idx[i]], p_val_1[idx[i]], p_val_2[idx[i]]))

    # beta
    cluster_11 = [2]
    cluster_12 = [13]
    cluster_21 = [14, 15]
    cluster_22 = [13]
    p_val_1 = de_p_value(gene_exp, cluster_labels, cluster_11, cluster_12)
    p_val_2 = de_p_value(gene_exp, cluster_labels, cluster_21, cluster_22)
    idx = np.argsort(p_val_1)
    print('\nBeta cells\n')
    for i in range(len(p_val_1)):
        if p_val_1[idx[i]] <= p_thr and p_val_2[idx[i]] <= p_thr:
            print('{:}\t{:.3E}\t{:.3E}'.format(gene_sym[idx[i]], p_val_1[idx[i]], p_val_2[idx[i]]))