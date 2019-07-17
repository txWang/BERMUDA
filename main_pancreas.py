#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

from BERMUDA import training, testing
from pre_processing import pre_processing, read_cluster_similarity
from evaluate import evaluate_scores
from helper import cal_UMAP, plot_labels, plot_expr, plot_loss, gen_dataset_idx

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# IMPORTANT PARAMETER
similarity_thr = 0.90 #S_thr in the paper, choose between 0.85-0.9

# nn parameter
code_dim = 20
batch_size = 50 # batch size for each cluster
num_epochs = 2000
base_lr = 1e-3
lr_step = 200  # step decay of learning rates
momentum = 0.9
l2_decay = 5e-5
gamma = 1  # regularization between reconstruction and transfer learning
log_interval = 1
# CUDA
device_id = 0 # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

pre_process_paras = {'take_log': True, 'standardization': True, 'scaling': True}
nn_paras = {'code_dim': code_dim, 'batch_size': batch_size, 'num_epochs': num_epochs,
            'base_lr': base_lr, 'lr_step': lr_step,
            'momentum': momentum, 'l2_decay': l2_decay, 'gamma': gamma,
            'cuda': cuda, 'log_interval': log_interval}

plt.ioff()

if __name__ == '__main__':
    data_folder = 'pancreas/'
    dataset_file_list = ['muraro_seurat.csv', 'baron_seurat.csv']
    cluster_similarity_file = data_folder + 'pancreas_metaneighbor.csv'
    code_save_file = data_folder + 'code_list.pkl'
    dataset_file_list = [data_folder+f for f in dataset_file_list]
    

    # read data
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cluster_pairs = read_cluster_similarity(cluster_similarity_file, similarity_thr)
    nn_paras['num_inputs'] = len(dataset_list[0]['gene_sym'])

    # training
    model, loss_total_list, loss_reconstruct_list, loss_transfer_list = training(dataset_list, cluster_pairs, nn_paras)
    plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, data_folder+'loss.png')
    # extract codes
    code_list = testing(model, dataset_list, nn_paras)
    with open(code_save_file,'wb') as f:
        pickle.dump(code_list, f)

    # combine datasets in dataset_list
    pre_process_paras = {'take_log': True, 'standardization': False, 'scaling': False}  # lof TPM for uncorrected data
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cell_list = []
    data_list = []
    cluster_list = []
    for dataset in dataset_list:
        data_list.append(dataset['gene_exp'])
        cell_list.append(dataset['cell_labels'])
        cluster_list.append(dataset['cluster_labels'])
    cell_labels = np.concatenate(cell_list)
    dataset_labels = gen_dataset_idx(data_list)
    cluster_labels = np.concatenate(cluster_list)

    # calculate UMAP
    with open(code_save_file,'rb') as f:
        code_list = pickle.load(f)
    code = np.concatenate(code_list, axis=1).transpose()
    data = np.concatenate(data_list, axis=1).transpose()
    umap_code = cal_UMAP(code)
    umap_uncorrected = cal_UMAP(data)

    # plot results
    cell_type_dict = {1:'alpha', 2:'beta', 3:'delta', 4:'acinar', 5:'ductal', 6:'endo', 7:'gamma', 8:'epsilon'}
    dataset_dict = {1: 'Muraro', 2:'Baron'}
    plot_labels(umap_code, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'ae_cell_type.png')
    plot_labels(umap_uncorrected, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'uncorrected_cell_type.png')
    plot_labels(umap_code, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'ae_dataset.png')
    plot_labels(umap_uncorrected, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'uncorrected_dataset.png')

    # evaluate using proposed metrics
    num_datasets = len(dataset_file_list)
    print('ae')
    div_score, ent_score, sil_score = evaluate_scores(umap_code, code, cell_labels, dataset_labels, num_datasets, 20, 20, 'cosine')
    print('divergence_score: {:.3f}, entropy_score: {:.3f}, silhouette_score: {:.3f}'.format(div_score, ent_score, sil_score))
    print('uncorrected')
    div_score, ent_score, sil_score = evaluate_scores(umap_uncorrected, data, cell_labels, dataset_labels, num_datasets, 20, 20, 'cosine')
    print('divergence_score: {:.3f}, entropy_score: {:.3f}, silhouette_score: {:.3f}'.format(div_score, ent_score, sil_score))