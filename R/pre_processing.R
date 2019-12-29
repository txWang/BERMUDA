library(Seurat)
memory.limit(size = 100000)
source("func_data.R")
source("2017-08-28-runMN-US.R")

folder_name = "pancreas"
dataset_names = c("muraro_human", "baron_human")

dataset_list = list()
var_genes = list() #  a subset of highly variable genes
num_cells = 0

# Detect clusters in each dataset using Seurat
for (i in 1:length(dataset_names)) {
  filename = paste(folder_name, paste0(dataset_names[i], ".csv"), sep="/")
  print(paste0("Dataset: ", filename))
  # Seurat
  dataset = read_dataset(filename)
  dataset_list[[i]] = seurat_preprocessing(dataset, dataset_names[[i]])
  var_genes[[i]] = dataset_list[[i]]@var.genes
  num_cells = num_cells + dim(dataset_list[[i]]@data)[2]
}
var_genes = unique(unlist(var_genes))
for (i in 1:length(dataset_list)) {
  var_genes = intersect(var_genes, rownames(dataset_list[[i]]@data))
}

# combine datasets for metaneighbor
# log transformed TPM by Seurat, for metaneighbor
data = matrix(0, nrow = length(var_genes), ncol = num_cells) 
# labels, starting from 1
cluster_label_list = list()
dataset_label_list = list()
cell_idx = 1
cluster_idx = 1
for (i in 1:length(dataset_list)) {
  cell = dim(dataset_list[[i]]@data)[2]
  data[,cell_idx:(cell_idx+cell-1)] = as.matrix(dataset_list[[i]]@data[var_genes,])
  cluster_label_list[[i]] = as.integer(dataset_list[[i]]@meta.data$res.0.6) + cluster_idx
  cluster_idx = max(cluster_label_list[[i]]) + 1
  dataset_label_list[[i]] = rep(i, cell)
  cell_idx = cell_idx + cell
}

# write dataset with shifted cluster labels
# no cluster labels overlap between clusters 
for (i in 1:length(dataset_list)) {
  seurat_csv = paste(folder_name, paste0(dataset_names[i], "_seurat.csv"), sep="/")
  write_dataset_cluster(seurat_csv, as.matrix(dataset_list[[i]]@raw.data[var_genes,]), 
                        dataset_list[[i]]@meta.data$sample_labels,
                        dataset_list[[i]]@meta.data$cell_labels,
                        cluster_label_list[[i]])
}

# Metaneighbor
cluster_labels = unique(unlist(cluster_label_list))
rownames(data) = var_genes
pheno = as.data.frame(list(Celltype = as.character(unlist(cluster_label_list)),
                           Study_ID = as.character(unlist(dataset_label_list))),
                      stringsAsFactors=FALSE)
# run metaneighbor
cluster_similarity =run_MetaNeighbor_US(var_genes, data, cluster_labels, pheno)

# set cluster pairs from the same dataset to 0
for (i in 1:length(dataset_list)) {
  cluster_idx_tmp = unique(cluster_label_list[[i]])
  cluster_similarity[cluster_idx_tmp, cluster_idx_tmp] = 0
}

# order rows and columns
cluster_similarity = cluster_similarity[order(as.numeric(rownames(cluster_similarity))),]
cluster_similarity = cluster_similarity[,order(as.numeric(colnames(cluster_similarity)))]

# write out metaneighbor file
metaneighbor_file = paste(folder_name, paste0(folder_name, "_metaneighbor.csv"), sep="/")
write.table(cluster_similarity, metaneighbor_file, sep = ",", quote = F, col.names = T, row.names = F)
