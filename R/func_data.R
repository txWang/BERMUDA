library(Seurat)
library("Matrix")

# read a scRNA-seq dataset from a csv file
# Input:
# filename: name of the dataset csv file
#   gene expression levels quantified using TPM
#   first row is sample labels, integer
#   second row is cell labels, integer
#   the rest rows are gene expression data
#   the first column are gene symbols
# Output:
# dataset: a list contains 3 elements
#   data: gene expression data, rownames are gene names
#   sample_labels: integer labels of samples
#   cell_labels: integer labels of cell types
read_dataset <- function(filename) {
  dat_csv = read.table(filename, sep = ",", header = F, row.names = 1)
  colnames(dat_csv) = paste0("cell_",as.character(1:ncol(dat_csv)))
  sample_labels = as.integer(dat_csv[1,])
  cell_labels = as.integer(dat_csv[2,])
  data = dat_csv[3:nrow(dat_csv),]
  dataset = list("data" = data, "sample_labels" = sample_labels, "cell_labels" = cell_labels)
  
  return(dataset)
}


# apply seurat_preprocessing on dataset
# Input:
# dataset: a dataset list, contains 3 elements
#          data, sample_labels, cell_labels
# Output:
# data_seurat: a Seurat object with clustering results
seurat_preprocessing <- function(dataset, dataset_name) {
  data_seurat = CreateSeuratObject(raw.data = dataset$data, project = dataset_name, min.cells = 5)
  data_seurat@meta.data$sample_labels = dataset$sample_labels
  data_seurat@meta.data$cell_labels = dataset$cell_labels
  data_seurat = NormalizeData(data_seurat, display.progress = FALSE)
  data_seurat = FindVariableGenes(data_seurat, do.plot = F, display.progress = FALSE)
  data_seurat <- ScaleData(data_seurat, display.progress = FALSE)
  data_seurat = RunPCA(object = data_seurat, pc.genes = data_seurat@var.genes, do.print = F)
  data_seurat <- FindClusters(data_seurat, reduction.type = "pca", resolution = 0.6, dims.use = 1:20,
                              print.output = 0)
  data_seurat <- RunTSNE(data_seurat, dims.use = 1:20, do.fast = T, check_duplicates = FALSE)
  
  return (data_seurat)
}


# write a scRNA-seq dataset into a csv file
# Input:
#   filename: name of the output data file
#   data: gene expression data, rownames are gene names
#   sample_labels: integer labels of samples (e.g. human_1)
#   cell_labels: integer labels of cell types
# Output:
# a .csv file
#   first row is sample labels, second row is cell labels
#   the rest rows are gene expression data
#   the first column is sample_label, cell_label and gene symbols
write_dataset <- function(filename, data, sample_labels, cell_labels) {
  cell_labels = data.frame(t(cell_labels))
  sample_labels = data.frame(t(sample_labels))
  rownames(cell_labels) = c("cell_label")
  rownames(sample_labels) = c("sample_labels")
  # write csv
  write.table(sample_labels, filename, sep = ",", quote = F, col.names = F, row.names = T)
  write.table(cell_labels, filename, sep = ",", quote = F, col.names = F, row.names = T, append = T)
  write.table(data, filename, sep = ",", quote = F, col.names = F, row.names = T, append = T)
}


# write a scRNA-seq dataset into a csv file, with cluster labels identified by Seurat
# Input:
#   filename: name of the output data file
#   data: gene expression data, rownames are gene names
#   sample_labels: integer labels of samples
#   cell_labels: integer labels of cell type
#   cluster_labels: integer labels of clusters
# Output:
# a .csv file
#   first row are sample labels
#   second row are cell labels
#   thrid row are cluster labels
#   the rest rows are gene expression data
#   the first column are gene symbols
write_dataset_cluster <- function(filename, data, sample_labels, cell_labels, cluster_labels) {
  cell_labels = data.frame(t(cell_labels))
  sample_labels = data.frame(t(sample_labels))
  cluster_labels = data.frame(t(cluster_labels))
  rownames(cell_labels) = c("cell_label")
  rownames(sample_labels) = c("sample_labels")
  rownames(cluster_labels) = c("cluster_labels")
  # write csv
  write.table(sample_labels, filename, sep = ",", quote = F, col.names = F, row.names = T)
  write.table(cell_labels, filename, sep = ",", quote = F, col.names = F, row.names = T, append = T)
  write.table(cluster_labels, filename, sep = ",", quote = F, col.names = F, row.names = T, append = T)
  write.table(data, filename, sep = ",", quote = F, col.names = F, row.names = T, append = T)
}