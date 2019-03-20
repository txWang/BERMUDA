# BERMUDA: Batch-Effect ReMoval Using Deep Autoencoders
Tongxin Wang, Travis S Johnson, Wei Shao, Jie Zhang and Kun Huang

Codes and data for using BERMUDA, a novel transfer-learning-based method for batch-effect correction in single cell RNA sequencing (scRNA-seq) data.

![BERMUDA](https://github.com/txWang/BERMUDA/blob/master/BERMUDA.png "BERMUDA")

## Dependencies
* Python 3.6.5
* scikit-learn 0.19.1
* pyTorch 0.4.0
* imbalanced-learn 0.3.3
* rpy2 2.9.4
* universal-divergence 0.2.0

## Files
*main_pancreas.py*: An Example of combining two pancreas datasets\
*main_pbmc.py*: An Example of combining PBMCs with pan T cells\
*pre_processing.R*: Workflow of detecting clusters using Seurat and identifying similar clusters using MetaNeighbor\
*de_analysis*: Gene differential expression analysis using T-test with Benjamini-Hochberg Procedure on pancreas datasets
