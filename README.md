# BERMUDA: Batch Effect ReMoval Using Deep Autoencoders
Tongxin Wang, Travis S Johnson, Wei Shao, Zixiao Lu, Bryan R Helm, Jie Zhang and Kun Huang

Codes and data for using [BERMUDA](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6 "BERMUDA"), a novel transfer-learning-based method for batch-effect correction in single cell RNA sequencing (scRNA-seq) data.

![BERMUDA](https://github.com/txWang/BERMUDA/blob/master/BERMUDA.png "BERMUDA")
Overview of BERMUDA for removing batch effects in scRNA-seq data.     
<sup>a. The workflow of BERMUDA. Circles and triangles represent cells from Batch 1 and Batch 2, respectively. Different colors represent different cell types. A graph-based clustering algorithm was first applied on each batch individually to detect cell clusters. Then, MetaNeighbor, a method based on Spearman correlation, was used to identify similar clusters between batches. An autoencoder was subsequently trained to perform batch correction on the code of the autoencoder. The code of the autoencoder is a low-dimensional representation of the original data without batch effects and can be used for further analysis.     
b. Training an autoencoder to remove batch effects. The blue solid lines represent training with the cells in Batch 1 and the blue dashed lines represent training with cells in Batch 2. The black dashed lines represent the calculation of losses. The loss function we optimized contains two components: the reconstruction loss between the input and the output of the autoencoder, and the MMD-based transfer loss between the codes of similar clusters.</sup>

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
*R/pre_processing.R*: Workflow of detecting clusters using Seurat and identifying similar clusters using MetaNeighbor\
*R/gaussian.R*: Simulate data based on 2D Gaussian distributions\
*R/splatter.R*: Simulate data using Splatter package

## Cite
Wang, T., Johnson, T.S., Shao, W. et al. BERMUDA: a novel deep transfer learning method for single-cell RNA sequencing batch correction reveals hidden high-resolution cellular subtypes. Genome Biol 20, 165 (2019) doi:10.1186/s13059-019-1764-6
