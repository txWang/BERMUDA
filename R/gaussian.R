# Code modified from
# https://github.com/MarioniLab/MNN2017/blob/master/Simulations/simulateBatches.R
source("func_read_data.R")

# This script generates some (highly synthetic!) expression data with a batch effect 
# and uneven population composition between batches.
# this.dir <- dirname(parent.frame(2)$ofile)
# setwd(this.dir)

ncells <- 2000  # Number of cells
ngenes <- 100  # Number of genes

# Our simulation involves three cell types/components.
# Cells are distributed according to a bivariate normal in a 2-D biological subspace. 
# Each cell type has a different x/y center and a different SD.
num_clust = 4
xmus <- c(0,5,5,0)
xsds <- c(0.8,0.1,0.4,0.2)
ymus <- c(5,5,0,0)
ysds <- c(0.8,0.1,0.4,0.2)

set.seed(0)
prop1 <- runif(num_clust,0,1)
prop1 = prop1/sum(prop1)
set.seed(999)
prop2 <- runif(num_clust,0,1)
prop2 = prop2/sum(prop2)

# Note that the different centers should not lie on the same y=mx line; this represents populations that differ only in library size. 
# Such differences should not be present in normalized data, and will be eliminated by the cosine normalization step.
# The centers above are chosen so as to guarantee good separation between the different components.

#####################################
# Generating data for batch 1, with a given proportion of cells in each component. 
comp1 <- sample(1:num_clust, prob=prop1, size=ncells, replace=TRUE)

# Sampling locations for cells in each component.
set.seed(0)
samples1 <- cbind(rnorm(n=ncells, mean=xmus[comp1],sd=xsds[comp1]),
                  rnorm(n=ncells, mean=ymus[comp1],sd=ysds[comp1]))

# Random projection to D dimensional space, to mimic high-dimensional expression data.
set.seed(0)
proj <- matrix(rnorm(ngenes*ncells), nrow=ngenes, ncol=2)
A1 <- samples1 %*% t(proj)

# Add normally distributed noise.
A1 <- A1 + rnorm(ngenes*ncells)
rownames(A1) <- paste0("Cell", seq_len(ncells), "-1")
colnames(A1) <- paste0("Gene", seq_len(ngenes))

#####################################
# Setting proportions of each of the three cell types in batch 2.
comp2 <- sample(1:num_clust, prob=prop2, size=ncells, replace=TRUE) 
  
# Sampling locations for cells in each component.  
set.seed(0)
samples2 <- cbind(rnorm(n=ncells, mean=xmus[comp2], sd=xsds[comp2]),
                  rnorm(n=ncells, mean=ymus[comp2], sd=ysds[comp2]))
  
# Random projection, followed by adding batch effects and random noise.
A2 <- samples2 %*% t(proj) 
A2 <- A2 + matrix(rep(rnorm(ngenes), each=ncells), ncol=ngenes) # gene-specific batch effect (genes are columns)
A2 <- A2 + rnorm(ngenes*ncells) # noise
rownames(A2) <- paste0("Cell", seq_len(ncells), "-2")
colnames(A2) <- paste0("Gene", seq_len(ngenes))

#####################################
# save simulated data
write_dataset("gaussian_batch_1.csv", t(A1), rep(1, ncol(t(A1))), comp1)
write_dataset("gaussian_batch_2.csv", t(A2), rep(1, ncol(t(A2))), comp2)

