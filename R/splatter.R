library(splatter)
source("func_data.R")

num_cells = c(2000, 1000)
group_prob = c(0.4, 0.3, 0.2, 0.1)

num_batches = length(num_cells)
num_clusters = length(group_prob)

params <- newSplatParams()
simulated_data <- splatSimulate(batchCells = num_cells, group.prob = group_prob,
                            method = "groups", verbose = FALSE)
dat = counts(simulated_data)
# count to TMP
dat = apply(dat,2,function(x) (x*10^6)/sum(x))
# labels
batch = simulated_data@colData$Batch
cell = simulated_data@colData$Group
batch = unlist(lapply(batch,function(x) strtoi(substr(x, 6, 100))))
cell = unlist(lapply(cell,function(x) strtoi(substr(x, 6, 100))))

# save simulated data
idx1 = which(batch == 1)
idx2 = which(batch == 2)
write_dataset("splatter_batch_1.csv", dat[,idx1], rep(1, ncol(dat[,idx1])), cell[idx1])
write_dataset("splatter_batch_2.csv", dat[,idx2], rep(1, ncol(dat[,idx2])), cell[idx2])