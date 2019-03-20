# Code from
# https://github.com/maggiecrow/MetaNeighbor

run_MetaNeighbor_US<-function(vargenes, data, celltypes, pheno){
  
  cell.labels=matrix(0,ncol=length(celltypes),nrow=dim(pheno)[1])
  rownames(cell.labels)=colnames(data)
  colnames(cell.labels)=celltypes
  for(i in 1:length(celltypes)){
    type=celltypes[i]
    m<-match(pheno$Celltype,type)
    cell.labels[!is.na(m),i]=1
  }
  
  m<-match(rownames(data),vargenes)
  cor.dat=cor(data[!is.na(m),],method="s")
  rank.dat=cor.dat*0
  rank.dat[]=rank(cor.dat,ties.method="average",na.last = "keep")
  rank.dat[is.na(rank.dat)]=0
  rank.dat=rank.dat/max(rank.dat)
  sumin    =  (rank.dat) %*% cell.labels
  sumall   = matrix(apply(rank.dat,2,sum), ncol = dim(sumin)[2], nrow=dim(sumin)[1])
  predicts = sumin/sumall
  
  cell.NV=matrix(0,ncol=length(celltypes),nrow=length(celltypes))
  colnames(cell.NV)=colnames(cell.labels)
  rownames(cell.NV)=colnames(cell.labels)
  
  for(i in 1:dim(cell.labels)[2]){
    predicts.temp=predicts
    m<-match(pheno$Celltype,colnames(cell.labels)[i])
    study=unique(pheno[!is.na(m),"Study_ID"])
    m<-match(pheno$Study_ID,study)
    pheno2=pheno[!is.na(m),]
    predicts.temp=predicts.temp[!is.na(m),]
    predicts.temp=apply(abs(predicts.temp), 2, rank,na.last="keep",ties.method="average")
    filter=matrix(0,ncol=length(celltypes),nrow=dim(pheno2)[1])
    m<-match(pheno2$Celltype,colnames(cell.labels)[i])
    filter[!is.na(m),1:length(celltypes)]=1
    negatives = which(filter == 0, arr.ind=T)
    positives = which(filter == 1, arr.ind=T)
    predicts.temp[negatives] <- 0
    np = colSums(filter,na.rm=T)
    nn = apply(filter,2,function(x) sum(x==0,na.rm=T))
    p =  apply(predicts.temp,2,sum,na.rm=T)
    cell.NV[i,]= (p/np - (np+1)/2)/nn
  }
  
  cell.NV=(cell.NV+t(cell.NV))/2
  return(cell.NV)
  
}