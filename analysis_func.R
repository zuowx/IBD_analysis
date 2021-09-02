library(ggpubr)
library(ggplot2)

plot.shannon <- function(df, plot_title) {
  p <- ggplot(df, aes(x=Phenotype,y=shannon)) +
    geom_boxplot(aes(colour=Phenotype),width=0.5,size=0.5,outlier.fill="white",outlier.color="white")+ 
    geom_jitter(aes(colour=Phenotype,fill=Phenotype),width =0.2,shape = 21,size=2)+
    scale_y_continuous(name = "Shannon Index",limits = c(0,max(df$shannon)+0.5))+
    scale_x_discrete(name = "Phenotypes")+ 
    labs(title=plot_title)+
    stat_compare_means(comparisons=list(c("Healthy","UC")),correct=FALSE,label="p.format",method = "wilcox.test")+
    theme_bw()+
    theme(panel.grid.minor = element_blank())+
    theme(legend.position="none")
  return(p)
}


library(vegan)
library(ape)
plot.pcoa <- function(filename,metadata,plot_title){
  
  # feature table
  feature_table <- read.csv(filename,row.names=1, check.names = F)
  
  # filter the samples
  feature_table <- feature_table[,colnames(feature_table)%in%metadata$SampleID]
  
  # normalize the feature table
  feature_table <- as.data.frame(apply(feature_table,2,function(x) x/sum(x)))
  
  # compute the bray curtis distance
  beta_diversity <- as.matrix(vegdist(t(feature_table),method = "bray"))
  
  # permanova
  metadata <- metadata[rownames(beta_diversity),]
  permanova <- adonis(beta_diversity~SampleType+Gender+Age, data=metadata, permutations=1000)
  r2 <- permanova$aov.tab["SampleType","R2"]
  p.value <- permanova$aov.tab["SampleType","Pr(>F)"]
  
  # annotate the r2 and p value in the figure
  r2 <- sprintf("italic(R^2) == %.3f",r2)
  p.value <- sprintf("italic(p) == %.3f",p.value) 
  permanova_labels <- data.frame(r2=r2,p.value=p.value,stringsAsFactors = FALSE)
  
  # pcoa plot
  PCOA <- pcoa(as.dist(beta_diversity))
  # data frame for pcoa plot
  pcoa_df <- as.data.frame(PCOA$vectors[,1:2])
  pcoa_df$Axis.1 <- -pcoa_df$Axis.1
  pcoa_df$SampleID <- rownames(pcoa_df)
  pcoa_df <- merge(pcoa_df,metadata,by="SampleID")
  # axis
  pro1 <- as.numeric(sprintf("%.3f",PCOA$values[,"Relative_eig"][1]))*100  
  pro2 <- as.numeric(sprintf("%.3f",PCOA$values[,"Relative_eig"][2]))*100
  # plot
  pcoa_plot <- ggplot(pcoa_df,aes(x=Axis.1,y=Axis.2,col=SampleType)) +
    geom_point(size=3) +
    xlim(-0.7,0.6) + ylim(-0.6,0.4) +
    labs(x=paste0("PCOA1(",pro1,"%)"), y=paste0("PCOA2(",pro2,"%)"),title=plot_title) +
    
    geom_vline(aes(xintercept=0),linetype="dotted") +
    geom_hline(aes(yintercept=0),linetype="dotted") +
    stat_ellipse(aes(group=SampleType,col=SampleType), level = 0.8, show.legend=FALSE)+
    annotate("text",label="Phenotypes:",x=-0.65,y=-0.4,size=4,hjust=0)+
    geom_text(data=permanova_labels,mapping=aes(x=-0.65,y=-0.47,label=r2),parse=TRUE,inherit.aes=FALSE,size=4,hjust=0)+
    geom_text(data=permanova_labels,mapping=aes(x=-0.65,y=-0.54,label=p.value),parse=TRUE,inherit.aes=FALSE,size=4,hjust=0)+
    theme_bw() +
    theme(legend.title=element_blank(), legend.text=element_text(size=12))+
    theme(title = element_text(size = 14))+
    theme(axis.title = element_text(size = 16),axis.text = element_text(size = 12,colour="black"))+
    theme(legend.justification=c(0.02,0.98), legend.position=c(0.02,0.98),legend.background = element_rect(fill = NA))
  
  # return
  return(pcoa_plot)
}

lm_analysis <- function(df) {
  lm_fit <- lm(log(shannon)~Phenotype+Age+Gender, data=df)
  print(lm_result <- summary(lm_fit))
  return(lm_fit)
}

library(RcppCNPy)
library(ecodist)

mrm_analysis <- function(distance_matrix, sample_type_D1, sample_type_D2, age_D, gender_D1, gender_D2) {
  mrm_result <- MRM(as.dist(log(distance_matrix)) ~ as.dist(sample_type_D1)+as.dist(sample_type_D2)+as.dist(age_D)+as.dist(gender_D1)+as.dist(gender_D2)-1)
  print(mrm_result)
  return(mrm_result)
}

