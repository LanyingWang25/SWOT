

library(ggplot2)
library(scatterpie)

corlor_vec_cts <- c("#c1f1fc", "#FFFFCC", "#ffc2e5", "#CCFF99", "#CCCCFF",
                    "#81ceff", "#FFb500", "#ff78cb", "#19e3b1", "#a26eea",
                    "#0000FF", "#d2ea32", '#d52685', "#a0ac48", "#9933CC",
                    "#0C7EF3", "#f9a270", "#ff5454", "#007f4e", "#c377e0",
                    "#333399", "#ddb53e", "#FF00FF", "#72FD93", "#5707F8",
                    "#0cb9c1", "#FFFF00", "#FD72DD", "#28c101", "#9F72FF")


Plot_Scatterpie <- function(ctProp, st_xy, pie_pointsize, color_ct=NULL){
  
  if (!is.data.frame(ctProp)) stop('ERROR: ctProp must be a DataFrame!')
  if (!is.data.frame(st_xy)) stop('ERROR: st_xy must be a DataFrame!')
  if (any(rownames(ctProp) != rownames(st_xy)) == TRUE)
    stop('ERROR: The spot name should be same!')
  if (('x' %in% colnames(st_xy)) & ('y' %in% colnames(st_xy))){
    st_xy <- rename(st_xy, c('X' = 'x', 'Y' = 'y'))
  }
  st_xy <- st_xy[c('X', 'Y')]
  
  celltype = colnames(ctProp)
  
  spot_ctPro_xy <- as.data.frame(cbind(ctProp, st_xy))
  
  if (is.null(color_ct)){
    color_ct <- corlor_vec_cts[1:(length(celltype))]
  }
  else
    color_ct = color_ct
  
  Plot_spot_ctpro <- ggplot() +
    geom_scatterpie(data = spot_ctPro_xy,
                    aes(X, Y),
                    cols = celltype,
                    pie_scale = pie_pointsize) +
    scale_fill_manual(values = color_ct) +
    theme_bw() +
    labs(x="Spot_X", y="Spot_Y") + 
    ggtitle("ScatterPie of cell-type compositions") +
    theme(plot.title = element_text(hjust=0.5)) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank())
  print(Plot_spot_ctpro)
}


Plot_CTProp <- function(ctProp, st_xy, ct, point_size){

  if (!is.data.frame(ctProp)) stop("ERROR: ctProp must be a DataFrame!")
  if (!is.data.frame(st_xy)) stop("ERROR: st_xy must be a DataFrame!")
  if (any(rownames(ctProp) != rownames(st_xy)) == TRUE)
    stop('ERROR: The spot name should be same!')
  if (('x' %in% colnames(st_xy)) & ('y' %in% colnames(st_xy))){
    st_xy <- rename(st_xy, c('X' = 'x', 'Y' = 'y'))
  }
  st_xy <- st_xy[c('X', 'Y')]
  if (!ct %in% colnames(ctProp))
    stop(paste("The cell type", ct, "is not in the cell type vector."))

  celltypes <- colnames(ctProp)
  st_ctPro_xy <- as.data.frame(cbind(ctProp, st_xy))
  stX <- st_ctPro_xy$X
  stY <- st_ctPro_xy$Y

  st_ctPro_xy_tmp <- st_ctPro_xy[, c('X', 'Y', ct)]
  ctp <- st_ctPro_xy_tmp[, ct]

  plotgrid_tmp <- ggplot(st_ctPro_xy_tmp, aes(stX, stY)) +
    geom_point(aes(fill = ctp), size = point_size, shape = 21) +
    scale_fill_gradient2(low = "#FFFFFF", high = '#FF0000', limits=c(0,1)) +
    theme_bw() +
    labs(x="Spot_X", y="Spot_Y", fill='Proportion') +
    ggtitle(paste0("Spatial distribution of ", ct)) + 
    theme(plot.title = element_text(hjust=0.5)) +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank())
  print(plotgrid_tmp)
}


FindMarkerGenes <- function(sc_exp, sc_meta, n_tops=200, file_path, save = TRUE){

  if (!is.data.frame(sc_exp)) stop('ERROR: sc_exp must be a DataFrame!')
  if (!is.data.frame(sc_meta)) stop('ERROR: sc_meta must be a DataFrame!')
  if (!('celltype' %in% colnames(sc_meta))) stop('ERROR: sc_meta must have a "celltype" column!')

  library(Seurat)
  sc_Seu <- CreateSeuratObject(counts = as(as.matrix(sc_exp), 'sparseMatrix'))
  sc_Seu@meta.data$celltype <- sc_meta$celltype
  sc_Seu %>%
    Seurat::NormalizeData() %>%
    Seurat::FindVariableFeatures() %>%
    Seurat::ScaleData() -> sc_Seu
  Seurat::Idents(object = sc_Seu) <- sc_Seu@meta.data$celltype
  markers_all <- Seurat::FindAllMarkers(sc_Seu, only.pos = TRUE, logfc.threshold = 0.1, min.pct = 0.01)
  markers_top <- markers_all %>%
    group_by(cluster) %>%
    top_n(n = n_tops, wt = avg_log2FC)
  
  markers_top_df <- matrix(ncol = length(unique(markers_top$cluster)), nrow = n_tops)
  colnames(markers_top_df) <- unique(markers_top$cluster)
  markers_top_df <- as.data.frame(markers_top_df)
  for(ct in colnames(markers_top_df)){
    if (length(which(markers_top$cluster == ct)) < n_tops){
      markers_top_df[1:length(which(markers_top$cluster == ct)), ct] <- markers_top[which(markers_top$cluster == ct), 'gene']
      tmp <- unlist(markers_top[which(markers_top$cluster == ct), 'gene'])
      set.seed(123)
      new_vec <- sample(x= tmp, 
                        size = n_tops - length(which(markers_top$cluster == ct)), 
                        replace = TRUE)
      leng = length(which(markers_top$cluster == ct))
      leng1 = leng+1
      markers_top_df[leng1:n_tops, ct] <- new_vec
    } else {
      markers_top_df[,ct] <- markers_top[which(markers_top$cluster == ct), 'gene']
      }
  }

  if (save==TRUE){
    file_name_all <- paste0(file_path, 'markers_all_', find_method, '.csv')
    file_name_top <- paste0(file_path, 'markers_top_', find_method, '.csv')
    file_name_df_top <- paste0(file_path, 'markers_top_df_', find_method, '.csv')
    write.csv(markers_all, file_name_all)
    write.csv(markers_top, file_name_top)
    write.csv(markers_top_df, file_name_df_top)
    cat("The marker genes is saved in Markers",'\n')
  }
  #markers <- list(markers_all, markers_top, markers_top_df)
  return(markers_top_df)
}


Plot_Topmarkergenes <- function(st_exp, st_xy, interest_cts, markers_df, n_tops=100, 
                                NumCols = 2, Size=1, colors = colors, 
                                scale_color_optim='viridis', scale_color_direction = -1){
  
  if (!is.data.frame(st_exp)) stop('ERROR: sc_exp must be a DataFrame!')
  if (!is.data.frame(st_xy)) stop('ERROR: st_xy must be a DataFrame!')
  if (!is.vector(interest_cts)) stop('ERROR: interest_cts must be a vector!')
  if (!is.data.frame(markers_df)) stop('ERROR: markers_df must be a DataFrame!')
  if (!all(is.element(interest_cts, colnames(markers_df)))) stop('ERROR: interest_cts must be a subset of colnames(markers_df)!')
  if (n_tops > dim(markers_df)[1]) stop('ERROR: the n_tops must be ≤ the number of genes (rows) in markers_df!')
  if (sum(colnames(st_exp)==rownames(st_xy))!= nrow(st_xy)) 
    stop("The colnames of st_exp data does not match with the rownames of st_xy data!")
  if (('x' %in% colnames(st_xy)) & ('y' %in% colnames(st_xy))){
    st_xy <- rename(st_xy, c('X' = 'x', 'Y' = 'y'))
  }
  
  library(stringr)
  
  ct_exp <- matrix(NA, nrow = length(interest_cts), ncol = dim(st_exp)[2], 
                   dimnames = list(interest_cts, colnames(st_exp)))
  ct_exp <- as.data.frame(ct_exp)
  
  for (ct in interest_cts){
    top_genes <- markers_df[1:n_tops, ct]
    st_exp_top_genes <- st_exp[top_genes,]
    ct_exp[ct, ] <- colSums(st_exp_top_genes)
  }
  rownames(ct_exp) <- str_c(rownames(ct_exp), '_genes')
  
  #expression = sweep(ct_exp,2,colSums(ct_exp),"/")
  expression = ct_exp
  ctg_select <- rownames(ct_exp)
  Data = NULL
  for(i in 1:length(ctg_select)){
    ct_gene = ctg_select[i]
    ind = which(toupper(rownames(expression)) == toupper(ct_gene))
    df = as.numeric(expression[ind,])
    names(df) = colnames(expression)
    df = (df - min(df)) / (max(df) - min(df))
    d = data.frame(value = df,
                   x=as.numeric(st_xy$X),
                   y = as.numeric(st_xy$Y))
    d$ct_gene = ct_gene
    Data = rbind(Data,d)
  }
  
  p = suppressMessages(ggplot(Data, aes(x, y)) + 
                         geom_point(aes(color = value),size = Size,
                                    position = position_dodge2(width = 0.001, padding = 0))+
                         scale_x_discrete(expand = c(0, 1)) + 
                         scale_y_discrete(expand = c(0, 1)) +
                         coord_equal()+
                         facet_wrap(~ct_gene,ncol = NumCols)+
                         theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"),
                               legend.position="right",
                               panel.background = element_blank(),
                               plot.background = element_blank(),
                               panel.border = element_rect(colour = "grey89", fill=NA, linewidth=0.5),
                               axis.text =element_blank(),
                               axis.ticks =element_blank(),
                               axis.title =element_blank(),
                               legend.title=element_text(size = 10),
                               legend.text=element_text(size = 10),
                               strip.text = element_text(size = 10),
                               legend.key = element_rect(colour = "transparent", fill = "white"),
                               legend.key.size = unit(1.0, 'cm'))+ 
                         guides(color = guide_colourbar(title = "Expression")))
  if(is.null(colors)){
    p <- p + scale_color_viridis_c(option = scale_color_optim, labels = c("0","0.25","0.5","0.75","1.0"), 
                                   alpha = 0.8, direction = scale_color_direction)
  }else{
    p <- p + scale_color_gradientn(colours = colors)
  }
  return(p)
}

