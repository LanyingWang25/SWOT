
import pandas as pd
from .utils import dis_matrix, spa_weight, sw_usot, cell_mapping

import warnings
warnings.filterwarnings("ignore")

class NonConvergenceError(Exception):
    pass


class SWOTscsm(object):
    """
        This is a SWOTscsm class of SWOT, it products an object for the spatially weighted optimal transport model
        for the inference of cell-type composition and single-cell spatial maps.
        It contains two principal components: an optimal transport module for computing transport plan and
        a cell mapping module for estimating cell-type compositions, cell numbers per spot and spatial coordinates per cell.

        SWOT inputs a gene expression profile and cell type labels in scRNA-seq data,
        as well as a gene expression profile and spatial coordinates in ST data.
        The output of SWOT is a cell-type composition matrix and an inferred single-cell spatial map.

        See more details in our paper.
    """

    def __init__(self, sc_exp, sc_meta, st_exp, st_xy, file_path, use_gpu):
        """
        :param sc_exp: pandas.DataFrame, expression profile of scRNA-seq data with rows being genes and columns being cells.
        :param sc_meta: pandas.DataFrame, cell type information of scRNA-seq data with rows being cells and columns having 'celltype' for labels.
        :param st_exp: pandas.DataFrame, expression profile of ST data with rows being genes and columns being spots.
        :param st_xy: pandas.DataFrame, spatial coordinates information of ST data with rows being spots and columns being 'X' and 'Y'.
        :param file_path: string, file path for saving SWOT results.
        :param use_gpu: whether to use GPU or not?
        """

        self.sc_exp = sc_exp
        self.sc_meta = sc_meta
        self.st_exp = st_exp
        self.st_xy = st_xy
        self.file_path = file_path
        self.use_gpu = use_gpu

        if not isinstance(sc_exp, pd.DataFrame):
            self.sc_exp = pd.DataFrame(sc_exp)
        if not isinstance(sc_meta, pd.DataFrame):
            self.sc_meta = pd.DataFrame(sc_meta)
        if not isinstance(st_exp, pd.DataFrame):
            self.st_exp = pd.DataFrame(st_exp)
        if not isinstance(st_xy, pd.DataFrame):
            self.st_xy = pd.DataFrame(st_xy)

    # 1. Computing Transport Plan
    def compute_transportplan(self,
                              knn_scale_method='MinMaxScaler',
                              save_dis=True,
                              Spa_cost=True,
                              knn_metric_d12='correlation',
                              n_neighbors_cell=5, n_neighbors_spot=5, n_neighbors_pos=5,
                              cluster='celltype',
                              cluster_method='Leiden', resolu_cluster=0.2,
                              verbose=False,
                              ps_bandwidth=0.1, sp_bandwidth=0.1,
                              save_sw=True,
                              roh_indiff=10, roh_outsam=10,
                              cost2=None,
                              alpha=0.1, lamda=10.0, ent_reg=0.1,
                              initdis_method='minus',
                              save_swusot=True):
        """
        Optimal transport module for computing transport plan.
        :param knn_scale_method: scaling method, the string name can be: 'Max', 'MinMaxScaler', 'L2_Normalization'.
        :param save_dis: whether the computed distance matrices need to save in file_path?
        :param Spa_cost: whether used location and expression information for consturcting spatial distance matrix?
        :param knn_metric_d12: metric to be computed in scipy for D in KNN method.
        :param n_neighbors_cell: number of nearest neighbor cells connected to each cell in scRNA-seq data based on gene expression.
        :param n_neighbors_spot: number of nearest neighbor spots connected to each spot in ST data based on gene expression.
        :param n_neighbors_pos: number of nearest neighbor spots connected to each spot in ST data based on spatial location.
        :param cluster: the column name of cell type information in sc_meta data.
        :param cluster_method: clustering method, the string name can be: 'Leiden' or 'Louvain'.
        :param resolu_cluster: controlling the coarseness of the clustering. Higher values lead to more clusters.
        :param verbose: whether show the neighborhoods and cell type relationship between spots?
        :param ps_bandwidth: bandwidth of spatial coordinates determining the maximum neighbors radius.
        :param sp_bandwidth: bandwidth of gene expression determining the maximum neighbors radius.
        :param save_sw: whether the spatial weight and spatially weighted distance results need to save in file_path?
        :param roh_indiff: control the spatial weight strength for spots inside the neighborhood but of a different cluster.
        :param roh_outsam: control the spatial weight strength for spots outside the neighborhood but of the same cluster.
        :param cost2: cost matrix of gene expression between cells and spots, with rows being cells and columns being spots.
        :param alpha: weight for structure term.
        :param lamda: weight for KL divergence penalizing unbalanced transport.
        :param ent_reg: weight for entropy regularization term.
        :param initdis_method: initialization method, the string name can be: 'minus', 'minus_exp', 'uniform_distribution'.
        :param save_swusot: whether the transport plan result need to save in file_path?
        :return: a transport plan of cell-to-spot mapping between cells in scRNA-seq data and spots in ST data.
        """

        print('1. Optimal transport module for computing transport plan ......')
        D_cell, D_spot, D_pos, D_cell_spot = dis_matrix.compute_costs(sc_exp=self.sc_exp,
                                                                      st_exp=self.st_exp,
                                                                      st_xy=self.st_xy,
                                                                      file_path=self.file_path,
                                                                      use_gpu = self.use_gpu,
                                                                      n_neighbors_cell=n_neighbors_cell,
                                                                      n_neighbors_spot=n_neighbors_spot,
                                                                      n_neighbors_pos=n_neighbors_pos,
                                                                      knn_scale_method=knn_scale_method,
                                                                      save_dis=save_dis,
                                                                      knn_metric_d12=knn_metric_d12)
        if Spa_cost:
            Spa_cost = spa_weight.compute_sw(st_exp=self.st_exp,
                                             st_xy=self.st_xy,
                                             sc_meta=self.sc_meta,
                                             file_path=self.file_path,
                                             use_gpu=self.use_gpu,
                                             d_spot=D_spot, d_pos=D_pos,
                                             cluster=cluster, cluster_method=cluster_method,
                                             resolu_cluster=resolu_cluster,
                                             verbose=verbose,
                                             ps_bandwidth=ps_bandwidth, sp_bandwidth=sp_bandwidth,
                                             roh_indiff=roh_indiff, roh_outsam=roh_outsam,
                                             save_sw=save_sw)
        else:
            Spa_cost=None

        TransportPlan = sw_usot.compute_swusot(sc_exp=self.sc_exp,
                                               st_exp=self.st_exp,
                                               file_path=self.file_path,
                                               cost12=D_cell_spot, cost1=D_cell,
                                               spa_cost=Spa_cost, cost2=cost2,
                                               alpha=alpha, lamda=lamda, ent_reg=ent_reg,
                                               initdis_method=initdis_method,
                                               save_swusot=save_swusot)
        return TransportPlan

    def swot_deconvolution(self,
                           t_mapping,
                           cluster='celltype',
                           minto0=0.05, mincut=0.1,
                           save_ctmapping=True):
        """
        Estimation of cell-type compositions.
        :param t_mapping: a transport plan of cell-to-spot, with rows being cells and columns being spots.
        :param cluster: the column name of cell type information in sc_meta data.
        :param minto0: the threshold of setting 0.
        :param mincut: the minimum threshold.
        :param save_ctmapping: whether the cell-type mapping result need to save in file_path?
        :return: A cell-type proportion matrix.
        """
        print('2. Cell mapping module for inferring cell-type composition ......')
        ct_order = pd.Index(self.sc_meta.celltype.unique(), dtype=object)
        CT_proportions = cell_mapping.ct_mapping(t_mapping,
                                                 st_xy=self.st_xy,
                                                 sc_meta=self.sc_meta,
                                                 ct_order=ct_order,
                                                 file_path=self.file_path,
                                                 cluster=cluster,
                                                 save=save_ctmapping,
                                                 minto0=minto0, mincut=mincut)

        return CT_proportions

    def swot_restruction(self,
                         ct_mapping,
                         t_mapping,
                         tech='10XVisium',
                         save_cellmapping=False,
                         cells_num_min=None,
                         cells_num_max=None):
        """
        Estimation of cell numbers and cell coordinates per spot for inferring single-cell spatial map.
        :param ct_mapping: a transport plan of cell-to-spot, with rows being cells and columns being spots.
        :param t_mapping: a cell-type proportion matrix, with rows being spots and columns being cell types.
        :param tech: the sequencing technology of ST data.
        :param save_cellmapping: whether the cell mapping result need to save in file_path?
        :param cells_num_min: the minimum number of cells per spot.
        :param cells_num_max: the maximum number of cells per spot.
        :return: An inferred single-cell spatial map includes coordinates and expressions.
        """
        print('3. Cell mapping module for single-cell spatial maps inference ......')
        cellnum_spot = cell_mapping.compu_cells_eachspot(st_exp=self.st_exp,
                                                         tech=tech,
                                                         cells_num_max=cells_num_max,
                                                         cells_num_min=cells_num_min)

        Cell_mapping_xy, cs_ctnum = cell_mapping.cell_mapping_xy(st_xy=self.st_xy,
                                                                 ct_mapping_df=ct_mapping,
                                                                 cellnum_spot=cellnum_spot)

        cell_mapping_exp, cell_mapping_meta = cell_mapping.cell_mapping_expression(t_mapping=t_mapping,
                                                                                     cs_xy=Cell_mapping_xy,
                                                                                     cellnum_spot=cellnum_spot,
                                                                                     sc_exp=self.sc_exp,
                                                                                     sc_meta=self.sc_meta,
                                                                                     cs_ctnum=cs_ctnum,
                                                                                     file_path=self.file_path,
                                                                                     save=save_cellmapping)
        Cell_mapping = [cell_mapping_exp, cell_mapping_meta]
        return Cell_mapping



