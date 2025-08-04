
import os
import torch
import anndata
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

def compute_sw(st_exp, st_xy,
               sc_meta,
               file_path,
               d_pos,
               d_spot,
               cluster='celltype',
               cluster_method='Leiden',
               resolu_cluster=1.0,
               verbose=False,
               ps_bandwidth=0.1,
               sp_bandwidth=0.1,
               roh_indiff=10,
               roh_outsam=10,
               save_sw=False,
               use_gpu=False):
    """
    Compute the spatial weights and spatially weighted distance among spots.
    The spatially weighted strategy incorporates gene expression, derived from pre-clustering of spots,
    with spatial location, derived from spatial neighborhood of coordinates
    :param d_pos: distance metric of coordinates among spots in ST data.
    :param d_spot: distance metric of gene expression among spots in ST data.
    :param cluster: the column name of cell type information in sc_meta data.
    :param cluster_method:  clustering method, the string name can be: 'Leiden' or 'Louvain'.
    :param resolu_cluster: controlling the coarseness of the clustering. Higher values lead to more clusters.
    :param verbose: whether show the neighborhoods and cell type relationship between spots?
    :param ps_bandwidth: bandwidth of spatial coordinates determining the maximum neighbors radius.
    :param sp_bandwidth: bandwidth of gene expression determining the maximum neighbors radius.
    :param roh_indiff: control the spatial weight strength for spots inside the neighborhood but of a different cluster.
    :param roh_outsam: control the spatial weight strength for spots outside the neighborhood but of the same cluster.
    :param save_sw: whether the spatial weight and spatially weighted distance results need to save in file_path?
    :returns:
        Spa_cost: spatially weighted distance matrix among spots.
    """

    print('Computing spatial weight ......')

    st_exp_adata = create_anndata_st(st_exp=st_exp, st_xy=st_xy)

    st_clustering = pre_clustering(st_exp_adata=st_exp_adata,
                                   sc_meta=sc_meta,
                                   file_path=file_path,
                                   cluster=cluster,
                                   cluster_method=cluster_method,
                                   resolu_cluster=resolu_cluster,
                                   save=False)

    spa_cost = spatial_weight_cost(d_pos=d_pos, d_spot=d_spot,
                                   st_clustering=st_clustering,
                                   file_path=file_path,
                                   save=save_sw,
                                   verbose=verbose,
                                   ps_bandwidth=ps_bandwidth,
                                   sp_bandwidth=sp_bandwidth,
                                   roh_indiff=roh_indiff,
                                   roh_outsam=roh_outsam,
                                   use_gpu=use_gpu)

    return spa_cost





def create_anndata_st(st_exp, st_xy):
    """
    Create AnnData object for ST data.
    """
    if 'X' not in st_xy.columns or 'Y' not in st_xy.columns:
        print("Please check the column names of st_xy data, using 'X' and 'Y' to represent the position of spots.")
        return None

    st_exp_counts = csr_matrix(st_exp.T)
    st_exp_adata = anndata.AnnData(st_exp_counts)
    st_exp_adata.obs_names = st_exp.columns
    st_exp_adata.var_names = st_exp.index
    st_exp_adata.obs['X'] = pd.Categorical(st_xy['X'])
    st_exp_adata.obs['Y'] = pd.Categorical(st_xy['Y'])

    return st_exp_adata




def pre_clustering(st_exp_adata,
                   sc_meta,
                   file_path, save=False,
                   cluster='celltype', cluster_method='Leiden',
                   resolu_cluster=1.5):
    """
    Pre_clustering for ST gene expression profiles.
    """
    if not isinstance(st_exp_adata, anndata._core.anndata.AnnData):
        print("Please enter st_exp_adata data of AnnData type or "
              "using 'create_anndata_st' function to create Anndata.")
        return None
    if not isinstance(cluster, str):
        print('Please enter cluster of string type, which represent the column name '
              'of cell type information in sc_meta data.')
        return None
    assert (cluster_method in ["Leiden", "Louvain"]), \
        "cluster_method argument has to be either one of 'Leiden' or 'Louvain'. "
    if not (isinstance(resolu_cluster, float) or isinstance(resolu_cluster, int)):
        print('Please enter resolu_cluster of float or int type.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save argument.")
        return None

    sc.pp.neighbors(st_exp_adata, n_neighbors=20, use_rep='X')
    ct_num = len(sc_meta[cluster].unique())

    # Leiden clustering in Scanpy package
    if cluster_method == 'Leiden':
        sc.tl.leiden(st_exp_adata, resolution=resolu_cluster)
        st_clustering = st_exp_adata.obs['leiden']
        cluster_num = len(st_clustering.unique())

        if cluster_num < ct_num:
            print('Leiden clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest increasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        elif cluster_num > ct_num:
            print('Leiden clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest decreasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        else:
            print('The number of Leiden clusters is: ' + str(cluster_num) +
                  ', it is equals to the number of cell types in scRNA-seq data!')

    # Louvain clustering in Scanpy package
    if cluster_method == 'Louvain':
        sc.tl.louvain(st_exp_adata, resolution=resolu_cluster)
        st_clustering = st_exp_adata.obs['louvain']
        cluster_num = len(st_clustering.unique())

        if cluster_num < ct_num:
            print('Louvain clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest increasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        elif cluster_num > ct_num:
            print('Louvain clusters obtained ' + str(cluster_num) +
                  ' clusters. The scRNA-seq data has ' + str(ct_num) + ' cell types, ' +
                  'and we suggest decreasing the resolu_cluster so that the number of clusters '
                  'in ST data is the same as the number of cell types in scRNA-seq data!')
        else:
            print('The number of Louvain clusters is: ' + str(cluster_num) +
                  ', it is equals to the number of cell types in scRNA-seq data!')

    st_clustering = pd.DataFrame(st_clustering)

    if save:
        os.makedirs(os.path.join(file_path, 'OptimalTransport'), exist_ok=True)
        st_clustering.to_csv(file_path + 'OptimalTransport/st_clustering.csv', sep=',', index=True, header=True)
        print('The pre-clustering result of ST data is saved in ' + file_path + 'OptimalTransport/st_clustering.csv.')

    return st_clustering


def spatial_weight_cost(d_pos, d_spot,
                        st_clustering,
                        file_path, save=False,
                        verbose=True,
                        ps_bandwidth=0.1, sp_bandwidth=0.1,
                        roh_indiff=10, roh_outsam=10,
                        use_gpu=False):
    """
    Compute spatial weights and spatial distance of each spot with respect to others.
    """
    if (not isinstance(d_pos, pd.DataFrame) or
            not isinstance(d_spot, pd.DataFrame) or
            not isinstance(st_clustering, pd.DataFrame)):
        print("Inputs must be pandas DataFrames")
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if verbose != True and verbose != False:
        print("Please select 'True' or 'False' for verbose.")
        return None
    if (not isinstance(ps_bandwidth, float) or
            not isinstance(sp_bandwidth, float)):
        print('Please enter ps_bandwidth or sp_bandwidth of float type.')
        return None

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    ps_dist = torch.tensor(d_pos.values, device=device, dtype=torch.float32)
    sp_dist = torch.tensor(d_spot.values, device=device, dtype=torch.float32)

    from sklearn.preprocessing import LabelEncoder
    cluster_labels = st_clustering.iloc[:, 0].values
    encoder = LabelEncoder()
    encoded_clusters = encoder.fit_transform(cluster_labels)
    clustering = torch.tensor(encoded_clusters, device=device)

    spot_num = ps_dist.shape[0]
    spa_weight = torch.eye(spot_num, device=device)
    spa_cost = torch.zeros((spot_num, spot_num), device=device)

    i, j = torch.triu_indices(spot_num, spot_num, offset=1, device=device)

    ps_vals = ps_dist[i, j]
    sp_vals = sp_dist[i, j]
    cluster_eq = (clustering[i] == clustering[j])

    mask1 = (ps_vals <= ps_bandwidth) & cluster_eq
    mask2 = (ps_vals <= ps_bandwidth) & ~cluster_eq
    mask3 = (ps_vals > ps_bandwidth) & cluster_eq
    mask4 = (ps_vals > ps_bandwidth) & ~cluster_eq

    # Case 1: spots inside the neighborhood and of the same cluster
    inter_neigh = (1 - (ps_vals[mask1] / ps_bandwidth) ** 2) ** 2
    spa_weight[i[mask1], j[mask1]] = inter_neigh
    spa_cost[i[mask1], j[mask1]] = ps_vals[mask1] * (1 - inter_neigh)

    # Case 2: spots inside the neighborhood but of a different cluster
    inter_neigh = (1 - (ps_vals[mask2] / ps_bandwidth) ** 2) ** 2
    exponent = roh_indiff * sp_vals[mask2]
    spa_weight[i[mask2], j[mask2]] = inter_neigh ** exponent
    spa_cost[i[mask2], j[mask2]] = 1 / (1 + sp_vals[mask2] ** (-inter_neigh ** sp_vals[mask2]))

    # Case 3: spots outside the neighborhood but of the same cluster
    intra_neigh = (1 - (sp_vals[mask3] / (sp_vals[mask3] + sp_bandwidth)) ** 2) ** 2
    spa_weight[i[mask3], j[mask3]] = intra_neigh
    spa_cost[i[mask3], j[mask3]] = ps_vals[mask3] ** (roh_outsam * intra_neigh)

    # Case 4: spots outside the neighborhood and of a different cluster
    spa_weight[i[mask4], j[mask4]] = 0
    rand_vals = torch.rand(mask4.sum(), device=device) * 0.2 + 0.5
    spa_cost[i[mask4], j[mask4]] = rand_vals

    spa_cost = spa_cost + spa_cost.t()
    counts = [mask1.sum().item(), mask2.sum().item(),
              mask3.sum().item(), mask4.sum().item()]

    spot_name = d_pos.index
    spa_cost = pd.DataFrame(spa_cost.cpu().numpy(), index=spot_name, columns=spot_name).round(3)

    if verbose:
        print('The number of inside the neighborhood and of the same cluster: ' + str(counts[0]))
        print('The number of inside the neighborhood but of a different cluster: ' + str(counts[1]))
        print('The number of outside the neighborhood but of the same cluster: ' + str(counts[2]))
        print('The number of outside the neighborhood and of a different cluster: ' + str(counts[3]))

    if save:
        os.makedirs(os.path.join(file_path, 'OptimalTransport'), exist_ok=True)
        spa_cost.to_csv(file_path + 'OptimalTransport/spa_cost.csv', sep=',', index=True, header=True)
        print('The spatial weight is saved in ' + file_path + 'OptimalTransport/spa_cost.csv.')

    return spa_cost

