
import os
import scipy
import torch
import scipy.stats
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, normalize

import warnings
warnings.filterwarnings("ignore")

class NonConvergenceError(Exception):
    pass


def compute_costs(sc_exp, st_exp, st_xy,
                  file_path,
                  n_neighbors_cell,
                  n_neighbors_spot,
                  n_neighbors_pos,
                  knn_scale=True,
                  knn_scale_method='MinMaxScaler',
                  knn_metric_d12='correlation',
                  save_dis=False,
                  use_gpu=False):
    """
    Calculate four costs/distances matrices.
    :param n_neighbors_cell: number of neighbors to use for k-neighbors queries of gene expression in scRNA-seq data.
    :param n_neighbors_spot: number of neighbors to use for k-neighbors queries of gene expression in ST data.
    :param n_neighbors_pos: number of neighbors to use for k-neighbors queries of spatial position in ST data.
    :param knn_scale: whether the cost matrices by KNN need to scaling?
    :param knn_scale_method: scaling method, the string name can be: 'Max', 'MinMaxScaler', 'L2_Normalization'.
    :param knn_metric_d12: metric to be computed in scipy for D in KNN method.
    :param save_dis: whether the computed distance matrices need to save in file_path?
    :returns:
        four distances matrices：
        # Dse: distance metric of gene expression among cells in scRNA-seq data.
        # Dte: distance metric of gene expression among spots in ST data.
        # Dtc: distance metric of coordinates among spots in ST data.
        # D:   distance metric of gene expression between cells and spots in scRNA-seq and ST data.
    """

    print('Computing distance matrices ......')

    d_cell, d_spot, d_pos = compute_cost_knn(sc_exp=sc_exp,
                                             st_exp=st_exp,
                                             st_xy=st_xy,
                                             file_path=file_path,
                                             save=save_dis,
                                             scaling=knn_scale,
                                             scale_method=knn_scale_method,
                                             n_neighbors_cell=n_neighbors_cell,
                                             n_neighbors_spot=n_neighbors_spot,
                                             n_neighbors_pos=n_neighbors_pos,
                                             use_gpu = use_gpu)

    d_cellspot = compute_cost_c12(sc_exp=sc_exp,
                                  st_exp=st_exp,
                                  file_path=file_path,
                                  save=save_dis,
                                  metric=knn_metric_d12,
                                  scaling=knn_scale,
                                  scale_method=knn_scale_method,
                                  use_gpu=use_gpu)

    return d_cell, d_spot, d_pos, d_cellspot





def data_scaling(data, scale_method, use_gpu=False):
    if isinstance(data, pd.DataFrame):
        if use_gpu:
            data = torch.tensor(data.values, dtype=torch.float32).to('cuda')
        if not use_gpu:
            data = data.to_numpy()

    assert (scale_method in ['Max', 'MinMaxScaler', 'L2_Normalization']), \
        "scale_method argument has to be either one of 'Max', 'MinMaxScaler', 'L2_Normalization'."

    if use_gpu:
        if scale_method == 'Max':
            data_max = torch.max(data)
            data_scaled = data / data_max
        elif scale_method == 'MinMaxScaler':
            data_range = data.max(dim=0).values - data.min(dim=0).values
            data_range[data_range == 0] = 1e-8
            min_range, max_range = (0, 1)
            scale = (max_range - min_range) / data_range
            offset = min_range - data.min(dim=0).values * scale
            data_scaled = data * scale + offset
        elif scale_method == 'L2_Normalization':
            norm = torch.norm(data, p=2, dim=1, keepdim=True)
            data_scaled = data / norm
    else:
        if scale_method == 'Max':
            data_scaled = data / np.max(data)
        elif scale_method == 'MinMaxScaler':
            sklearn_scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = sklearn_scaler.fit_transform(data)
        elif scale_method == 'L2_Normalization':
            data_scaled = normalize(data, norm="l2", axis=1)

    return data_scaled



def construct_KNNgraph(data, n_neighbors, mode="connectivity", use_gpu=False):
    if isinstance(data, pd.DataFrame):
        if use_gpu:
            data = torch.tensor(data.values, dtype=torch.float32).to('cuda')
        if not use_gpu:
            data = data.to_numpy()
    assert (mode in ["connectivity", "distance"]), \
        "mode argument has to be either one of 'connectivity', or 'distance'. "

    if use_gpu:
        data_gpu = data.clone().detach().to('cuda')
        if mode == "connectivity":
            include_self = True
        else:
            include_self = False
        data_graph = kneighbors_graph(X=data_gpu.cpu().numpy(), n_neighbors=n_neighbors, mode=mode, metric="minkowski", include_self=include_self)
    else:
        if mode == "connectivity":
            include_self = True
        else:
            include_self = False
        data_graph = kneighbors_graph(X=data, n_neighbors=n_neighbors, mode=mode, metric="minkowski", include_self=include_self)
    return data_graph


def compute_cost_knn(sc_exp, st_exp, st_xy, file_path,
                     scaling=True,
                     scale_method='MinMaxScaler',
                     save=True,
                     n_neighbors_cell=5, n_neighbors_spot=5, n_neighbors_pos=5,
                     use_gpu = False):
    """
    Compute three distances between samples in kNN graph using the shortest path distance.
    # the gene expression distance among cells in scRNA-seq data,
      the gene expression distance among spots in ST data,
      the spatial coordinates distance among spots in ST data.
    """
    if scaling != True and scaling != False:
        print("Please select 'True' or 'False' for scaling.")
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if not isinstance(n_neighbors_cell, int):
        print('Please enter n_neighbors_cell of int type.')
        return None
    if not isinstance(n_neighbors_spot, int):
        print('Please enter n_neighbors_spot of int type.')
        return None
    if not isinstance(n_neighbors_pos, int):
        print('Please enter n_neighbors_pos of int type.')
        return None

    if use_gpu:
        sc_exp_gpu = torch.tensor(sc_exp.T.values, dtype=torch.float32).to('cuda')
        st_exp_gpu = torch.tensor(st_exp.T.values, dtype=torch.float32).to('cuda')
        st_xy_gpu = torch.tensor(st_xy.values, dtype=torch.float32).to('cuda')
        sc_exp_graph = construct_KNNgraph(sc_exp_gpu, n_neighbors=n_neighbors_cell, use_gpu=True)
        st_exp_graph = construct_KNNgraph(st_exp_gpu, n_neighbors=n_neighbors_spot, use_gpu=True)
        st_xy_graph = construct_KNNgraph(st_xy_gpu, n_neighbors=n_neighbors_pos, use_gpu=True)
    else:
        sc_exp_graph = construct_KNNgraph(sc_exp.T, n_neighbors=n_neighbors_cell)
        st_exp_graph = construct_KNNgraph(st_exp.T, n_neighbors=n_neighbors_spot)
        st_xy_graph = construct_KNNgraph(st_xy, n_neighbors=n_neighbors_pos)

    scexp_shortestPath = dijkstra(csgraph=csr_matrix(sc_exp_graph), directed=False, return_predecessors=False)
    stexp_shortestPath = dijkstra(csgraph=csr_matrix(st_exp_graph), directed=False, return_predecessors=False)
    stxy_shortestPath = dijkstra(csgraph=csr_matrix(st_xy_graph), directed=False, return_predecessors=False)

    scexp_shortestPath_max = np.nanmax(scexp_shortestPath[scexp_shortestPath != np.inf])
    stexp_shortestPath_max = np.nanmax(stexp_shortestPath[stexp_shortestPath != np.inf])
    stxy_shortestPath_max = np.nanmax(stxy_shortestPath[stxy_shortestPath != np.inf])

    scexp_shortestPath[scexp_shortestPath > scexp_shortestPath_max] = scexp_shortestPath_max
    stexp_shortestPath[stexp_shortestPath > stexp_shortestPath_max] = stexp_shortestPath_max
    stxy_shortestPath[stxy_shortestPath > stxy_shortestPath_max] = stxy_shortestPath_max

    if scaling:
        assert (scale_method in ['Max', 'MinMaxScaler', 'L2_Normalization']), \
            "scale_method argument has to be either one of 'Max', 'MinMaxScaler', 'L2_Normalization'."
        if use_gpu:
            scexp_shortestPath = torch.tensor(scexp_shortestPath, dtype=torch.float32).to('cuda')
            stexp_shortestPath = torch.tensor(stexp_shortestPath, dtype=torch.float32).to('cuda')
            stxy_shortestPath = torch.tensor(stxy_shortestPath, dtype=torch.float32).to('cuda')
            d_cell = data_scaling(scexp_shortestPath, scale_method, use_gpu=True)
            d_spot = data_scaling(stexp_shortestPath, scale_method, use_gpu=True)
            d_pos = data_scaling(stxy_shortestPath, scale_method, use_gpu=True)
            d_cell = d_cell.cpu()
            d_spot = d_spot.cpu()
            d_pos = d_pos.cpu()
        else:
            d_cell = data_scaling(scexp_shortestPath, scale_method, use_gpu=False)
            d_spot = data_scaling(stexp_shortestPath, scale_method, use_gpu=False)
            d_pos = data_scaling(stxy_shortestPath, scale_method, use_gpu=False)

    cell_name = sc_exp.columns
    spot_name = st_xy.index

    d_cell = pd.DataFrame(d_cell, index=cell_name, columns=cell_name).round(3)
    d_spot = pd.DataFrame(d_spot, index=spot_name, columns=spot_name).round(3)
    d_pos = pd.DataFrame(d_pos, index=spot_name, columns=spot_name).round(3)

    if save:
        os.makedirs(os.path.join(file_path, 'OptimalTransport'), exist_ok=True)
        d_cell.to_csv(file_path + 'OptimalTransport/D_cell.csv', sep=',', index=True, header=True)
        d_spot.to_csv(file_path + 'OptimalTransport/D_spot.csv', sep=',', index=True, header=True)
        d_pos.to_csv(file_path + 'OptimalTransport/D_pos.csv', sep=',', index=True, header=True)
        print('Three distance matrices (Dse, Dte, Dtc) are saved in ' + file_path +
              'OptimalTransport/D_cell.csv, D_spot.csv, D_pos.csv.')
    return d_cell, d_spot, d_pos


def compute_cost_c12(sc_exp, st_exp,
                     file_path, save=True,
                     metric='correlation',
                     scaling=True,
                     scale_method='MinMaxScaler',
                     use_gpu = False):

    if not isinstance(metric, str):
        print('Please enter metric of string type.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if scaling != True and scaling != False:
        print("Please select 'True' or 'False' for scaling.")
        return None

    if use_gpu:
        sc_exp_gpu = torch.tensor(sc_exp.T.values, dtype=torch.float32).to('cuda')
        st_exp_gpu = torch.tensor(st_exp.T.values, dtype=torch.float32).to('cuda')
        if metric == 'cosine':
            distcosine = 1 - torch.mm(sc_exp_gpu / torch.norm(sc_exp_gpu, p=2, dim=1, keepdim=True),
                                      (st_exp_gpu / torch.norm(st_exp_gpu, p=2, dim=1, keepdim=True)).T)
            dist_12_tmp = distcosine.clamp_(min=0)
        if metric == 'correlation':
            XA_centered = sc_exp_gpu - sc_exp_gpu.mean(dim=1, keepdim=True)
            XB_centered = st_exp_gpu - st_exp_gpu.mean(dim=1, keepdim=True)
            numerator = torch.matmul(XA_centered, XB_centered.T)
            denominator = torch.outer(torch.norm(XA_centered, dim=1), torch.norm(XB_centered, dim=1))
            dist_12_tmp = 1 - (numerator / denominator)
        if metric == 'euclidean':
            dist_12_tmp = torch.cdist(sc_exp_gpu, st_exp_gpu, p=2)

        if scaling:
            dist_12_scaled = data_scaling(dist_12_tmp, scale_method, use_gpu=True)
            dist_12 = dist_12_scaled.cpu().numpy()
        else:
            dist_12 = dist_12_tmp.cpu().numpy()
    else:
        dist_12_tmp = scipy.spatial.distance.cdist(sc_exp.T, st_exp.T, metric=metric)
        if scaling:
            dist_12 = data_scaling(dist_12_tmp, scale_method)
        else:
            dist_12 = dist_12_tmp

    d_cell_spot = pd.DataFrame(dist_12, index=sc_exp.columns, columns=st_exp.columns).round(3)

    if save:
        os.makedirs(os.path.join(file_path, 'OptimalTransport'), exist_ok=True)
        d_cell_spot.to_csv(file_path + 'OptimalTransport/D_cell_spot.csv', sep=',', index=True, header=True)
        print('The distance matrices (D) is saved in ' + file_path + 'OptimalTransport/D_cell_spot.csv.')
    return d_cell_spot
