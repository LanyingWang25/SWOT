
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd().replace('\\', '/'))))


import pandas as pd
from SWOT import swot


if __name__ == '__main__':

    #### 1. reading dataset
    SC_exp = pd.read_csv('Data_PDAC/PDAC_sc_exp.csv', sep=',', index_col=0)
    SC_meta = pd.read_csv('Data_PDAC/PDAC_sc_meta.csv', sep=',', index_col=0)
    ST_exp = pd.read_csv('Data_PDAC/PDAC_st_exp.csv', sep=',', index_col=0)
    ST_xy = pd.read_csv('Data_PDAC/PDAC_st_xy.csv', sep=',', index_col=0)

    result_path = 'Data_PDAC/'

    if 'SWOT_files' not in os.listdir(result_path):  # Save all results in 'SWOT_files' folder
        os.mkdir(os.path.join(result_path, 'SWOT_files'))
    swot_path = result_path + 'SWOT_files/'

    swotclass = swot.SWOTscsm(sc_exp=SC_exp, sc_meta=SC_meta,
                              st_exp=ST_exp, st_xy=ST_xy,
                              file_path=swot_path, use_gpu=False)

    #### 2. An optimal transport module for computing transport plan
    TransportPlan = swotclass.compute_transportplan(knn_scale_method='MinMaxScaler',
                                                    save_dis=True,
                                                    knn_metric_d12='correlation',
                                                    cluster='celltype', cluster_method='Leiden',
                                                    resolu_cluster=2.5,
                                                    verbose=False, ps_bandwidth=0.1, sp_bandwidth=0.1,
                                                    save_sw=True, cost2=None,
                                                    alpha=0.1, lamda=100.0, ent_reg=0.05,
                                                    initdis_method='minus', save_swusot=True)

    #### 3. A cell mapping module for estimating cell-type compositions, cell numbers and cell coordinates per spot
    ###### 3.1 Cell-type composition
    CT_mapping = swotclass.swot_deconvolution(t_mapping=TransportPlan,
                                              cluster='celltype',
                                              minto0=0.001, mincut=0.0,
                                              save_ctmapping=True)

    ###### 3.2 Single-cell spatial maps inference
    Cell_mapping = swotclass.swot_restruction(ct_mapping=CT_mapping,
                                              t_mapping=TransportPlan,
                                              tech='SpatialTranscriptomics',
                                              save_cellmapping=True)
