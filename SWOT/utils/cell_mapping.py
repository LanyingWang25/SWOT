
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")
class NonConvergenceError(Exception):
    pass

def sum_to_1(ct_spot):
    """
    Summing the cell-type proportions in each spot equals 1.
    """
    row_sums = ct_spot.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return ct_spot.div(row_sums, axis=0)


def min_cut(ct_spot, mincut):
    """
    Pruning the cell-type proportion matrix.
    """
    data = ct_spot.to_numpy(copy=False)
    max_vals = data.max(axis=1, keepdims=True)
    mask = data < (max_vals * mincut)
    data[mask] = 0
    return pd.DataFrame(data, columns=ct_spot.columns, index=ct_spot.index)

def ct_mapping(t_mapping, st_xy, sc_meta,
               ct_order, file_path,
               cluster='celltype', save=True,
               minto0=0.05, mincut=0.1):
    """
    Estimation of cell-type compositions.
    """
    if (not isinstance(t_mapping, pd.DataFrame) or
            not isinstance(st_xy, pd.DataFrame) or
            not isinstance(sc_meta, pd.DataFrame)):
        print('Please enter T_mapping, st_xy, and sc_meta data of pandas DataFrame type represents a transport plan.')
        return None

    if not isinstance(ct_order, pd.Index):
        print("Please enter ct_order of Index object, e.g. Index(['a', 'b', 'c'], "
              "dtype='object', name='name1').")
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if not isinstance(cluster, str):
        print('Please enter cluster of string type represents the column name of '
              'cell type information in sc_meta data.')
        return None
    if cluster not in sc_meta.columns:
        print('Please confirm whether cluster name is a column of sc_meta?')
        return None
    if (not isinstance(minto0, float) or
            not isinstance(mincut, float)):
        print('Please enter minto0 and mincut of float type.')
        return None

    if set(t_mapping.index) == set(sc_meta.index):
        t_mapping = t_mapping.apply(lambda x: pd.to_numeric(x), axis=0)
    else:
        print("Please make the cell names of T_mapping and sc_meta correspond.")
        return None

    print('Computing cell type proportions ... ')
    if 'cellname' not in sc_meta.columns:
        sc_meta['cellname'] = sc_meta.index

    celltype_to_cells = sc_meta.groupby(cluster)['cellname'].apply(list).to_dict()
    all_cells = sc_meta['cellname'].values
    t_mapping_np = t_mapping.loc[all_cells].to_numpy()

    celltypes = sc_meta[cluster].unique()
    mask = np.zeros((len(celltypes), len(all_cells)), dtype=bool)
    for idx, ct in enumerate(celltypes):
        ct_cells = sc_meta[sc_meta[cluster] == ct].index
        mask[idx, :] = sc_meta.index.isin(ct_cells)

    ct_means = (mask @ t_mapping_np) / mask.sum(axis=1, keepdims=True)

    OT_spotmap = pd.DataFrame(ct_means.T, columns=celltypes, index=st_xy.index)

    OT_spotmap_mincut = min_cut(OT_spotmap, mincut)
    OT_spotmap_sum1 = sum_to_1(OT_spotmap_mincut)
    OT_spotmap_cut = OT_spotmap_sum1.mask(OT_spotmap_sum1 < minto0, 0)
    OT_spotmap_cut_sum1 = sum_to_1(OT_spotmap_cut)
    ct_proportion = OT_spotmap_cut_sum1[ct_order]

    if save:
        os.makedirs(os.path.join(file_path, 'CellMapping'), exist_ok=True)
        ct_proportion.to_csv(os.path.join(file_path, 'CellMapping', 'Celltype_composition.csv'), sep=',', index=True, header=True)
        print('The cell-type composition results are saved in ' + file_path + 'CellMapping/' + 'Celltype_composition.csv' + '.')

    return ct_proportion


def compu_cells_eachspot(st_exp, tech=None,
                         cells_num_min=None, cells_num_max=None):
    """
    Estimation of cell numbers per spot for inferring single-cell spatial maps.
    """
    cellnum_spot = pd.DataFrame(columns=(['cell_num']), index=st_exp.columns)

    # automatically assigns a varying number of cells to different spots based on the proportion of zero values in gene expressions.
    assert (tech in ['10XVisium', 'SpatialTranscriptomics', 'Slide-seq', 'user-defined']), \
        "tech argument has to be either one of '10XVisium', 'SpatialTranscriptomics', 'Slide-seq', 'user-defined'."
    zero_ratio = (st_exp == 0).mean(axis=0)
    cell_num_log = (-10 * np.log(zero_ratio + 0.001) - 1).round().astype(int)
    cell_num_log[cell_num_log <= 0] = 1
    cell_num = cell_num_log

    if tech == '10XVisium':
        print('Based on 10X Visium technology and zero proportions to estimate cell numbers.')
        cells_num_min = 1
        cells_num_max = 10
    elif tech == 'SpatialTranscriptomics':
        print('Based on Spatial Transcriptomics technology and zero proportions to estimate cell numbers')
        cells_num_min = 10
        cells_num_max = 40
    elif tech == 'Slide-seq':
        print('Based on Slide-seq technology and zero proportions to estimate cell numbers')
        cells_num_min = 1
        cells_num_max = 3
    elif tech == 'user-defined':
        print('Based on the defined by user and zero proportions to estimate cell numbers')
        cells_num_min = cells_num_min
        cells_num_max = cells_num_max

    cells_num_avg = (cells_num_min + cells_num_max) / 2
    indices_exceed_max = cell_num >= cells_num_max
    cell_num[indices_exceed_max] = np.random.randint(cells_num_avg, cells_num_max+1, size=indices_exceed_max.sum())
    indices_exceed_min = cell_num <= cells_num_min
    cell_num[indices_exceed_min] = np.random.randint(cells_num_min, cells_num_avg+1, size=indices_exceed_min.sum())
    cellnum_spot['cell_num'] = cell_num
    cells_sum = sum(cellnum_spot.cell_num)
    print('The number of cells in all spots is: ', str(cells_sum))
    return cellnum_spot



def cell_mapping_xy(ct_mapping_df, st_xy, cellnum_spot):
    """
    Estimation of spatial coordinates for inferring single-cell spatial maps.
    """
    if not isinstance(ct_mapping_df, pd.DataFrame):
        print('Please enter ct_mapping_df data of pandas DataFrame type represents cell-type proportions.')
        return None

    cell_total = sum(cellnum_spot.cell_num)
    cs_columns = np.array(['cs_name', 'cs_type', 'cs_x', 'cs_y', 'spot_name', 'spot_x', 'spot_y'])
    cs_index = np.array(['cell_' + str(i) for i in np.arange(1, cell_total + 1, step=1)])
    cs_xy = pd.DataFrame(index=cs_index, columns=cs_columns)
    cs_xy['cs_name'] = cs_index

    spot_name_list = []
    spot_x_list = []
    spot_y_list = []
    for st in cellnum_spot.index:
        spot_name_list_tmp = np.repeat(st, cellnum_spot.at[st, 'cell_num'])
        spot_x_list_tmp = np.repeat(st_xy.at[st, 'X'], cellnum_spot.at[st, 'cell_num'])
        spot_y_list_tmp = np.repeat(st_xy.at[st, 'Y'], cellnum_spot.at[st, 'cell_num'])
        spot_name_list.append(spot_name_list_tmp)
        spot_x_list.append(spot_x_list_tmp)
        spot_y_list.append(spot_y_list_tmp)
    spot_name = [item for list in spot_name_list for item in list]
    spot_x = [item for list in spot_x_list for item in list]
    spot_y = [item for list in spot_y_list for item in list]
    cs_xy['spot_name'] = spot_name
    cs_xy['spot_x'] = spot_x
    cs_xy['spot_y'] = spot_y

    coord = st_xy[['X', 'Y']].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    for i in range(0, len(distances)):
        if distances[i, 1] == 0:
            dist_tmp = distances[i,]
            dist_non0 = next((num for num in dist_tmp if num != 0), None)
            distances[i, 1] = dist_non0

    radius = distances[:, 1] / 2
    radius = radius.tolist()
    all_coord = cs_xy[['spot_x', 'spot_y']].to_numpy()

    spot_len = cellnum_spot.values.tolist()
    all_radius = list()
    for i in range(len(spot_len)):
        all_radius_tmp = [radius[i]] * spot_len[i][0]
        all_radius.extend(all_radius_tmp)

    # a square root transformation to the radius to mitigate the radial density effect
    u = np.random.uniform(0, 1, len(all_radius))
    length = np.array(all_radius) * np.sqrt(u)
    angle = np.random.uniform(0, 2 * np.pi, len(u))

    cs_xy['cs_x'] = all_coord[:, 0] + length * np.cos(angle)
    cs_xy['cs_y'] = all_coord[:, 1] + length * np.sin(angle)

    cs_ctnum = pd.DataFrame(index=ct_mapping_df.index, columns=ct_mapping_df.columns)
    for st in ct_mapping_df.index:
        st_cenum = cellnum_spot.at[st, 'cell_num']
        ct_prop = ct_mapping_df.loc[st]
        cs_ctnum.loc[st] = round(ct_prop * st_cenum)

        rowsum = cs_ctnum.loc[st].sum()
        if rowsum > st_cenum:
            duo = int(rowsum - st_cenum)
            ctprop_csctnum = pd.DataFrame(ct_prop)
            ctprop_csctnum['csctnum'] = cs_ctnum.loc[st]
            ctprop_csctnum_order = ctprop_csctnum.sort_values(by=st, axis=0, ascending=True)
            ct_non0 = []
            for i in ctprop_csctnum_order.index:
                if (ctprop_csctnum_order.at[i, st] != 0) and (ctprop_csctnum_order.at[i, 'csctnum'] != 0):
                    ct_non0.append(i)
            ct_non0_jian = ct_non0[0:duo]
            cs_ctnum.loc[st, ct_non0_jian] = cs_ctnum.loc[st, ct_non0_jian] - 1

        elif rowsum < st_cenum:
            ctp_min = min([i for i in ct_prop if i != 0])
            ctp_min_index = ct_prop.loc[ct_prop == ctp_min].index[0]
            cs_ctnum.loc[st, ctp_min_index] = cs_ctnum.loc[st, ctp_min_index] + (st_cenum - rowsum)

        cs_st_ctnum = cs_ctnum.loc[st]
        cs_st_name = cs_xy.loc[cs_xy['spot_name'] == st].index
        cs_st_name_ct = []
        for ct in cs_st_ctnum.index:
            cs_st_name_ct_tmp = [ct] * int(cs_st_ctnum[ct])
            cs_st_name_ct.append(cs_st_name_ct_tmp)
        cs_ct = [item for list in cs_st_name_ct for item in list]
        cs_xy.loc[cs_st_name, 'cs_type'] = cs_ct

    return cs_xy, cs_ctnum



def cell_mapping_expression(t_mapping, cs_xy, cellnum_spot, sc_exp,
                            sc_meta, cs_ctnum, file_path, save=False):
    """
    Obtain the single-cell spatial gene expression profiles for inferring single-cell spatial maps,
    with rows being genes and columns being cells.
    :return:
    cs_expression： gene expression profile of estimated cells.
    cs_xy_new: spatial coordinates of estimated single-cell spatial maps
               with rows being cells and columns being:
                'cs_name': new cell name;
                'cs_type': new cell type;
                'cs_x': new cell coordinate of X;
                'cs_y': new cell coordinate of Y;
                'spot_name': name of the spot to which the new cell belongs;
                'spot_x': X coordinate of the spot to which the new cell belongs;
                'spot_y': Y coordinate of the spot to which the new cell belongs;
                'cell_name_old': old cell name in scRNA-seq data.
    """

    if not isinstance(t_mapping, pd.DataFrame):
        print('Please enter T_mapping data of pandas DataFrame type represents a transport plan.')
        return None
    if not isinstance(cs_xy, pd.DataFrame):
        print('Please enter st_xy data of pandas DataFrame type with rows being spots and columns being X and Y.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save argument.")
        return None

    cs_ce_st = pd.concat([cs_xy['cs_name'], cs_xy['cs_type'], cs_xy['spot_name']], axis=1)
    cs_ce_st['cell_name'] = cs_xy['cs_name']

    for st in t_mapping.columns:
        st_cells_ct = pd.concat([t_mapping[st], sc_meta['celltype']], axis=1)
        st_cells_ct_order = st_cells_ct.sort_values(by=st, ascending=False)
        st_ct = cs_xy['cs_type'][cs_xy['spot_name'] == st]
        st_ct_num = cellnum_spot.loc[st]
        st_ct_list = []
        st_ct = pd.DataFrame(st_ct)
        for ct in st_ct['cs_type'].unique().tolist():
            if (sc_meta['celltype'] == ct).sum() < (st_ct['cs_type'] == ct).sum():
                shangeshu = -((st_ct['cs_type'] == ct).sum() - (sc_meta['celltype'] == ct).sum())
                st_ct_drop = st_ct.drop(st_ct[st_ct['cs_type'] == ct].index[shangeshu:])
                st_ct = st_ct_drop
                st_ct_list_tmp11 = st_cells_ct_order.index[st_cells_ct_order['celltype'] == ct]
                st_ct_list_tmp21 = st_ct_list_tmp11[:(st_ct['cs_type'] == ct).sum()].values.tolist()
                st_ct_list.append(st_ct_list_tmp21)
                continue

            st_ct_list_tmp1 = st_cells_ct_order.index[st_cells_ct_order['celltype'] == ct]
            st_ct_list_tmp2 = st_ct_list_tmp1[:(st_ct['cs_type'] == ct).sum()].values.tolist()
            st_ct_list.append(st_ct_list_tmp2)

        st_ct_list1 = [item for list in st_ct_list for item in list]
        if len(st_ct_list1) != len(cs_ce_st['cell_name'][cs_ce_st['spot_name'] == st]):
            cs_ce_st_tmp1 = cs_ce_st.loc[st_ct_drop.index, 'cell_name']
            cs_ce_st['cell_name'][cs_ce_st.loc[cs_ce_st_tmp1, 'cs_name']] = st_ct_list1
        else:
            cs_ce_st.loc[cs_ce_st['spot_name'] == st, 'cell_name'] = st_ct_list1

    cs_cells_fei = cs_ce_st['cell_name'][cs_ce_st['cell_name'] == cs_ce_st['cs_name']]
    cs_ce_st_fin = cs_ce_st.drop(cs_cells_fei)
    cs_cells = cs_ce_st_fin['cell_name'].tolist()

    cs_expression = sc_exp[cs_cells]
    cs_expression.columns = cs_ce_st_fin['cs_name']

    cs_xy_new = cs_xy.drop(cs_cells_fei)
    cs_ctnum_new = cs_ctnum.copy()
    for st in cs_ctnum.index:
        for ct in cs_ctnum.columns:
            ctnumt_SC = sc_meta['celltype'].value_counts()[ct]
            if cs_ctnum.at[st, ct] > ctnumt_SC:
                cs_ctnum_new.at[st, ct] = ctnumt_SC
            else:
                cs_ctnum_new.at[st, ct] = cs_ctnum.at[st, ct]

    cs_xy_new['cell_name_old'] = cs_cells

    file_name = 'Cell_maps_exp.csv'
    file_name_cs_xy = 'Cell_maps_xy.csv'
    if save:
        os.makedirs(os.path.join(file_path, 'CellMapping'), exist_ok=True)
        cs_expression.to_csv(file_path + 'CellMapping/' + file_name, sep=',', index=True, header=True)
        cs_xy_new.to_csv(file_path + 'CellMapping/' + file_name_cs_xy, sep=',', index=True, header=True)
        print('The gene expression and single cell information after cell mapping is saved in ' +
              file_path + 'CellMapping/Cell_maps_xy.csv, Cell_maps_exp.csv.')

    return cs_expression, cs_xy_new
