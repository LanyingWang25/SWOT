
import os
import ot
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
class NonConvergenceError(Exception):
    pass


def compute_swusot(sc_exp,st_exp,
                   file_path,
                   cost12,
                   cost1,
                   cost2,
                   spa_cost,
                   alpha=0.1,
                   lamda=10.0,
                   ent_reg=0.1,
                   initdis_method='minus',
                   numItermax=5,
                   save_swusot=False):
    """
    A spatially weighted unbalanced and structured optimal transport module for computing transport plan.
    :param cost12: cost matrix of gene expression between cells and spots, with rows being cells and columns being spots.
    :param cost1: cost matrix of cells in scRNA-seq data.
    :param cost2: cost matrix of coordinates in ST data.
    :param spa_cost: spatially weighted distance matrix among spots.
    :param alpha: weight for structure term.
    :param lamda: weight for KL divergence penalizing unbalanced transport.
    :param ent_reg: weight for entropy regularization term.
    :param initdis_method: initialization method, the string name can be: 'minus', 'minus_exp', 'uniform_distribution'.
    :param numItermax: max number of iterations.
    :param save_swusot: whether the transport plan result need to save in file_path?
    :return: a transport plan of cell-to-spot mapping between cells in scRNA-seq data and spots in ST data.
    """

    print('Computing transport plan ......')

    TransportPlan = transport_plan(cost12=cost12,
                                   spa_cost=spa_cost,
                                   cost1=cost1,
                                   cost2=cost2,
                                   alpha=alpha,
                                   lamda=lamda,
                                   ent_reg=ent_reg,
                                   numItermax=numItermax,
                                   initdis_method=initdis_method,
                                   file_path=file_path,
                                   save=save_swusot)
    cell_name = sc_exp.columns
    spot_name = st_exp.columns

    TransportPlan = pd.DataFrame(TransportPlan, index=cell_name, columns=spot_name)

    return TransportPlan





def init_distribution(cost12, method='uniform_distribution'):
    """
    Initialization the two distributions in source and target domains.
    """
    if method == 'minus':
        weight12 = 1 - cost12
        sc_a = np.sum(weight12, axis=1)
        st_b = np.sum(weight12, axis=0)
        sc_a = sc_a / np.sum(sc_a)
        st_b = st_b / np.sum(st_b)
    if method == 'minus_exp':
        weight12 = np.exp(1 - cost12)
        sc_a = np.sum(weight12, axis=1)
        st_b = np.sum(weight12, axis=0)
        sc_a = sc_a / np.sum(sc_a)
        st_b = st_b / np.sum(st_b)
    # Without any prior information, setting the probabilities to what we observe empirically:
    # uniform over all observed samples
    if method == 'uniform_distribution':
        num_cells = cost12.shape[0]
        num_posis = cost12.shape[1]
        sc_a = ot.unif(num_cells)
        st_b = ot.unif(num_posis)
    return sc_a, st_b



def unbal_ot(sc_a, st_b,
             cost12,
             ent_reg, lamda=np.Inf,
             numItermax=10, stopThr=1e-6):
    """
    Unbalanced optimal transport.
    reference:
        [1] Chizat, L., Peyr'e, G., Schmitzer, B., & Vialard, F. (2016).
            Scaling Algorithms for Unbalanced Transport Problems. arXiv: Optimization and Control.
    """

    sc_a = np.asarray(sc_a, dtype=np.float64).reshape(-1, 1)
    st_b = np.asarray(st_b, dtype=np.float64).reshape(-1, 1)

    N = [sc_a.shape[0], st_b.shape[0]]
    H1 = np.ones([N[0], 1])
    H2 = np.ones([N[1], 1])
    tau = -0.5

    err = {}
    W_primal = {}
    W_dual = {}
    u = np.zeros([N[0], 1], float)
    v = np.zeros([N[1], 1], float)

    def dotp(x, y):
        dotp_v = np.sum(x * y)
        return dotp_v

    def M(u, v, H1, H2, cost12, ent_reg):
        mtmp = (-cost12 + np.matmul(u.reshape(-1, 1), H2.reshape(1, -1)) +
                np.matmul(H1.reshape(-1, 1), v.reshape(1, -1)))
        m_v = mtmp / ent_reg
        return m_v

    def ave(tau, u, u1):
        ave_v = tau * u + (1 - tau) * u1
        return ave_v

    def lse(A):
        lse_v = np.log(np.sum(np.exp(A), axis=1)).reshape(-1, 1)
        return lse_v

    def H(p):
        h_v = -np.sum(p * np.log(p + 1e-20) - 1)
        return h_v

    def KL(h, p):
        kl_v = np.sum(h * np.log(h / p) - h + p)
        return kl_v

    def KLD(u, p):
        kld_v = np.sum(p * (np.exp(-u) - 1))
        return kld_v

    if np.isinf(lamda):
        fi = 1
    else:
        fi = lamda / (lamda + ent_reg)

    for it in range(numItermax):
        # print(f"Progress: {it + 1}/{numItermax}")
        u1 = u
        u = ave(tau, u,
                fi * ent_reg * np.log(sc_a) -
                fi * ent_reg * lse(M(u, v, H1, H2, cost12, ent_reg)) +
                fi * u)

        v = ave(tau, v,
                fi * ent_reg * np.log(st_b) -
                fi * ent_reg * lse(M(u, v, H1, H2, cost12, ent_reg).T) +
                fi * v)
        # coupling
        T_coupling = np.exp(M(u, v, H1, H2, cost12, ent_reg ))

        if lamda == np.inf:  # marginal violation
            W_primal[it] = (dotp(cost12, T_coupling) -
                            ent_reg * H(T_coupling))
            W_dual[it] = (dotp(u, sc_a) + dotp(v, st_b) -
                          ent_reg * np.sum(T_coupling))
            err[it] = np.linalg.norm(np.sum(T_coupling, axis=1) - sc_a)
            err.get(it)
        else:  # difference with previous iterate
            W_primal[it] = (dotp(cost12, T_coupling) +
                            lamda * KL(np.sum(T_coupling, axis=1), sc_a) +
                            lamda * KL(np.sum(T_coupling, axis=0), st_b) -
                            ent_reg * H(T_coupling))
            W_dual[it] = (- lamda * KLD(u / lamda, sc_a) -
                          lamda * KLD(v / lamda, st_b) -
                          ent_reg * np.sum(T_coupling))
            err[it] = np.linalg.norm(u - u1, ord=1)

        if err.get(it) < stopThr and it > numItermax:
            break
    return T_coupling



def gw_loss(c1, c2, sc_a, st_b, loss_gw):
    """
    Initialize loss function for Gromov-Wasserstein discrepancy between two cost matrices.
    The matrices are computed as described in Proposition 1 in [2].
    reference:
        [2] Peyré G, Cuturi M, Solomon J. Gromov-Wasserstein averaging of kernel and distance matrices.
            International Conference on Machine Learning. 2016.
    """
    assert (loss_gw in ['square_loss', 'KL_loss']), ("loss_gw argument has to be either one of "
                                                     "'square_loss', 'KL_loss'.")

    if loss_gw == 'square_loss':
        f1a = c1 ** 2
        f2b = c2 ** 2
        h1a = c1
        h2b = 2 * c2
    if loss_gw == 'KL_loss':
        f1a = c1 * np.log(c1) - c1
        f2b = c2
        h1a = c1
        h2b = np.log(c2)

    constC1 = np.dot(np.dot(f1a, sc_a.reshape(-1, 1)),
                     np.ones(len(st_b)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(sc_a)).reshape(-1, 1),
                     np.dot(st_b.reshape(1, -1), f2b.T))
    constC = constC1 + constC2

    return constC, h1a, h2b



def solve_line_search(a, b):
    if a > 0:  # convex
        minimum = min(1, max(0, -b / (2 * a)))
        return minimum
    elif a + b < 0:
        return 1
    else:
        return 0


def line_search(G, deltaG, c1, c2,
                alpha=None, constC=None, c12=None):
    """
    Line search for solving the quadratic optimization problem in Fused Gromov-Wasserstein distance.
    Algorithm 2 in [3].
    reference:
        [3]. Vayer T, Chapel L, Flamary R, et al. Optimal transport for structured data with application on graphs.
             Proceedings of the 36th International Conference on Machine Learning, PMLR. 2019 97:6275-6284.
    """
    dot1 = np.dot(c1, deltaG)
    dot12 = dot1.dot(c2)  # C1 deltaG C2
    a = -2 * alpha * np.sum(dot12 * deltaG)  # -2 * alpha * <C1 deltaG C2, deltaG>
    b = (np.sum(((1 - alpha) * c12 + alpha * constC) * deltaG) -
         2 * alpha * (np.sum(dot12 * G) +
                      np.sum(np.dot(c1, G).dot(c2) * deltaG)))
    tuta_i = solve_line_search(a, b)

    return tuta_i



def st_unbal_ot(sc_a, st_b, cost12,
                alpha, ent_reg,  lamda=np.Inf,
                cost1=None, cost2=None,
                numItermax=10, T0=None):
    """
    Unbalanced and structured optimal transport.
    """
    sc_a = np.asarray(sc_a, float).reshape(-1, 1)
    st_b = np.asarray(st_b, float).reshape(-1, 1)

    if T0 is None:
        T_old = np.outer(sc_a, st_b)
    else:
        T_old = T0

    for it in range(numItermax):
        constC, h1a, h2b = gw_loss(cost1, cost2, sc_a, st_b, loss_gw='square_loss')

        gw_grad = 2 * alpha * (constC - np.dot(h1a, T_old).dot(h2b.T))
        Mi = cost12 + gw_grad  # gradient
        Mi += Mi.min()  # set M positive

        # solve linear program
        if lamda == np.inf:
            T_tuta = ot.sinkhorn(sc_a, st_b, Mi, ent_reg)
        else:
            T_tuta = unbal_ot(sc_a, st_b, Mi, ent_reg, lamda=lamda)

        # linear search
        deltaG = T_tuta - T_old
        tuta_i = line_search(G=T_old, deltaG=deltaG, c1=cost1, c2=cost2,
                             alpha=alpha, constC=constC, c12=cost12)

        if tuta_i is None or np.isnan(tuta_i):
            raise NonConvergenceError('tuta_i not found.')
        else:
            T_su = T_old + tuta_i * deltaG
        T_old = T_su

    return T_su



def sw_st_unbal_ot(sc_a, st_b, cost12,
                   alpha, ent_reg, lamda,
                   cost1, spa_cost,
                   numItermax=5):

    T_sw = st_unbal_ot(sc_a=sc_a,
                       st_b=st_b,
                       cost12=cost12,
                       alpha=alpha,
                       ent_reg=ent_reg,
                       lamda=lamda,
                       cost1=cost1,
                       cost2=spa_cost,
                       numItermax=numItermax)
    return T_sw



def transport_plan(cost12, spa_cost,
                   file_path, save=True,
                   alpha=0.1, lamda=1.0, ent_reg=1.0,
                   cost1=None, cost2=None,
                   initdis_method='minus',
                   numItermax=5):
    """
    Computing transport plan.
    """
    if not isinstance(cost12, pd.DataFrame):
        print('Please enter cost12 data of pandas DataFrame type represents the '
              'distance matrix between scRNA-seq and ST data with rows being cells and columns being spots.')
        return None
    if save != True and save != False:
        print("Please select 'True' or 'False' for save.")
        return None
    if not isinstance(alpha, float) or alpha < 0.0 or alpha >= 1.0:
        print("Please enter alpha with float type and from 0 to 1, [0, 1)")
        return None
    if not isinstance(lamda, float) or lamda < 0.0:
        print("Please enter lamda with float type and lamda > 0")
        return None
    if not isinstance(ent_reg, float) or ent_reg <= 0.0:
        print("Please enter ent_reg with float type and ent_reg ≥ 0")
        return None
    if spa_cost is not None and not isinstance(spa_cost, pd.DataFrame):
        print('Please enter spa_cost data of pandas DataFrame type represents the '
              'spatially weighted distance matrix among spots.')
        return None
    if cost1 is not None and not isinstance(cost1, pd.DataFrame):
        print('Please enter cost1 data of pandas DataFrame type represents the '
              'distance matrix of cells in scRNA-seq data.')
        return None
    if cost2 is not None and not isinstance(cost2, pd.DataFrame):
        print('Please enter cost2 data of pandas DataFrame type represents the '
              'distance matrix of spatial coordinates in ST data.')
        return None
    assert (initdis_method in ['minus', 'minus_exp', 'uniform_distribution']), \
        "method argument has to be either one of 'minus', 'minus_exp', 'uniform_distribution'."
    if not isinstance(numItermax, int):
        print('Please enter numItermax of int type.')
        return None

    cell_name = cost12.index
    spot_name = cost12.columns

    cost12 = cost12.to_numpy()
    cost12 = np.asarray(cost12, dtype=np.float64)

    if spa_cost is not None:
        spa_cost = spa_cost.to_numpy()
    if cost1 is not None:
        cost1 = cost1.to_numpy()
        cost1 = np.asarray(cost1, dtype=np.float64)
    if cost2 is not None:
        cost2 = cost2.to_numpy()
        cost2 = np.asarray(cost2, dtype=np.float64)

    sc_a, st_b = init_distribution(cost12, method=initdis_method)

    if alpha == 0.0 and lamda == 0.0:
        T = ot.sinkhorn(a=sc_a, b=st_b, M=cost12, reg=ent_reg)

    if alpha == 0.0 and lamda > 0.0:
        T = unbal_ot(sc_a=sc_a, st_b=st_b, cost12=cost12, ent_reg=ent_reg,
                     lamda=lamda, numItermax=numItermax, stopThr=1e-6)

    if 0.0 < alpha < 1.0 and lamda > 0.0 and spa_cost is None:
         T = st_unbal_ot(sc_a=sc_a, st_b=st_b, cost12=cost12,
                         alpha=alpha, ent_reg=ent_reg, lamda=lamda,
                         cost1=cost1, cost2=cost2,
                         numItermax=numItermax, T0=None)

    if 0.0 < alpha < 1.0 and lamda > 0.0 and spa_cost is not None:
        T = sw_st_unbal_ot(sc_a=sc_a, st_b=st_b, cost12=cost12,
                           alpha=alpha, ent_reg=ent_reg, lamda=lamda,
                           cost1=cost1, spa_cost=spa_cost,
                           numItermax=numItermax)

    T_mapping = T
    T_mapping = pd.DataFrame(T_mapping, index=cell_name, columns=spot_name)

    file_name = 'TransportPlan.csv'
    if save:
        os.makedirs(os.path.join(file_path, 'OptimalTransport'), exist_ok=True)
        T_mapping.to_csv(file_path + 'OptimalTransport/' + file_name, sep=',', index=True, header=True)
        print('The transport plan is saved in ' + file_path + 'OptimalTransport/' + file_name + '.')

    return T_mapping
