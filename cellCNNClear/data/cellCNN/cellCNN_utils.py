
""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains utility functions.

"""
# The code is slightly changed depending on original implementation to make it compatible with decentralized settings
import os
import errno
from collections import Counter
import numpy as np
import pandas as pd
import copy
import sklearn.utils as sku
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy import stats
from scipy.sparse import coo_matrix
import flowio
try:
    import igraph
except ImportError:
    pass


def random_subsample(X, target_nobs, replace=True):

    """ Draws subsets of cells uniformly at random. """

    nobs = X.shape[0]
    if (not replace) and (nobs <= target_nobs):
        return X
    else:
        indices = np.random.choice(nobs, size=target_nobs, replace=replace)
        return X[indices, :]

def weighted_subsample(X, w, target_nobs, replace=True, return_idx=False):
    nobs = X.shape[0]
    if (not replace) and (nobs <= target_nobs):
        return X
    else:
        indices = weighted_choice(w, target_nobs)
        if return_idx:
            return X[indices], indices
        else:
            return X[indices]
        
def kmeans_subsample(X, n_clusters, random_state=None, n_local_trials=10):

    """ Draws subsets of cells according to kmeans++ initialization strategy.
        Code slightly modified from sklearn, kmeans++ initialization. """

    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape
    x_squared_norms = row_norms(X, squared=True)
    centers = np.empty((n_clusters, n_features))
    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]
    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0].reshape(1, -1), X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)
        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

def outlier_subsample(X, x_ctrl, to_keep, return_idx=False):

    """ Performs outlier selection. """

    outlier_scores = knn_dist(X, x_ctrl, s=100, p=1)
    indices = np.argsort(outlier_scores)[-to_keep:]
    if return_idx:
        return X[indices], outlier_scores[indices], indices
    else:
        return X[indices], outlier_scores[indices]

# extra arguments accepted for backwards-compatibility (with the fcm-0.9.1 package)
def loadFCS(filename, *args, **kwargs):
    f = flowio.FlowData(filename)
    events = np.reshape(f.events, (-1, f.channel_count))
    channels = []
    for i in range(1, f.channel_count+1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            channels.append(f.channels[key]['PnS'])
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            channels.append(f.channels[key]['PnN'])
        else:
            channels.append('None')
    return FcmData(events, channels)

class FcmData(object):
    def __init__(self, events, channels):
        self.channels = channels
        self.events = events
        self.shape = events.shape

    def __array__(self):
        return self.events


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_data(indir, info, marker_names, do_arcsinh, cofactor):
    fnames, phenotypes = info[:, 0], info[:, 1]
    sample_list = []
    for fname in fnames:
        full_path = os.path.join(indir, fname)
        fcs = loadFCS(full_path, transform=None, auto_comp=False)
        marker_idx = [fcs.channels.index(name) for name in marker_names]
        x = np.asarray(fcs)[:, marker_idx]
        if do_arcsinh:
            x = ftrans(x, cofactor)
        sample_list.append(x)
    return sample_list, list(phenotypes)

def save_results(results, outdir, labels):
    csv_dir = os.path.join(outdir, 'exported_filter_weights')
    mkdir_p(csv_dir)
    nmark = len(labels)
    nc = results['w_best_net'].shape[1] - (nmark+1)
    labels_ = labels + ['constant'] + ['out %d' % i for i in range(nc)]
    w = pd.DataFrame(results['w_best_net'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_best_net.csv'), index=False)
    w = pd.DataFrame(results['selected_filters'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_consensus.csv'), index=False)
    w = pd.DataFrame(results['clustering_result']['w'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_all.csv'), index=False)

def get_items(l, idx):
    return [l[i] for i in idx]

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def ftrans(x, c):
    return np.arcsinh(1./c * x)

def rectify(X):
    return np.max(np.hstack([X.reshape(-1, 1), np.zeros((X.shape[0], 1))]), axis=1)

def relu(x):
    return x * (x > 0)

def combine_samples(data_list, sample_id):
    accum_x, accum_y = [], []
    for x, y in zip(data_list, sample_id):
        accum_x.append(x)
        accum_y.append(y * np.ones(x.shape[0], dtype=int))
    return np.vstack(accum_x), np.hstack(accum_y)

def keras_param_vector(params):
    W = np.squeeze(params[0])
    b = params[1]
    W_out = params[2]
    # store the (convolutional weights + biases + output weights) per filter
    W_tot = np.hstack([W, b.reshape(-1, 1), W_out])
    return W_tot

def representative(data, metric='cosine', stop=None):
    if stop is None:
        i = np.argmax(np.sum(pairwise_kernels(data, metric=metric), axis=1))
    else:
        i = np.argmax(np.sum(pairwise_kernels(data[:, :stop], metric=metric), axis=1))
    return data[i]

def cluster_tightness(data, metric='cosine'):
    centroid = np.mean(data, axis=0).reshape(1, -1)
    return np.mean(pairwise_kernels(data, centroid, metric=metric))

def cluster_profiles(param_dict, nmark, accuracies, accur_thres=.99,
                     dendrogram_cutoff=.5):
    accum = []
    # if not at least 3 models reach the accuracy threshold, select the filters from the 3 best
    if np.sort(accuracies)[-3] < accur_thres:
        accur_thres = np.sort(accuracies)[-3]

    # combine filters from multiple models
    for i, params in param_dict.items():
        if accuracies[i] >= accur_thres:
            W_tot = keras_param_vector(params)
            accum.append(W_tot)
    w_strong = np.vstack(accum)

    # perform hierarchical clustering on cosine distances
    Z = linkage(w_strong[:, :nmark+1], 'average', metric='cosine')
    clusters = fcluster(Z, dendrogram_cutoff, criterion='distance') - 1
    c = Counter(clusters)
    cons = []
    for key, val in c.items():
        if val > 1:
            members = w_strong[clusters == key]
            cons.append(representative(members, stop=nmark+1))
    if cons != []:
        cons_profile = np.vstack(cons)
    else:
        cons_profile = None
    cl_res = {'w': w_strong, 'cluster_linkage': Z, 'cluster_assignments': clusters}
    return cons_profile, cl_res

def normalize_outliers(X, lq=.5, hq=99.5, stop=None):
    if stop is None:
        stop = X.shape[1]
    for jj in range(stop):
        marker_t = X[:, jj]
        low, high = np.percentile(marker_t, lq), np.percentile(marker_t, hq)
        X[marker_t < low, jj] = low
        X[marker_t > high, jj] = high
    return X

def normalize_outliers_to_control(ctrl_list, list2, lq=.5, hq=99.5, stop=None):
    X = np.vstack(ctrl_list)
    accum = []
    if stop is None:
        stop = X.shape[1]

    for xx in ctrl_list + list2:
        for jj in range(stop):
            marker_ctrl = X[:, jj]
            low, high = np.percentile(marker_ctrl, lq), np.percentile(marker_ctrl, hq)
            marker_t = xx[:, jj]
            xx[marker_t < low, jj] = low
            xx[marker_t > high, jj] = high
        accum.append(xx)
    return accum

## Utilities for generating random subsets ##

def filter_per_class(X, y, ylabel):
    return X[np.where(y == ylabel)]

def per_sample_subsets(X, nsubsets, ncell_per_subset, k_init=False):
    nmark = X.shape[1]
    shape = (nsubsets, nmark, ncell_per_subset)
    Xres = np.zeros(shape)

    if not k_init:
        for i in range(nsubsets):
            X_i = random_subsample(X, ncell_per_subset)
            Xres[i] = X_i.T
    else:
        for i in range(nsubsets):
            X_i = random_subsample(X, 2000)
            X_i = kmeans_subsample(X_i, ncell_per_subset, random_state=i)
            Xres[i] = X_i.T
    return Xres
def generate_subsets(X, pheno_map, sample_id, nsubsets, ncell,
                     per_sample=False, k_init=False):
    S = dict()
    n_out = len(np.unique(sample_id))

    for ylabel in range(n_out):
        X_i = filter_per_class(X, sample_id, ylabel)
        if per_sample:
            S[ylabel] = per_sample_subsets(X_i, nsubsets, ncell, k_init)
        else:
            if nsubsets[0]==float("inf"):
                n= 1000
            else:
                n = nsubsets[pheno_map[ylabel]]
            S[ylabel] = per_sample_subsets(X_i, int(n), int(ncell), k_init)
    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))

    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)
    Xt, yt = sku.shuffle(Xt, yt)
    return Xt, yt

def per_sample_biased_subsets(X, x_ctrl, nsubsets, ncell_final, to_keep, ratio_biased):
    nmark = X.shape[1]
    Xres = np.empty((nsubsets, nmark, ncell_final))
    nc_biased = int(ratio_biased * ncell_final)
    nc_unbiased = ncell_final - nc_biased

    for i in range(nsubsets):
        x_unbiased = random_subsample(X, nc_unbiased)
        if (i % 100) == 0:
            x_outlier, outlierness = outlier_subsample(X, x_ctrl, to_keep)
        x_biased = weighted_subsample(x_outlier, outlierness, nc_biased)
        Xres[i] = np.vstack([x_biased, x_unbiased]).T
    return Xres

def generate_biased_subsets(X, pheno_map, sample_id, x_ctrl, nsubset_ctrl, nsubset_biased,
                            ncell_final, to_keep, id_ctrl, id_biased):
    S = dict()
    for ylabel in id_biased:
        X_i = filter_per_class(X, sample_id, ylabel)
        n = nsubset_biased[pheno_map[ylabel]]
        S[ylabel] = per_sample_biased_subsets(X_i, x_ctrl, n,
                                              ncell_final, to_keep, 0.5)
    for ylabel in id_ctrl:
        X_i = filter_per_class(X, sample_id, ylabel)
        S[ylabel] = per_sample_subsets(X_i, nsubset_ctrl, ncell_final, k_init=False)

    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))
    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)
    Xt, yt = sku.shuffle(Xt, yt)
    return Xt, yt

def single_filter_output(filter_params, valid_samples, mp):
    y_pred = np.zeros(len(valid_samples))
    nmark = valid_samples[0].shape[1]
    w, b = filter_params[:nmark], filter_params[nmark]
    w_out = filter_params[nmark+1:]

    for i, x in enumerate(valid_samples):
        g = relu(np.sum(w.reshape(1, -1) * x, axis=1) + b)
        ntop = max(1, int(mp/100. * x.shape[0]))
        gpool = np.mean(np.sort(g)[-ntop:])
        y_pred[i] = gpool
    return y_pred, np.argmax(w_out)


def get_filters_classification(filters, scaler, valid_samples, valid_phenotypes, mp):
    y_true = np.array(valid_phenotypes)
    filter_diff = np.zeros(len(filters))

    if scaler is not None:
        valid_samples = copy.deepcopy(valid_samples)
        valid_samples = [scaler.transform(x) for x in valid_samples]

    for i, filter_params in enumerate(filters):
        y_pred, filter_class = single_filter_output(filter_params, valid_samples, mp)
        filter_diff[i] = np.mean(y_pred[y_true == filter_class]) -\
                         np.mean(y_pred[y_true != filter_class])
    return filter_diff

def get_filters_regression(filters, scaler, valid_samples, valid_phenotypes, mp):
    y_true = np.array(valid_phenotypes)
    filter_tau = np.zeros(len(filters))

    if scaler is not None:
        valid_samples = copy.deepcopy(valid_samples)
        valid_samples = [scaler.transform(x) for x in valid_samples]

    for i, filter_params in enumerate(filters):
        y_pred, _dummy = single_filter_output(filter_params, valid_samples, mp)
        # compute Kendall's tau for filter i
        w_out = filter_params[-1]
        filter_tau[i] = stats.kendalltau(y_true, w_out * y_pred)[0]
    return filter_tau

def get_selected_cells(filter_w, data, scaler=None, filter_response_thres=0,
                       export_continuous=False):
    nmark = data.shape[1]
    if scaler is not None:
        data = scaler.transform(data)
    w, b = filter_w[:nmark], filter_w[nmark]
    g = np.sum(w.reshape(1, -1) * data, axis=1) + b
    if export_continuous:
        g = relu(g).reshape(-1, 1)
        g_thres = (g > filter_response_thres).reshape(-1, 1)
        return np.hstack([g, g_thres])
    else:
        return (g > filter_response_thres).astype(int)

def create_graph(x1, k, g1=None, add_filter_response=False):

    # compute pairwise distances between all points
    # optionally, add cell filter activity as an extra feature
    if add_filter_response:
        x1 = np.hstack([x1, g1.reshape(-1, 1)])

    d = pairwise_distances(x1, metric='euclidean')
    # create a k-NN graph
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]

    # create a weighted adjacency matrix from the distances (use gaussian kernel)
    # code from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    gauss_sigma = np.mean(d[:, -1])**2
    w = np.exp(- d**2 / gauss_sigma)

    # weight matrix
    M = x1.shape[0]
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = w.reshape(M*k)
    W = coo_matrix((V, (I, J)), shape=(M, M))
    W.setdiag(0)
    adj = W.todense()

    # now reweight graph edges according to cell filter response similarity
    #def min_kernel(v):
    #   xv, yv = np.meshgrid(v, v)
    #   return np.minimum(xv, yv)
    #activity_kernel = pairwise_kernels(g1.reshape(-1, 1), g1.reshape(-1, 1), metric="rbf")
    #activity_kernel = min_kernel(g1)
    #adj = np.multiply(activity_kernel, adj)

    # create a graph from the adjacency matrix
    # first add the adges (binary matrix)
    G = igraph.Graph.Adjacency((adj > 0).tolist())
    # specify that the graph is undirected
    G.to_undirected()
    # now add weights to the edges
    G.es['weight'] = adj[adj.nonzero()]
    # print a summary of the graph
    igraph.summary(G)
    return G

def generate_data(train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                maxpool_percentages=[0.01, 1., 5., 20., 100.], nfilter_choice=range(3, 10),
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1):

    #mkdir_p(outdir)


    # copy the list of samples so that they are not modified in place
    train_samples = copy.deepcopy(train_samples)
    if valid_samples is not None:
        valid_samples = copy.deepcopy(valid_samples)

    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':
        ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
        test_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) != 0)[0]]
        train_samples = normalize_outliers_to_control(ctrl_list, test_list)

        if valid_samples is not None:
            ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
            test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) != 0)[0]]
            valid_samples = normalize_outliers_to_control(ctrl_list, test_list)

    # merge all input samples (X_train, X_valid)
    # and generate an identifier for each of them (train_id, valid_id)
    if (valid_samples is None) and (not generate_valid_set):
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)

    elif (valid_samples is None) and generate_valid_set:
        sample_ids = range(len(train_phenotypes))
        X, sample_id = combine_samples(train_samples, sample_ids)
        valid_phenotypes = train_phenotypes

        # split into train-validation partitions
        eval_folds = 5
        #kf = StratifiedKFold(sample_id, eval_folds)
        #train_indices, valid_indices = next(iter(kf))
        kf = StratifiedKFold(n_splits=eval_folds)
        train_indices, valid_indices = next(kf.split(X, sample_id))
        X_train, id_train = X[train_indices], sample_id[train_indices]
        X_valid, id_valid = X[valid_indices], sample_id[valid_indices]

    else:
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)
        sample_ids = range(len(valid_phenotypes))
        X_valid, id_valid = combine_samples(valid_samples, sample_ids)

    if quant_normed:
        print("quant normed")
        z_scaler = StandardScaler(with_mean=True, with_std=False)
        z_scaler.fit(0.5 * np.ones((1, X_train.shape[1])))
        X_train = z_scaler.transform(X_train)
    elif scale:
        print("scale")
        z_scaler = StandardScaler(with_mean=True, with_std=True)
        z_scaler.fit(X_train)
        X_train = z_scaler.transform(X_train)
    else:
        print("no scaling")
        z_scaler = None

    X_train, id_train = shuffle(X_train, id_train)
    train_phenotypes = np.asarray(train_phenotypes)

    # an array containing the phenotype for each single cell
    y_train = train_phenotypes[id_train]

    if (valid_samples is not None) or generate_valid_set:
        if scale:
            X_valid = z_scaler.transform(X_valid)

        X_valid, id_valid = shuffle(X_valid, id_valid)
        valid_phenotypes = np.asarray(valid_phenotypes)
        y_valid = valid_phenotypes[id_valid]

    # number of measured markers
    nmark = X_train.shape[1]

    # generate multi-cell inputs
    print('Generating multi-cell inputs...')

    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = X_train[y_train == 0]
        to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotypes)))
        nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)

        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotypes))):
            nsubset_biased.append(nsubset / np.sum(train_phenotypes == pheno))

        X_tr, y_tr = generate_biased_subsets(X_train, train_phenotypes, id_train, x_ctrl_train,
                                             nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                             id_ctrl=np.where(train_phenotypes == 0)[0],
                                             id_biased=np.where(train_phenotypes != 0)[0])

        if (valid_samples is not None) or generate_valid_set:
            x_ctrl_valid = X_valid[y_valid == 0]
            nsubset_ctrl = nsubset / np.sum(valid_phenotypes == 0)

            # generate a fixed number of subsets per class
            nsubset_biased = [0]
            for pheno in range(1, len(np.unique(valid_phenotypes))):
                nsubset_biased.append(nsubset / np.sum(valid_phenotypes == pheno))

            to_keep = int(0.1 * (X_valid.shape[0] / len(valid_phenotypes)))
            X_v, y_v = generate_biased_subsets(X_valid, valid_phenotypes, id_valid, x_ctrl_valid,
                                               nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                               id_ctrl=np.where(valid_phenotypes == 0)[0],
                                               id_biased=np.where(valid_phenotypes != 0)[0])

        else:
            cut = X_tr.shape[0] / 5
            X_v = X_tr[:cut]
            y_v = y_tr[:cut]
            X_tr = X_tr[cut:]
            y_tr = y_tr[cut:]
    else:
        # generate 'nsubset' multi-cell inputs per input sample
        if per_sample:
            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset, ncell, per_sample)
            if (valid_samples is not None) or generate_valid_set:
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset, ncell, per_sample)
        # generate 'nsubset' multi-cell inputs per class
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(train_phenotypes))):
                nsubset_list.append(nsubset / np.sum(train_phenotypes == pheno))
            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset_list, ncell, per_sample)

            if (valid_samples is not None) or generate_valid_set:
                nsubset_list = []
                for pheno in range(len(np.unique(valid_phenotypes))):
                    nsubset_list.append(nsubset / np.sum(valid_phenotypes == pheno))
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset_list, ncell, per_sample)

    mkdir_p(outdir + 'X_train/')
    mkdir_p(outdir + 'X_valid/')
    for i in range(len(X_tr)):
        np.savetxt(outdir + 'X_train/' + str(i) +'.txt', X_tr[i])
    np.savetxt(outdir + 'y_train.txt', y_tr)
    if generate_valid_set:
        for i in range(len(X_v)):
            np.savetxt(outdir + 'X_valid/' + str(i) +'.txt', X_v[i])
        np.savetxt(outdir + 'y_valid.txt', y_v)
    print('Done.')
    if generate_valid_set:
        return z_scaler,X_tr,y_tr,X_v,y_v
    else:
        return z_scaler,X_tr,y_tr
    
def generate_normalized_data(train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random'):

    # copy the list of samples so that they are not modified in place
    train_samples = copy.deepcopy(train_samples)
    if valid_samples is not None:
        valid_samples = copy.deepcopy(valid_samples)

    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':
        ctrl_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
        test_list = [train_samples[i] for i in np.where(np.array(train_phenotypes) != 0)[0]]
        train_samples = normalize_outliers_to_control(ctrl_list, test_list)

        if valid_samples is not None:
            ctrl_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
            test_list = [valid_samples[i] for i in np.where(np.array(valid_phenotypes) != 0)[0]]
            valid_samples = normalize_outliers_to_control(ctrl_list, test_list)

    # merge all input samples (X_train, X_valid)
    # and generate an identifier for each of them (train_id, valid_id)
    if (valid_samples is None) and (not generate_valid_set):
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)

    elif (valid_samples is None) and generate_valid_set:
        sample_ids = range(len(train_phenotypes))
        X, sample_id = combine_samples(train_samples, sample_ids)
        valid_phenotypes = train_phenotypes

        # split into train-validation partitions
        eval_folds = 5
        #kf = StratifiedKFold(sample_id, eval_folds)
        #train_indices, valid_indices = next(iter(kf))
        kf = StratifiedKFold(n_splits=eval_folds)
        train_indices, valid_indices = next(kf.split(X, sample_id))
        X_train, id_train = X[train_indices], sample_id[train_indices]
        X_valid, id_valid = X[valid_indices], sample_id[valid_indices]

    else:
        sample_ids = range(len(train_phenotypes))
        X_train, id_train = combine_samples(train_samples, sample_ids)
        sample_ids = range(len(valid_phenotypes))
        X_valid, id_valid = combine_samples(valid_samples, sample_ids)

    if quant_normed:
        z_scaler = StandardScaler(with_mean=True, with_std=False)
        z_scaler.fit(0.5 * np.ones((1, X_train.shape[1])))
        X_train = z_scaler.transform(X_train)
    elif scale:
        z_scaler = StandardScaler(with_mean=True, with_std=True)
        z_scaler.fit(X_train)
        X_train = z_scaler.transform(X_train)
    else:
        z_scaler = None

    X_train, id_train = shuffle(X_train, id_train)
    train_phenotypes = np.asarray(train_phenotypes)

    # an array containing the phenotype for each single cell
    y_train = train_phenotypes[id_train]

    if (valid_samples is not None) or generate_valid_set:
        if scale:
            X_valid = z_scaler.transform(X_valid)

        X_valid, id_valid = shuffle(X_valid, id_valid)
        valid_phenotypes = np.asarray(valid_phenotypes)
        y_valid = valid_phenotypes[id_valid]

    # number of measured markers
    nmark = X_train.shape[1]

    # generate multi-cell inputs
    print('Generating multi-cell inputs...')

    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = X_train[y_train == 0]
        to_keep = int(0.1 * (X_train.shape[0] / len(train_phenotypes)))
        nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)

        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotypes))):
            nsubset_biased.append(nsubset / np.sum(train_phenotypes == pheno))

        X_tr, y_tr = generate_biased_subsets(X_train, train_phenotypes, id_train, x_ctrl_train,
                                             nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                             id_ctrl=np.where(train_phenotypes == 0)[0],
                                             id_biased=np.where(train_phenotypes != 0)[0])

        if (valid_samples is not None) or generate_valid_set:
            x_ctrl_valid = X_valid[y_valid == 0]
            nsubset_ctrl = nsubset / np.sum(valid_phenotypes == 0)

            # generate a fixed number of subsets per class
            nsubset_biased = [0]
            for pheno in range(1, len(np.unique(valid_phenotypes))):
                nsubset_biased.append(nsubset / np.sum(valid_phenotypes == pheno))

            to_keep = int(0.1 * (X_valid.shape[0] / len(valid_phenotypes)))
            X_v, y_v = generate_biased_subsets(X_valid, valid_phenotypes, id_valid, x_ctrl_valid,
                                               nsubset_ctrl, nsubset_biased, ncell, to_keep,
                                               id_ctrl=np.where(valid_phenotypes == 0)[0],
                                               id_biased=np.where(valid_phenotypes != 0)[0])
        else:
            cut = X_tr.shape[0] / 5
            X_v = X_tr[:cut]
            y_v = y_tr[:cut]
            X_tr = X_tr[cut:]
            y_tr = y_tr[cut:]
    else:
        # generate 'nsubset' multi-cell inputs per input sample
        if per_sample:
            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset, ncell, per_sample)
            if (valid_samples is not None) or generate_valid_set:
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset, ncell, per_sample)
        # generate 'nsubset' multi-cell inputs per class
        else:
            nsubset_list = []
            print(train_phenotypes)
            for pheno in range(len(np.unique(train_phenotypes))):
                nsubset_list.append(nsubset // np.sum(train_phenotypes == pheno))

            X_tr, y_tr = generate_subsets(X_train, train_phenotypes, id_train,
                                          nsubset_list, ncell, per_sample)

            if (valid_samples is not None) or generate_valid_set:
                nsubset_list = []
                for pheno in range(len(np.unique(valid_phenotypes))):
                    nsubset_list.append(nsubset // np.sum(valid_phenotypes == pheno))
                X_v, y_v = generate_subsets(X_valid, valid_phenotypes, id_valid,
                                            nsubset_list, ncell, per_sample)
                
    mkdir_p(outdir + 'X_train/')
    mkdir_p(outdir + 'X_test/')

    for i in range(len(X_tr)):
        np.savetxt(outdir + 'X_train/' + str(i) +'.txt', normalize(X_tr[i]))
    np.savetxt(outdir + 'y_train.txt', y_tr)
    for i in range(len(X_v)):
        np.savetxt(outdir + 'X_test/' + str(i) +'.txt', normalize(X_v[i]))
    np.savetxt(outdir + 'y_test.txt', y_v)
    print('Done.')