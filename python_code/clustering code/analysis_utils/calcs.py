import math
import statistics
import time
from typing import List
# import npeet.entropy_estimators as ee
import numpy as np
import pandas as pd
import termtables
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, spectral_embedding
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score, mutual_info_score

from analysis_utils.constants import *
import pprint
from tabulate import tabulate
from bisect import bisect
from statistics import mean


def calc_separation_metric(losses_good, losses_bad):
    """
    Calculate a metric of performance (loss) of group "good" with respect to group "bad". For each model in G
    we calc how many models in B does it outperform, then normalize by len(B), and calc the mean of all models
    in G. Finally we return 1 minus the averaged value we got.
    Meaning of result X: X% of the models in G outperform the models in B
    :param losses_good: list of losses of models in the G group
    :param losses_bad: list of losses of models in the B group
    :return: percentage of models in G that outperform models in B
    """
    losses_bad.sort()
    indices_good = []
    for loss in losses_good:
        indices_good.append(bisect(losses_bad, loss))

    indices_good = [i/len(losses_bad) for i in indices_good]
    proba = mean(indices_good)
    comp_proba = 1.0 - proba
    return comp_proba



def calc_cosine_sim(models_weights_diff):
    n = len(models_weights_diff)
    sim_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
               range(len(models_weights_diff[0]))]  # layer x model x model
    for layer_number in range(len(sim_mat)):
        for i in range(n):
            for k in range(i):
                sim_mat[layer_number][i][k] = torch.cosine_similarity(
                    torch.flatten(models_weights_diff[i][layer_number]),
                    torch.flatten(models_weights_diff[k][layer_number]), dim=0)
                sim_mat[layer_number][k][i] = sim_mat[layer_number][i][k]
            sim_mat[layer_number][i][i] = torch.ones(1)
    return sim_mat


def calc_euclidian_dist(models_weights_diff):
    n = len(models_weights_diff)
    dist_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
                range(len(models_weights_diff[0]))]  # layer x model x model
    for layer_number in range(len(dist_mat)):
        for i in range(n):
            for k in range(i):
                dist_mat[layer_number][i][k] = (
                        torch.flatten(models_weights_diff[i][layer_number]) -
                        torch.flatten(models_weights_diff[k][layer_number])).pow(2).sum()
                dist_mat[layer_number][k][i] = dist_mat[layer_number][i][k]
            dist_mat[layer_number][i][i] = torch.zeros(1)
    return dist_mat


def calc_models_diff(base_model_weights, updated_model_weights):
    result = []
    for j, (updated_model_weight, base_model_weight) in enumerate(zip(updated_model_weights, base_model_weights)):
        result.append(updated_model_weight.detach() - base_model_weight.detach())
    return result


def calc_avg_models_weights(models_weights):
    result = [] # list of tuples of tensors
    layer2weights = {str(i):[] for i in range(len(models_weights[0]))}
    for model_w in models_weights: # model_w = tuple of tensors
        for i, layer in enumerate(model_w): # tensor
            layer2weights[str(i)].append(layer.detach())
    for layer, weights in layer2weights.items():
        result.append(torch.mean(torch.stack(weights)))
    return result


def calc_spectral_clustering(sim_matrix, names, num_clusters, affinity):
    print('\nCalculating clustering')
    clustering = SpectralClustering(n_clusters=num_clusters, assign_labels='discretize', random_state=0,
                                    affinity=affinity)
    clustering.fit(sim_matrix)
    print('clustering labels:')
    for name, cluster in zip(names, clustering.labels_):
        print(f'Model {name}: cluster: {cluster}')
    return clustering.labels_


# not used anymore - changed metrics
def binary_calc_clustering_metrics(labels, models_names):
    # pred_clusters = [(name, cluster) for name, cluster in zip(models_names, labels)]
    '''
    first_seed = models_names[0].split('_')[1] # normally 1
    models_numbers = set([name.split('_')[-1] for name in models_names]) # 1 2 3 4 ... 14
    name_template = re.sub('\d', 'XXX',models_names[0]).replace('XXX', first_seed, 1) # 'seed_{first_seed}_data_XXX'
    models_new_names = [name_template.replace('XXX', i) for i in models_numbers] # 'seed_{first_seed}_data_{i}'
    models_indices = [models_new_names.index(name) for name in models_new_names] # normally [0, 1, ..., 14]
    model2label = {model_num: labels[model_idx] for model_num, model_idx in zip(models_numbers, models_indices)}
    true_labels = [model2label[name.split('_')[-1]] for name in models_new_names]
    '''

    datasets = sorted(set([data[-1] for data in models_names.split('_')]))  # e.g.: 1 4
    data2cluster = {data: idx for idx, data in enumerate(datasets)}  # e.g.: 1:0, 4:1
    true_clusters = [data2cluster[name[-1]] for name in models_names]
    average = 'binary' if len(set(labels)) == 2 else 'weighted'
    f1 = f1_score(true_clusters, labels, average=average)
    precision = precision_score(true_clusters, labels, average=average)
    recall = recall_score(true_clusters, labels, average=average)
    for name, metric in zip(['f1', 'precision', 'recall'], [f1, precision, recall]):
        print(f'{name}: {metric}')


def calc_clustering_metrics(matrix, labels, matrix_method):
    metric = 'cosine' if matrix_method == 'cosine similarity' else 'euclidean'  # ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'nan_euclidean', 'haversine']
    Sil = silhouette_score(matrix, labels, metric=metric)
    CH = calinski_harabasz_score(matrix, labels)
    DB = davies_bouldin_score(matrix, labels)
    lists = []
    for name, metric, range in zip(['Silhouette score', 'Calinski and Harabasz score', 'Davies-Bouldin score'],
                                   [Sil, CH, DB],
                                   ['[-1,1] - higher is better', '[0,inf] - higher is better',
                                    '[0,inf) - smaller is better']):
        # print(f'{name}: {metric} out of range: {range}')
        lists.append([name, metric, range])
    string = termtables.to_string(lists, header=['metric', 'value', 'range'], style=termtables.styles.ascii_thin_double)
    print(string)

    # print table of metrics as a plot
    res_df = pd.DataFrame(columns=['metric', 'value', 'range'], data=lists)
    # fig, ax = plt.subplots()
    # # hide axes
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    #
    # ax.table(cellText=res_df.values, colLabels=res_df.columns, loc='center')
    # plt.title('clustering metrics')
    # fig.tight_layout()
    # plt.show()

    return res_df


def calc_spectral_embed(matrix, num_dims, matrix_method: str):
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.spectral_embedding.html
    if matrix_method in ['euclidean distance', 'cg']: # convert dist matrix to sim matrix
        delta = 1.0
        matrix = np.exp(-matrix ** 2 / (2. * delta ** 2))
    else: # convert to np.array
        matrix = np.array(matrix)

    # For spectral embedding, this should be True as the first eigenvector should be constant vector for connected graph
    # but for spectral clustering, this should be kept as False to retain the first eigenvector.
    drop_first = True

    se_result = spectral_embedding(matrix, n_components=num_dims, drop_first=drop_first)
    dig2str = {1: 'one', 2: 'two', 3: 'three'}
    se_cols = [f'se_{dig2str[k]}' for k in range(1, num_dims + 1)]
    df = pd.DataFrame(columns=se_cols)

    df['se_one'] = se_result[:, 0]
    df['se_two'] = se_result[:, 1]
    if len(se_cols) == 3:
        df['se_three'] = se_result[:, 2]
    return df


def calc_pca(matrix, num_dims):
    pca = PCA(n_components=num_dims)
    pca_result = pca.fit_transform(matrix)

    dig2str = {1: 'one', 2: 'two', 3: 'three'}
    pca_cols = [f'pca_{dig2str[k]}' for k in range(1, num_dims + 1)]
    df = pd.DataFrame(columns=pca_cols)

    df['pca_one'] = pca_result[:, 0]
    df['pca_two'] = pca_result[:, 1]
    if len(pca_cols) == 3:
        df['pca_three'] = pca_result[:, 2]
    return df


def calc_tsne(matrix, num_dims, matrix_method: str):
    tsne = TSNE(n_components=num_dims, metric='precomputed') # “precomputed” - X is assumed to be a distance matrix.
    if matrix_method == 'cosine similarity':
        tsne_results = tsne.fit_transform(1. - np.matrix(matrix))
    elif matrix_method == 'euclidean distance':
        tsne_results = tsne.fit_transform(np.matrix(matrix))
    elif matrix_method == 'mutual information':
        np_mat = np.matrix(matrix)
        tsne_results = tsne.fit_transform(1. - np_mat/np.linalg.norm(np.matrix(matrix)))
    elif matrix_method == 'cka':
        #  transformed = np.exp(-matrix ** 2 / (2. ** 2))
        max_val = matrix.max().max() # https://stackoverflow.com/questions/4064630/how-do-i-convert-between-a-measure-of-similarity-and-a-measure-of-difference-di
        tsne_results = tsne.fit_transform(max_val - np.matrix(matrix))
    elif matrix_method == 'cg':
        tsne_results = tsne.fit_transform(np.matrix(matrix)) # todo- check if its the right transformation
    else:
        print(f'matrix method can be cosine sim or euclidean distance or mutual information but it is {matrix_method}')

    dig2str = {1: 'one', 2: 'two', 3: 'three'}
    tsne_cols = [f'tsne_{dig2str[k]}' for k in range(1, num_dims + 1)]
    df = pd.DataFrame(columns=tsne_cols)

    df['tsne_one'] = tsne_results[:, 0]
    df['tsne_two'] = tsne_results[:, 1]
    if num_dims == 3:
        df['tsne_three'] = tsne_results[:, 2]
    return df


def calc_sum_of_weights(models_weights_diff):
    sum_of_weights = []
    for i, model_weights_diff in enumerate(models_weights_diff):
        if i == 0:
            for j, weight in enumerate(model_weights_diff):
                sum_of_weights.append(weight.clone().detach())
        else:
            for j, weight in enumerate(model_weights_diff):
                sum_of_weights[j] += weight
    return sum_of_weights


def create_matrix(flat_models_weights_diff, matrix_type, debiased=None):
    print("\nCreating matrix between models' weights")
    if matrix_type == 'cosine similarity':
        flat_mat = calc_cosine_sim(flat_models_weights_diff)
    elif matrix_type == 'euclidean distance':
        flat_mat = calc_euclidian_dist(flat_models_weights_diff)
    elif matrix_type == 'mutual information':
        flat_mat = calc_mutual_information(flat_models_weights_diff, is_full=False)
    elif matrix_type == 'cka':
        flat_mat = calc_cka(flat_models_weights_diff, debiased)
    else:
        print(f'Unknown matrix type in analyze_matrix(): {matrix_type}')
        return -1

    return flat_mat


def transform_matrix(matrix, is_sim: bool):
    if is_sim:
        if np.amin(np.array(matrix)) < 0.0:
            trans_matrix = 0.5 * (np.array(matrix) + 1)  # convert to only positive values for spectral clustering (which uses sqrt)
        else:
            trans_matrix = matrix
    else:
        trans_matrix = np.exp(-np.array(matrix) ** 2 / (2. * 1 ** 2)) # convert dist to sim - from Saphra's code
        if np.median(trans_matrix) == 0.0:
            trans_matrix = 1 - np.array(matrix) / np.max(matrix) # convert dist to sim in a different way
    return trans_matrix


def calc_knn_over_matrix(matrix_df):
    model2knn = {}
    for model_name, row in matrix_df.iterrows():
        row.sort_values(inplace=True, ascending=False)
        model2knn[model_name] = row.index.to_list()
    # for k,v in dict(sorted(model2knn.items())).items():
    #     print(k,v)

    df = pd.DataFrame.from_dict(model2knn, orient='index')
    df.columns = [f'{i+1}-NN' for i in range(len(list(model2knn.values())[0]))]
    K = min(10, len(df.columns))
    df = df[[f'{i+1}-NN' for i in range(K)]]
    df = df.sort_index(axis=0) # sort rows by index
    print(tabulate(df, headers='keys', tablefmt='psql'))


#-------------------------------------- CKA metric functions --------------------------------------
# code from https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
"""
How to use: 
- Linear CKA can be computed either based on dot products between examples or dot products between features:
    ⟨vec(𝑋𝑋T),vec(𝑌𝑌T)⟩=||𝑌T𝑋||2F
- The formulation with features (right-hand side) is faster when the number of examples exceeds the number of features:
    cka_from_features = feature_space_linear_cka(X, Y)
- The left-side formulation usage: cka_from_examples = cka(gram_linear(X), gram_linear(Y))
- For computing cka with nonlinear kernels, we can use RBF kernel with the bandwidth set to  12  the median distance in the distance matrix: 
  rbf_cka = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))
- If the number of examples is small, it might help to compute a "debiased" form of CKA. The resulting estimator of CKA is still generally biased, but the bias is reduced.
    use the same functions, with additional argument: debiased=True 
"""


def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x.cpu().detach().numpy() # convert tensor to numpy
  features_y = features_y.cpu().detach().numpy()

  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)


def cca(features_x, features_y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.

    Returns:
    The mean squared CCA correlations between X and Y.
    """
    qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
    qy, _ = np.linalg.qr(features_y)
    result = np.linalg.norm(qx.T.dot(qy)) ** 2 / min(
      features_x.shape[1], features_y.shape[1])

    print('Mean Squared CCA Correlation: {:.5f}'.format(result))
    return result


def calc_cka(models_activations, debiased:bool):
    if len(models_activations[0].shape) == 3: # not flatten - calc per layer
        models_activations = [list(torch.unbind(acts, dim=0)) for acts in models_activations] # list of num_models lists, each inner list have num_layers tensors of shape [num_samples, hid_size]
    else: # calc for flatten activations
        models_activations = [[acts] for acts in models_activations] # list of num_models lists, each inner list have 1 tensor of shape [num_samples, hid_size]

    n = len(models_activations)
    cka_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
               range(len(models_activations[0]))]  # layer x model x model
    for layer_number in range(len(cka_mat)):
        for i in range(n):
            for k in range(i):
                cka_mat[layer_number][i][k] = feature_space_linear_cka(models_activations[i][layer_number], models_activations[k][layer_number], debiased)
                cka_mat[layer_number][k][i] = cka_mat[layer_number][i][k] # symmetric
            cka_mat[layer_number][i][i] = torch.ones(1)

    return cka_mat


#-------------------------------------- MUTUAL INFORMATION EXP --------------------------------------
def calc_mutual_information(models_weights_diff, is_full=False):

    if is_full:
        num_bins = NUM_BINS
        pooling_size = 1
    else:
        # num bins calculation from: https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information#:~:text=There%20is%20no%20best%20number,on%20histograms%20have%20been%20proposed.
        num_bins = math.floor(math.sqrt(2*len(models_weights_diff[0][0])/5))
        pooling_size = 10
    print(f'\nNum bins is {num_bins}, pooling size: {pooling_size}')

    m = nn.AvgPool1d(pooling_size)

    start_time = time.time()
    n = len(models_weights_diff)
    mi_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
               range(len(models_weights_diff[0]))]  # layer x model x model
    for layer_number in range(len(mi_mat)):
        for i in range(n):
            for k in range(i+1):
                print(i, k)
                mi_mat[layer_number][i][k] = calc_mi(

                    m(models_weights_diff[i][layer_number].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy(),
                    m(models_weights_diff[k][layer_number].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy(), num_bins)

                mi_mat[layer_number][k][i] = mi_mat[layer_number][i][k]
            # mi_mat[layer_number][i][i] = torch.ones(1)      not true for mutual information! mi can be > 1
    print(f'\nTotal time for calculating mutual information matrix is: {(time.time()-start_time)/60} mins')
    return mi_mat


# code from: https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
def calc_mi(x: np.array, y: np.array, num_bins:int):
    c_xy = np.histogram2d(x, y, num_bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def shuffle_tensor(t):
    indices = torch.randperm(t.shape[0])
    shuffled_t = t[indices]
    return shuffled_t


def find_num_bins(models_weights:List[List], models_names, bin_range=None):
    """
    After running s1_sst2 & s2_sst2, s1_mnli & s2_mnli, s1_sst2 & s1_mnli and s2_sst2 & s2_mnli, I have choose b=18,000

    :param models_weights:
    :param models_names:
    :param bin_range:
    :return:
    """
    bin_range = range(100, 20000, int((20000-100)/15))

    model1a, name1a = shuffle_tensor(models_weights[0][0].detach().numpy()), models_names[0]
    model2a, name2a = shuffle_tensor(models_weights[1][0].detach().numpy()), models_names[1]
    model1b, name1b = shuffle_tensor(models_weights[2][0].detach().numpy()), models_names[2]
    model2b, name2b = shuffle_tensor(models_weights[3][0].detach().numpy()), models_names[3]

    print('\nCalculating MI for different bins and models')

    for m1_name, m2_name, x, y in [[name1a, name1b, model1a, model1b], [name2a, name2b, model2a, model2b],
                            [name1a, name2a, model1a, model2a], [name1b, name2b, model1b, model2b]]:
        pair_names = ' & '.join((m1_name, m2_name))
        all_mi_1 = []
        # all_mi_2 = []
        all_b = list(bin_range)

        for b in bin_range:
            all_mi_1.append(calc_mi(x,y,b))
            # all_mi_2.append(calc_mi(y,x, b))

        plt.figure(figsize=(20,10))
        plt.plot(all_b, all_mi_1, '-o', label=f"MI of reshuffled {pair_names}", c='blue')
        # plt.plot(all_b, all_mi_2, label="MI of y & x")
        plt.hlines(0.0, all_b[0], all_b[-1], ls='--', label='no MI', color='black')
        plt.legend()
        plt.title(f'MI for randomly reshuffled vectors {pair_names}\nBins: {all_b}')
        plt.xlabel('num bins')
        plt.ylabel('MI')
        plt.ylim((-0.01, plt.ylim()[1]*1.5))
        ticks = [t for i,t in enumerate(all_b) if i % 2 == 0]
        plt.xticks(ticks, ticks)
        plt.show()

        print(f'max MI found for pair: {pair_names} is: {max(all_mi_1)}')


    return 1

# not used
def calc_percentiles(models_weights:List[List]):
    data = torch.flatten(torch.stack([x for xs in models_weights for x in xs])).detach().numpy()
    N = len(data)
    desired_perc = [(i*100)/NUM_BINS for i in range(1,NUM_BINS)]
    start = time.time()
    perc = [np.percentile(data, q) for q in desired_perc]
    print(f'\nCalculating percentiles took: {time.time()-start} secs')
    print(len(perc), perc[:10])
    exit()
    return perc

# not used
def count_num_weights_per_bin(model_weight, percentiles: List):
    hist = np.histogram(model_weight, percentiles)[0]

# not used
def calc_mi_matrix_using_percentiles_and_counts(models_weights):
    percentiles = calc_percentiles(models_weights)
    counts = []
    for model in models_weights:
        counts.append(count_num_weights_per_bin(model, percentiles))

    n = len(models_weights)
    mi_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
              range(len(models_weights[0]))]  # layer x model x model
    for layer_number in range(len(mi_mat)):
        for i in range(n):
            for k in range(i + 1):
                mi_mat[layer_number][i][k] = calc_mi(counts[i], counts[k], num_bins=NUM_BINS)
                mi_mat[layer_number][k][i] = mi_mat[layer_number][i][k]

# not used
def calc_mi_matrix_using_npeet_library_not_bins(models_weights_diff):
    """
    MI calculation is from library: https://github.com/gregversteeg/NPEET
    documentation: https://github.com/gregversteeg/NPEET/blob/master/npeet_doc.pdf
    relevant paper (according to documentation): https://journals.aps.org/pre/pdf/10.1103/PhysRevE.69.066138
    :param models_weights_diff:
    :return:
    """
    start_time = time.time()
    n = len(models_weights_diff)
    mi_mat = [[[[0] for _ in range(n)] for _ in range(n)] for _ in
               range(len(models_weights_diff[0]))]  # layer x model x model

    m = nn.AvgPool1d(100)

    for layer_number in range(len(mi_mat)):
        for i in range(n):
            for k in range(i+1):
                print(i,k)
                mi_mat[layer_number][i][k] = ee.mi(
                    m(models_weights_diff[i][layer_number].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).unsqueeze(1).detach().numpy(),
                    m(models_weights_diff[k][layer_number].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).unsqueeze(
                        1).detach().numpy())
                mi_mat[layer_number][k][i] = mi_mat[layer_number][i][k]
            # mi_mat[layer_number][i][i] = torch.ones(1)      not true for mutual information! mi can be > 1
    print(f'\nTotal time for calculating mutual information matrix is: {(time.time()-start_time)/60} mins')
    return mi_mat