import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from analysis_utils.calcs import calc_spectral_clustering, calc_clustering_metrics, transform_matrix, calc_models_diff, \
    calc_avg_models_weights, calc_knn_over_matrix
from analysis_utils.data_utils import normalize_weights, flatten_model
from analysis_utils.plots import plot_clusters, plot_heatmap_from_matrix
from analysis_utils.saphra_code import get_sc_centroids, get_sc_centroid_dists


def analyze_clustering(matrix, is_sim, models_names, reduction_method, matrix_method, with_base_model, title_add='',
                       num_clusters=None, save_path=None):
    transformed_matrix = transform_matrix(matrix, is_sim) # convert to similarity matrix if it is a dist matrix - used for clustering calc and for tsne

    # remove base model from clustering calc
    if with_base_model:
        # base_model_name = models_names[-1]
        models_names = models_names[:-1]
        base_model_row = transformed_matrix[-1][:-1]
        full_matrix = transformed_matrix
        transformed_matrix = [l[:-1] for l in transformed_matrix[:-1]]

    if not num_clusters:
        num_clusters = len(set([model.split('_')[-1] for model in models_names])) # not including base model
    print(f'\n\nClustering using {num_clusters} clusters, for {models_names}\n\n')
    sim_labels = calc_spectral_clustering(transformed_matrix, models_names, num_clusters=num_clusters,
                                          affinity='precomputed')
    metrics_df = calc_clustering_metrics(transformed_matrix, sim_labels, 'cosine similarity') # cosine sim is for the silhouette metric

    # get centroids and distances
    centroids, centroids_names = get_sc_centroids(transformed_matrix, matrix_method, num_clusters, models_names, method='weights_mean')
    if with_base_model:
        centroids.append(base_model_row) # add base model as centroid
        centroids_names.append('cent_basemodel')
    centroids = np.array(centroids)
    cent_cent_dists = get_sc_centroid_dists(centroids, None, True)
    cent_model_dists = get_sc_centroid_dists(centroids, transformed_matrix, False)
    cent_dist_df = pd.DataFrame(cent_model_dists, columns=models_names, dtype=float)
    cent_dist_df.index = centroids_names
    cent_df = pd.DataFrame(cent_cent_dists, columns=cent_dist_df.index, dtype=float)
    cent_df.index = cent_dist_df.index
    cent_and_models_df = pd.concat([cent_dist_df, cent_df], axis=1, join='inner')
    plot_heatmap_from_matrix(cent_and_models_df.copy(), is_sim=False, title_add='Centroids-models distances '+title_add, to_annot=True) # cent-models heatmap
    plot_heatmap_from_matrix(cent_df.copy(), is_sim=False, title_add='Centroids-centroids distances' + title_add,
                             to_annot=True) # cent-cent heatmap
    if is_sim:
        cent_and_models_df = 1 - cent_and_models_df / cent_and_models_df.max().max() # convert dist to sim
    matrix_with_centroids_df = combine_matrix_centroids(transformed_matrix, cent_and_models_df, models_names, cent_df.index)

    labels_with_centroids = np.append(sim_labels, [name.split('cent_')[-1] for name in centroids_names])
    names_with_centroids = models_names + centroids_names

    #plot_clusters(sim_labels, transformed_matrix, models_names, num_dims=2, reduction_method='tsne',
    #              matrix_method=matrix_method, metrics_df=metrics_df, title_add=title_add+'-no centroids',
    #              save_path='/u/eladv/fusion/figures/clusters_nli_no_twitter.pdf', remove_twitter=True)

    plot_clusters(sim_labels, transformed_matrix, models_names, num_dims=2, reduction_method='tsne',
                  matrix_method=matrix_method, metrics_df=metrics_df, title_add=title_add + '-no centroids',
                  save_path='/u/eladv/fusion/figures/clusters_nli.pdf', remove_twitter=False)

    #plot_clusters(labels_with_centroids, matrix_with_centroids_df, names_with_centroids, num_dims=2,
    #              reduction_method=reduction_method, matrix_method=matrix_method, title_add=title_add+'-with centroids', metrics_df=metrics_df)


def get_labels_with_centroids(cent_names, models_names, df, labels):
    df = df.loc[cent_names][models_names] # get cent rows with models columns only [num_cents, num_models]
    df['closest'] = df.idxmin(axis=1)
    closest_indices = [models_names.index(i) for i in df['closest'].values.tolist()]
    df['label'] = labels[closest_indices]
    return np.append(labels, df['label'].values.tolist())


def combine_matrix_centroids(matrix, cent_models_dists_df, models_names, cent_names):
    names = []
    for matrix_names in [models_names, cent_models_dists_df.columns.values.tolist()]:
        name2cnt = {n:matrix_names.count(n) for n in matrix_names}
        matrix_names = []
        for key, value in name2cnt.items():
            if value == 1:
                matrix_names.append(key)
            if value > 1:
                for i in range(value):
                    matrix_names.append(f'{key}-{i}')
        names.append(matrix_names)
    models_names = names[0]
    cent_models_dists_df.columns = names[1]

    matrix_df = pd.DataFrame(matrix, columns=models_names, dtype=float)
    matrix_df.index = models_names
    matrix_with_centroids_df = pd.concat([matrix_df, cent_models_dists_df], axis=0)
    for model in models_names:
        for cent in cent_names:
            matrix_with_centroids_df.loc[model][cent] = matrix_with_centroids_df.loc[cent][model]
    return matrix_with_centroids_df


def analyze_weights_n_norms(models_weights, models_names, base_model, weights_reference_point, weights_normalization_type, norm_thr):
    models_weights_diff = []  # num of models x model layers
    flat_models_weights_diff = []
    norms = []
    # chosen_models_indices = []
    for i, model_weights in enumerate(models_weights):
        if weights_reference_point == 'base_model':
            if i == 0:
                print('\nCalculating differences with respect to base model')
            model_weights_diff = calc_models_diff(base_model.base_model.parameters(), model_weights)
        elif weights_reference_point == 'avg weights':
            if i == 0:
                print("\nCalculating differences with respect to average of models' weights")
                avg_weight = calc_avg_models_weights(models_weights)
            model_weights_diff = calc_models_diff(avg_weight, model_weights)
        else:  # reference point is axes origin
            if i == 0:
                print('\nCalculating differences with respect to axes origin')
            model_weights_diff = model_weights
        if i == 0:
            print(f"\nNormalizing weights with respect to {weights_normalization_type}")
        model_weights_diff = normalize_weights(model_weights_diff, weights_normalization_type)
        flat_model_weights_diff = flatten_model(model_weights_diff)
        # permuted_weights_diff = np.random.permutation(flat_model_weights_diff)
        norm = torch.linalg.norm(flat_model_weights_diff)
        if norm >= norm_thr: # and norm not in norms:
            norms.append(norm)
            models_weights_diff.append(model_weights_diff)
            flat_models_weights_diff.append([flat_model_weights_diff])
            # chosen_models_indices.append(i)
        del model_weights
        del model_weights_diff
        del flat_model_weights_diff

    # sum_of_weights = calc_sum_of_weights(models_weights_diff)
    print(f'\nDim of models_weights_diff: {len(models_weights_diff)}x{len(models_weights_diff[0])}')

    print('\ncalculating norms:')
    base_norm = flatten_model(base_model.parameters())
    base_norm = torch.linalg.norm(base_norm)
    print("Base_norm", float(base_norm))
    for i, model_norm in enumerate(norms):
        print(f"Model {models_names[i]} norm", float(model_norm))

    # flat_sum_of_weights = calc_sum_of_weights(flat_models_weights_diff)
    # print_norm_for_weight(0, flat_sum_of_weights, flat_models_weights_diff, weights_names)

    return flat_models_weights_diff


def analyze_matrix(matrix, models_names, is_sim, title_add=''):
    """

    :param matrix: matrix of similarities between pairs of models
    :param models_names:
    :param is_sim:
    :param title_add:
    :return:
    """
    matrix_df = pd.DataFrame(matrix, columns=models_names, dtype=float)
    matrix_df.index = models_names
    print(tabulate(matrix_df, headers='keys', tablefmt='psql'))
    plot_heatmap_from_matrix(matrix_df, is_sim=is_sim, title_add=title_add, to_annot=False)

    calc_knn_over_matrix(matrix_df)
