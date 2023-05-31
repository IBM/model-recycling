import math
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from analysis_utils.calcs import calc_pca, calc_tsne, calc_spectral_embed
from analysis_utils.constants import *

# code from https://stackoverflow.com/questions/42697933/colormap-with-maximum-distinguishable-colours
from post_experiments_analyzer import columns_replacer


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(
        number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades,
                                                          number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (
                        j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)


def plot_clusters(labels, matrix, datasets_and_seeds_str, num_dims, reduction_method: str,
                  matrix_method: str, title_add: str, metrics_df:pd.DataFrame, save_path=None, remove_twitter=False):
    if num_dims == 2:
        plot_2D_clusters(labels, matrix, datasets_and_seeds_str, reduction_method, matrix_method, title_add, metrics_df,
                         save_path, remove_twitter)
    elif num_dims == 3:
        plot_3D_clusters(labels, matrix, datasets_and_seeds_str, reduction_method, matrix_method, title_add, metrics_df)
    else:
        print(f'Error- num dims can be 2 or 3, but it is {num_dims}')


def plot_2D_clusters(labels, matrix, datasets_and_seeds_str, reduction_method: str,
                     matrix_method: str, title_add: str, metrics_df: pd.DataFrame,
                     save_path=None, remove_twitter=False):
    sns.set(font_scale=2)
    if remove_twitter:
        is_tweet = [True if 'tweet' in d else False for d in datasets_and_seeds_str]
        labels = [l for l, is_ in zip(labels, is_tweet) if not is_]
        matrix = [[e for e, is__ in zip(line, is_tweet) if not is__]
                  for line, is_ in zip(matrix, is_tweet) if not is_]
        datasets_and_seeds_str = [l for l, is_ in zip(datasets_and_seeds_str, is_tweet) if not is_]
    if reduction_method == 'pca':
        reduced_df = calc_pca(matrix, 2)
    elif reduction_method == 'tsne':
        reduced_df = calc_tsne(matrix, 2, matrix_method)
    elif reduction_method == 'se': # spectral embedding
        reduced_df = calc_spectral_embed(matrix, 2, matrix_method)
    else:
        print(f'Error- unimplemented reduction method: {reduction_method}')
    reduced_df['cluster'] = labels

    # separate centroids from models
    centroids_names = [name for name in datasets_and_seeds_str if name.split('_')[0]=='cent']
    if len(centroids_names)>0:
        datasets_and_seeds_str = datasets_and_seeds_str[:-len(centroids_names)]
        centroids_df = reduced_df.iloc[-len(centroids_names):]
        reduced_df = reduced_df.iloc[:-len(centroids_names)]

    # cluster by data
    if len(reduced_df['cluster'].unique()) == len(set([name.split('_')[-1] for name in datasets_and_seeds_str])):
        datasets_and_seeds_str = datasets_and_seeds_str # {size}_{data}
    # cluster by size
    elif len(reduced_df['cluster'].unique()) == len(set([name.split('_')[0] for name in datasets_and_seeds_str])):
        datasets_and_seeds_str = [f'{x.split("_")[-1]}_{x.split("_")[0]}' for x in datasets_and_seeds_str] # flip order to {data}_{size}
    else:
        print('Number of clusters do not match either #datasets or #sizes')

    reduced_df['data'] = [name.split('_')[-1] for name in datasets_and_seeds_str]
    try:
        reduced_df['seed'] = [name.split('_')[0].split('s')[1] for name in datasets_and_seeds_str]
    except:
        reduced_df['seed'] = [name.split('_')[0] for name in datasets_and_seeds_str]


    datasets = set([name.split('_')[-1] for name in datasets_and_seeds_str])
    try:
        cm = generate_colormap(len(datasets))
        data2color = {d: cm.colors[i] for i, d in enumerate(datasets)}
    except:
        cm = sns.color_palette("deep", len(datasets))
        data2color = {d: cm[i] for i, d in enumerate(datasets)}

    reduced_df['data_color'] = [data2color[data.split('_')[-1]] for data in datasets_and_seeds_str]
######
    data2cluster = reduced_df.groupby('cluster')['data'].apply(list).apply(lambda x: Counter(x).most_common(1)[0][0])
    while len(data2cluster.unique()) != len(reduced_df.data.unique()):
        missed = list(set(reduced_df.data.unique()) - set(data2cluster.unique()))
        for m in missed:
            cc = reduced_df.groupby('cluster')['data'].apply(list).apply(lambda x: Counter(x))
            cc = cc[cc.apply(lambda x:x[m])==cc.apply(lambda x:x[m]).max()]
            if Counter(data2cluster).most_common(1)[0][1] > 1:
                cc = cc[cc.apply(lambda x: x[m]) == cc.apply(lambda x: x[m]).max()]
                cc = cc[cc.apply(lambda x: Counter(data2cluster).most_common(1)[0][0] in x)]
                cc = cc[cc.apply(lambda x: x[Counter(data2cluster).most_common(1)[0][0]]) ==
                        min(cc.apply(lambda x: x[Counter(data2cluster).most_common(1)[0][0]]))]
                data2cluster.at[cc.index[0]] = m
    cluster2color = {l: data2color[data2cluster[l]] for l in labels if l in data2cluster}
    reduced_df['cluster_color'] = reduced_df['cluster'].apply(lambda cluster: cluster2color[cluster])

    reduced_df.data_color = reduced_df.data_color.apply(lambda st:str(st))
    reduced_df.cluster_color = reduced_df.cluster_color.apply(lambda st: str(st))
    color_2_indx = {c: i for i, c in enumerate(reduced_df.data_color.unique())}
    color_2_type = {r.data_color: r.data for _, r in reduced_df.groupby(['data', 'data_color']).mean().reset_index().iterrows()}
    index_2_type = {p[1]: color_2_type[p[0]] for p in color_2_indx.items()}
    labels = [color_2_indx[x] for x in reduced_df.data_color]
    preds = [color_2_indx[x] for x in reduced_df.cluster_color]
    f1 = f1_score(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average=None)
    recall = recall_score(labels, preds, average=None)

    print(f'index_2_type: {index_2_type}')
    print(f'f1: {f1}')
    print(f'acc: {acc}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    reduced_df.rename(columns={f"{reduction_method}_one": f"t-SNE x",
                               f"{reduction_method}_two": f"t-SNE y"},
                      inplace=True)

    # plot clusters colors
    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(
        x=f"t-SNE x", y=f"t-SNE y",
        hue="cluster",
        palette=cluster2color,  # sns.color_palette("deep", len(set(labels))),
        data=reduced_df,
        legend=None,
        alpha=1.0,
        s=300,
        linewidth=0.0,
    )
    # replace data names to capitalized names
    reduced_df['data'] = reduced_df['data'].apply(columns_replacer)
    reduced_df.rename(columns={'data':'Data'}, inplace=True)
    capitalized_data2color = {columns_replacer(k): v for k,v in data2color.items()}

    # plot data colors
    ax = sns.scatterplot(
        x=f"t-SNE x", y=f"t-SNE y",
        hue="Data",
        palette=capitalized_data2color,  # 'tab20',  # sns.color_palette("deep", len(set(labels))),
        data=reduced_df,
        legend='full',
        alpha=1.0,
        s=100,
        linewidth=0.0,
    )
    #plt.setp(ax.get_legend().get_texts(), fontsize='30')
    ax.legend(loc='center right',
              bbox_to_anchor=(1.16, 0.5),
              #ncol=1
              )

    # plot centroids
    if len(centroids_names)>0:
        if centroids_names[-1] == 'cent_basemodel': # if with_base_model
            data2color['basemodel'] = [0.0, 0.0, 0.0]  # black
            # add to legend only the marker of the base model
            handles, labels = plt.gca().get_legend_handles_labels()
            line = matplotlib.lines.Line2D([0], [0], label='base model', color='k', marker='*', markersize=10)
            handles.extend([line])
            plt.legend(handles=handles)

        ax = sns.scatterplot(
            x=f"{reduction_method}_one", y=f"{reduction_method}_two",
            hue="cluster",
            palette= data2color, #'dark',  # 'tab20',  # sns.color_palette("deep", len(set(labels))),
            data= centroids_df,
            legend=False, #'full',
            marker='*',
            alpha=1.0,
            s=300,
            linewidth=0.0,
        )

    # plot dataset size
    # plot train_size if no seed
    # if len(datasets_and_seeds_str[0].split('_')[0].split('s')) == 1:  # train_size, not seed
    #     for idx, row in reduced_df.iterrows():
    #         ax.text(row[f"{reduction_method}_one"] + 0.0, row[f"{reduction_method}_two"] + 1.5,
    #                 row['seed'], horizontalalignment='center',
    #                 size='medium', color='black', weight='regular')
    # else: # plot regular data size
    #     try:
    #         for idx, row in reduced_df.iterrows():
    #             ax.text(row[f"{reduction_method}_one"] + 0.0, row[f"{reduction_method}_two"] + 1.5,
    #                     DATA_NAME2SIZE[row['data']], horizontalalignment='center',
    #                     size='medium', color='black', weight='regular')
    #     except:
    #         print('\ncannot print data sizes')


    #plt.title(
    #    f'{reduction_method} clusters by {matrix_method} {title_add}, \nfound {len(reduced_df["cluster"].unique())} clusters, Numbers above dots are data sizes', fontsize=20)
    metrics_list = metrics_df[['metric', 'value']].to_dict('records')
    metrics_list = [list(li.values()) for li in metrics_list]
    metrics_list = ',  '.join([i + ': ' + str(round(v, 4)) for i, v in metrics_list])
    print(f'####### metrics_list\n{metrics_list}')
    #plt.figtext(x=0.01, y=0.01, s=metrics_list, fontsize=18)
    # remove axis labels and ticks
    plt.tick_params(axis='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_3D_clusters(labels, matrix, datasets_and_seeds_str, reduction_method: str, matrix_method: str,
                     title_add: str, metrics_df: pd.DataFrame, centroids):
    if reduction_method == 'pca':
        reduced_df = calc_pca(matrix, 3)
    elif reduction_method == 'tsne':
        reduced_df = calc_tsne(matrix, 3, matrix_method)
    else:
        print(f'Error- reduction method can be PCA or TSNE but it is {reduction_method}')
    reduced_df['cluster'] = labels

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=reduced_df[f"{reduction_method}_one"],
        ys=reduced_df[f"{reduction_method}_two"],
        zs=reduced_df[f"{reduction_method}_three"],
        c=reduced_df["cluster"],
        cmap='tab10'
    )
    ax.set_xlabel(f'{reduction_method}_one')
    ax.set_ylabel(f'{reduction_method}_two')
    ax.set_zlabel(f'{reduction_method}_three')
    ax.set_title(f'{reduction_method} clusters by {matrix_method} {title_add}, found {len(set(labels))} labels')
    plt.show()


def plot_heatmap_from_matrix(matrix_df, is_sim: bool, title_add: str, to_annot:bool, one_vs_100=False, to_sort=True):
    if not one_vs_100:
        # change order such that first it will be data name and then seed number
        if '_d_' in matrix_df.columns.tolist()[0]:
            matrix_df.columns = [x.split('_')[-1] + '_' + x.split('_')[0] for x in
                                 matrix_df.columns.tolist()]  # {mnli}_{s0}
            matrix_df.index = [x.split('_')[-1] + '_' + x.split('_')[0] for x in
                               matrix_df.index.tolist()]  # {mnli}_{s0} or {mnli}_{cent}
        else:
            matrix_df.columns = [x.split('_')[1] + '_' + x.split('_')[0] for x in matrix_df.columns.tolist()]  # {mnli}_{s0}
            matrix_df.index = [x.split('_')[1] + '_' + x.split('_')[0] for x in matrix_df.index.tolist()]  # {mnli}_{s0} or {mnli}_{cent}

    # capitalize names and correct dataset names' format
    if not one_vs_100:
        matrix_df.columns = [f'{columns_replacer(name.split("_")[0])}_{columns_replacer(name.split("_")[1])}' for name in matrix_df.columns]
        matrix_df.index = [f'{columns_replacer(name.split("_")[0])}_{columns_replacer(name.split("_")[1])}' for name in matrix_df.index]
    else: # for 1 vs 100
        matrix_df.columns = [f'{columns_replacer(name)}' if name != 'avg_loss' else 'avg_loss' for name in matrix_df.columns]
        matrix_df.index = [f'{columns_replacer(name)}' if name != 'avg_loss' else 'avg_loss' for name in matrix_df.index]

    if to_sort:
        matrix_df = matrix_df[sorted(matrix_df.columns)].sort_index()
        matrix_df = matrix_df.astype(float)

    if to_annot:
        if one_vs_100:
            f=plt.figure(figsize=(20, 20), dpi=100)
            fontsize = 18
            ax = sns.heatmap(matrix_df, square=1, cbar=1, annot_kws={'size': 18}, annot=True, fmt='.3f', cmap = 'RdYlGn_r')
        elif title_add.split('distances')[0]=='Centroids-centroids ':
            f=plt.figure(figsize=(20,20), dpi=100)
            fontsize = 18
            ax = sns.heatmap(matrix_df, square=1, cbar=1, annot_kws={'size': 16}, annot=True, fmt='.4f', cmap='RdYlGn')
        elif len(matrix_df.columns.tolist()) <= 40:
            f=plt.figure(figsize=(25, 10), dpi=100)
            fontsize = 22
            ax = sns.heatmap(matrix_df, square=1, cbar=1, annot_kws={'size': 15}, annot=True, fmt='.4f', cmap='RdYlGn')
        else:
            f=plt.figure(figsize=(40, 10), dpi=100)
            fontsize = 18
            ax = sns.heatmap(matrix_df, square=1, cbar=1, annot_kws={'size': 15}, annot=False, fmt='.4f', cmap='RdYlGn')
    else:
        f=plt.figure(figsize=(20, 20))
        fontsize = 14
        ax = sns.heatmap(matrix_df, square=1, cbar=1, annot_kws={'size': 15})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=fontsize)

    tasks = [name.split('_')[0] for name in matrix_df.columns] # [basemodel, mnli, sst2, ...]
    if not one_vs_100:
        first_tasks_idx = [tasks.index(value) for value in set(tasks)]
    else:
        if len(matrix_df.index) > 5:
            first_tasks_idx = [i for i in range(len(matrix_df.index))]
        else:
            first_tasks_idx = [i for i in range(len(matrix_df.columns))]
    color = 'black' if to_annot else 'white'
    ax.vlines(first_tasks_idx, *ax.get_ylim(), colors=color, linewidth=5.0)
    if not to_annot or one_vs_100:
        ax.hlines(first_tasks_idx, *ax.get_xlim(), colors='white', linewidth=5.0)

    heatmap_type = 'Similarity' if is_sim else 'Distances' if not one_vs_100 else one_vs_100
    #plt.title(f'{heatmap_type} Heatmap \n{title_add}', fontsize=fontsize)
    plt.show()

    if one_vs_100:
        return f

