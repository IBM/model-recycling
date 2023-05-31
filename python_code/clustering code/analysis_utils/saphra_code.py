import collections
import math
from typing import OrderedDict, List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering, k_means
from sklearn.manifold import spectral_embedding
from torch import nn as nn

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, FixedFormatter, FixedLocator
import seaborn as sns
from matplotlib import pyplot as plt


# https://github.com/aNOnWhyMooS/connectivity/blob/main/interpolate/interpolate_1d.py
from post_experiments_analyzer import columns_replacer


def linear_comb(w1: OrderedDict[str, torch.Tensor],
                w2: OrderedDict[str, torch.Tensor],
                coeff1: float, coeff2: float,
                model: nn.Module) -> None:
    """Linearly combines weights w1 and w2 as coeff1*w1 and coeff2*w2 and loads
    into provided model.
    Args:
        w1:     State dict of first model.
        w2:     State dict of second model.
        coeff1: Coefficient for scaling weights in w1.
        coeff2: Coefficient for scaling weights in w2.
        model:   The model in which to load the linear combination of w1 and w2.
    """
    new_state_dict = collections.OrderedDict()
    buffers = [name for (name, _) in model.named_buffers()]

    for (k1, v1), (k2, v2) in zip(w1.items(), w2.items()):
        if k1 != k2:
            raise ValueError(f"Mis-matched keys {k1} and {k2} encountered while \
                               forming linear combination of weights.")
        if k1 not in buffers:
            new_state_dict[k1] = coeff1 * v1 + coeff2 * v2
        else:
            new_state_dict[k1] = v1
    model.load_state_dict(new_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


# https://github.com/aNOnWhyMooS/connectivity/blob/b251adb69e2829037cb74416b549103817beca60/src/constellations/theoretical/metrics.py
def barrier_height(losses):
    if len(losses) <= 2:
        return 0
    return max([losses[i] - (((len(losses) - i - 1) * losses[0] + i * losses[-1]) / (len(losses) - 1))
                for i in range(len(losses))] + [0])


def connectivity_gap(losses):
    max_height = 0
    for i in range(len(losses) + 1):
        for j in range(len(losses) + 1):
            new_height = barrier_height(losses[i:j])
            if new_height > max_height:
                max_height = new_height
        return max_height


# https://github.com/aNOnWhyMooS/connectivity/blob/b251adb69e2829037cb74416b549103817beca60/src/constellations/theoretical/clustering.py#L10
def get_clusters(dist_matrix: np.ndarray, n_clusters: int = 2, delta=1.0, ):
    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity="precomputed")
    sim_matrix = np.exp(-dist_matrix ** 2 / (2. * delta ** 2))
    sc_out = sc.fit(sim_matrix)
    clusters = {}
    for model_no, cluster in enumerate(sc_out.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(model_no)
    return list(clusters.values())


def get_sc_centroid_dists(cluster_centers, se_matrix, between_centers:bool) -> List[List[float]]:
    dists = []
    if between_centers:
        for center1 in cluster_centers:
            dists.append([float(np.linalg.norm(center2 - center1)) for center2 in cluster_centers])

    else: # dist between centers to models
        for center in cluster_centers:
            dists.append([float(np.linalg.norm(embed - center)) for embed in se_matrix])

    return dists


def get_sc_centroids(matrix, matrix_type, n_clusters, models_names, delta=1.0, method='weights_mean'):
    print(f'getting centroids using {method}')
    if method == 'saphra_method':
        if matrix_type in ['euclidean distance', 'cg']: # convert dist matrix to sim matrix
            matrix = np.exp(-matrix ** 2 / (2. * delta ** 2))

        se = spectral_embedding(matrix,
                                n_components=n_clusters,
                                drop_first=False,
                                )
        cluster_centers, labels, inertia = k_means(se, n_clusters)
        print('spectral embedding shape:', se.shape, 'cluster centers shape: ', cluster_centers.shape)
        return cluster_centers, se

    elif method == 'kmeans': # calc centroids from original matrix (after transformation to similarity matrix)
        cluster_centers, labels, inertia = k_means(matrix, n_clusters)
        print('spectral embedding shape:', len(matrix), len(matrix[0]), 'cluster centers shape: ', cluster_centers.shape)
        return cluster_centers

    elif method == 'weights_mean':
        df = pd.DataFrame(matrix, columns=models_names)
        names = set([name.split('_')[-1] for name in models_names]) # model name
        numbers = set([name.split('_')[0] for name in models_names]) # seed number or train_size
        if len(numbers) == len(names):
            print('Notice- num models == num numbers. Calculating centroids by models.')
        if len(names) == n_clusters: # centroids for each model
            df['model_name'] = [name.split('_')[-1] for name in models_names]
        elif len(numbers) == n_clusters: # centroid for each number
            df['model_name'] = [name.split('_')[0] for name in models_names]
        else:
            print('Cannot calaulte centroids because dont know according to what.')
            return -1
        mean_labels = df.groupby('model_name').mean()
        # mean_as_dict = mean_labels.to_dict('records')
        # mean_as_list = [list(v.values()) for v in mean_as_dict]
        mean_as_list = mean_labels.values.tolist()
        centroids_names = mean_labels.index.tolist()
        centroids_names = [f'cent_{name}' for name in centroids_names]
        return mean_as_list, centroids_names

    else:
        print(f'Not implemented method for getting centroids: {method}')


def annotate(data, **kws): # func from https://github.com/aNOnWhyMooS/connectivity/blob/b251adb69e2829037cb74416b549103817beca60/src/constellations/plot_utils.py#L213
    # n = len(data)
    ax = plt.gca()
    # ax.text(.1, .8, f"N = {int(n / 10)}", transform=ax.transAxes, fontsize=20)
    num_pairs = data.groupby('Position').count().values.tolist()[0][0]
    ax.text(.1, .8, f"N = {num_pairs}", transform=ax.transAxes, fontsize=16)

def annotate_losses_values(data, **kws):
    x = data.iloc[:len(data['Position'].unique())]['Position'].tolist()
    y = data.groupby('Position').mean()['Losses']
    s = y.map('{:,.4f}'.format).tolist()
    y = y.tolist()
    ax = plt.gca()
    # jumps = 2 if len(data['position'].unique()) < 20 else 4
    # for i, (xi, yi, si) in enumerate(zip(x, y, s)):
    #     if i % jumps == 0:
    #         ax.text(xi, yi, si)
    positions = [0, len(x)-1, s.index(min(s))]
    for pos in positions:
        ax.text(x[pos], y[pos], s[pos])

def annotate_models_positions(data, **kws):
    ax = plt.gca()
    ax.axvline(0, data['Losses'].min(), data['Losses'].min(), color='black', linestyle='--')
    ax.axvline(1, data['Losses'].min(), data['Losses'].min(), color='black', linestyle='--')

def df_names_capitalizations(df):
    df.columns = [col.capitalize() for col in df.columns]
    df['Data'] = df['Data'].apply(columns_replacer)
    df['Type'] = df['Type'].apply(lambda x: f'{columns_replacer(x.split("-and-")[0])}-{columns_replacer(x.split("-and-")[1])}')
    return df

def plot_plain_valley_peaks(df, save_file, col, x_axis, y_axis, hue_by, title_add, annot=True, pos=None, return_fig=False, only_interpolation=False):
    # https://github.com/aNOnWhyMooS/connectivity/blob/b251adb69e2829037cb74416b549103817beca60/src/constellations/plot_utils.py#L213
    if type(df)==dict: # for lp_on_interpolations_with_avg
        avg = df['avg']
        df = df['regular']
    else:
        avg = None

    # capitalize names and change datasets names
    df = df_names_capitalizations(df)
    try:
        avg = df_names_capitalizations(avg)
    except:
        avg = None
    col, x_axis, y_axis, hue_by = [param.capitalize() for param in [col, x_axis, y_axis, hue_by]]

    plt.figure(figsize=(10, 8), dpi=100)

    with mpl.rc_context({"lines.linewidth": 2, "axes.titlesize": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16}):

        grid = sns.FacetGrid(
            df,
            margin_titles=True,
            col=col,
            row=hue_by,
            sharex='all',
            sharey='row',
            # col_wrap=3,
            height=3, aspect=1.5,
            hue=hue_by,
        )

        # Draw a line plot to show the trajectory of each random walk
        grid.map(sns.lineplot, x_axis, y_axis, marker="o")
        grid.map_dataframe(annotate)

        if annot:
            grid.map_dataframe(annotate_losses_values)

        # grid.add_legend()
        grid.set(xticks=list(df['Position'].unique()))
        # grid.set(xticks=list(range(len(df['Position'].unique()))))
        n_sample = len(df['Position'].unique()) - 2
        min_pos, max_pos = (0, 1)  if only_interpolation else (int(df['Position'].min()), int(df['Position'].max()))
        grid.set_xticklabels([f'({min_pos},{max_pos})'] + [''] * n_sample + [f'({max_pos},{min_pos})'])


        # fix labels
        desired_left_ylabels = []
        for ax in grid.axes.flat:
            # Make x and y-axis labels slightly larger
            ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')  # Position
            # ax.set_ylabel(ax.get_ylabel(), fontsize='x-large') # Losses
            # ax.set_ylabel("", fontsize='x-large')  # remove right ylabel

            # Make top-xlabel more human-readable and larger
            if ax.get_title():  # {MNLI}-{SST}
                ax.set_title(ax.get_title().split('=')[1],
                             fontsize='x-large')

            # Make right-ylabel more human-readable and larger
            # Only the 2nd and 4th axes have something in ax.texts
            if ax.texts:
                idx=None
                for i in range(len(ax.texts)):
                    if ax.texts[i].get_text().split(' = ')[0] == hue_by:
                        idx = i
                if idx:
                    # This contains the right ylabel text
                    txt = ax.texts[idx]
                    desired_text = 'Average Loss' if txt.get_text().split(' = ')[1] == 'AVG' else f"Loss on {txt.get_text().split('=')[1]}"
                    desired_left_ylabels.insert(0, desired_text)
                    # ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                    #         desired_text,
                    #         transform=ax.transAxes,
                    #         va='center',
                    #         fontsize='x-large',
                    #         rotation=90,
                    #         )
                    # ax.set_ylabel(desired_text, fontsize='x-large')
                    # Remove the original text
                    ax.texts[idx].remove()

        for ax in grid.axes.flat:
            if ax.get_ylabel()=='Losses':
                ax.set_ylabel(desired_left_ylabels.pop())

        grid.fig.subplots_adjust(top=0.85)
        # grid.fig.suptitle(save_file[:-4], size=15)
        for ax in grid.axes.flat:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if pos:
            for ax in grid.axes.flat:
                ax.axvline(x=pos[0], color='black', linestyle=':')
                ax.axvline(x=pos[1], color='black', linestyle=':')

        if avg is not None:
            axes = grid.fig.axes
            for i, ax_idx in enumerate([i for i in range(1, len(axes)+1, int(math.sqrt(len(axes))))]):
                curr_avg_df = avg[avg.Data==avg.Data.unique().tolist()[i]]
                sns.lineplot(data=curr_avg_df,
                    x="Position", y="Losses", ax=axes[ax_idx], label='avg tasks', color='darkviolet')


        # grid.fig.suptitle(f'Losses during interpolations- {title_add}')
        # grid.savefig(save_file)
        if not only_interpolation:
            plt.xscale('symlog') # symlog scale
        plt.tight_layout()
        plt.show()

    if return_fig:
        return grid

def plot_single_interpolation_graph(df, col, x_axis, y_axis, hue_by, factor=1, is_tick=True,  legend=None, save_at=None,
                                    font_scale=1):
    #sns.set(font_scale=2)
    #plt.rcParams.update({'font.size': 0.1*font_scale})
    df = df_names_capitalizations(df)
    col, x_axis, y_axis, hue_by = [param.capitalize() for param in [col, x_axis, y_axis, hue_by]]
    with mpl.rc_context({"lines.linewidth": 2, "axes.titlesize": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16}):
        ax = sns.lineplot(data=df, x=x_axis, y=y_axis, hue=hue_by, marker="o", legend=True)
        handles, labels = ax.get_legend_handles_labels()
        if legend:
            ax.legend(handles=handles, labels=legend)
        else:
            ax.legend(handles=handles, labels=labels)
        ax.set(xlabel=r"$\alpha$")
        if is_tick:
            tick_labels = [x/factor for x in range(11)]
            tick_labels[0] = df.Type.iloc[0].split('-')[0]
            tick_labels[-1] = df.Type.iloc[0].split('-')[1]
            ax.set_xticklabels(tick_labels)
            ax.set_xticks([x/factor  for x in range(11)])
        else:
            ticks_labels = list(df['Position'].unique())
            ax.set_xticks(ticks_labels)
            print(f'ticks_labels: {ticks_labels}')
            #ax.set_xticks(list(range(len(df['Position'].unique()))))
            #n_sample = len(df['Position'].unique()) - 2
            #min_pos, max_pos = (0, 1) if only_interpolation else (int(df['Position'].min()), int(df['Position'].max()))
            #grid.set_xticklabels([f'({min_pos},{max_pos})'] + [''] * n_sample + [f'({max_pos},{min_pos})'])
            ax.axvline(x=0, color='black', linestyle=':')
            ax.axvline(x=1, color='black', linestyle=':')
            plt.xscale('symlog')
        plt.tight_layout()
        #plt.rcParams.update({'font.size': 2*font_scale})

        if save_at:
            plt.savefig(save_at)
        plt.show()



def all_pairs_plot_interpolation_losses(losses: dict, save_file:str, title_add:str):
    """
    Plot interpolation losses for each pair of models
    :param self:
    :return:
    """
    num_plots = len(list(losses.values())[0])
    fig, axs = plt.subplots(num_plots, num_plots, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(num_plots):
     for j in range(num_plots):
         for target in losses.keys():
             loss = losses[target].iloc[i][j]
             if type(loss)!=list:
                 continue
             axs[i][j].scatter(range(len(loss)), loss, s=4)
             axs[i][j].plot(range(len(loss)), loss, label=f'{target}')
         axs[i][j].set_title(f'{losses[target].index[i]}_{losses[target].columns[j]}')
         axs[i][j].tick_params(axis='y', which='major', pad=-5)
    for ax in axs.flat:
     ax.set(xlabel='Position', ylabel='Losses')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
     # ax.label_outer()
     ax._label_outer_xaxis(check_patch=False)
     ss = ax.get_subplotspec()
     label_position = ax.yaxis.get_label_position()
     if not ss.is_first_col():  # Remove left label/ticklabels/offsettext.
         if label_position == "left":
             ax.set_ylabel("")
     if not ss.is_last_col():  # Remove right label/ticklabels/offsettext.
         if label_position == "right":
             ax.set_ylabel("")

    model1, model2 = losses[target].columns[0].split('_')[0], losses[target].index[-1].split('_')[0]
    fig.suptitle(f'{model1} {model2} loss variation during linear interpolation, N={len(loss)}', fontsize=20,
              y=0.95)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=18)
    fig.suptitle(f'All models pairs losses during interpolations-{title_add}', fontsize=20)
    # plt.savefig(save_file)
    plt.show()
    return fig