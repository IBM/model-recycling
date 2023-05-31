import argparse
import numpy as np
import seaborn as sns

from analysis_utils.analyze_utils import analyze_clustering, analyze_weights_n_norms, analyze_matrix
from analysis_utils.calcs import calc_spectral_clustering, \
    calc_clustering_metrics, create_matrix, find_num_bins
from analysis_utils.data_utils import load_matrix, save_matrix, get_models_weights
from analysis_utils.plots import plot_clusters
from analysis_utils.saphra_code import barrier_height, connectivity_gap, get_clusters, get_sc_centroid_dists
from analysis_utils.constants import *
from utils.model_loading_utils import ModelLoadingInfo, load_model
from clearml import Task

sns.set_theme()


def run_saphra_analysis(datasets_and_seeds_str, data_num2name):
    # ------------------------------- saphra code
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # xy_min = (0, 1)
    # xy_max = (1, 0)
    # coef_samples = np.linspace(xy_min, xy_max, args.n_sample + 2).tolist()
    # columns = ["point_num", "loss"]
    # for k in range(args.n_sample + 2):
    #     coeffs_t = coef_samples[k]
    #     print(f'{coeffs_t}')
    #     linear_comb(w1, w2, coeffs_t[0], coeffs_t[1], model)
    #     metrics = eval(input_target_loader, model,
    #                    criterion, pred_fn, metric)

    v1 = np.array([0.48, 0.43, 0.41, 0.405, 0.41, 0.408, 0.05, 0.41, 0.43, 0.48])  # mnli, gen-to-gen
    v2 = np.array([0.48, 0.43, 0.42, 0.445, 0.5, 0.51, 0.46, 0.44, 0.45, 0.5])  # mnli, gen-to-heur
    v3 = np.array([0.51, 0.45, 0.425, 0.43, 0.45, 0.45, 0.43, 0.425, 0.45, 0.5])  # mnli, heur-to-heur

    for losses in [v1, v2, v3]:
        bh_metric = barrier_height(losses)
        cg_metric = connectivity_gap(losses)
        print(f'bh: {bh_metric}')
        print(f'cg: {cg_metric}\n')

    dist_matrix = np.zeros((len([v1, v2, v3]), len([v1, v2, v3])))
    models_losses = [[1.0, v1, v3], [v1, 1.0, v2], [v3, v2, 1.0]]
    for i in range(len(models_losses)):
        for j in range(len(models_losses)):
            if i == j:
                dist_matrix[i][j] = 1.0
            else:
                dist_matrix[i][j] = connectivity_gap(models_losses[i][j])

    # dist_matrix = np.array([[1.0, connectivity_gap(v1)], [connectivity_gap(v2), 1.0]])

    print('dist matrix:\n', dist_matrix)

    clusters = get_clusters(dist_matrix, 2)

    dists = get_sc_centroid_dists(dist_matrix)
    print('centroid dists:\n', dists)
    # indices = [[acc_keys.index(k) for k in cluster] for cluster in clusters]

    # clusters = [[suf_ordered_models[idx] for idx in cluster]
    #             for cluster in clusters]

    print("Clusters found:", clusters)

    # my clustering code
    sim_matrix = np.exp(-dist_matrix ** 2 / (2. * 1 ** 2))
    models_names = [f'v{i}' for i in range(1, len(models_losses) + 1)]
    sim_labels = calc_spectral_clustering(sim_matrix, models_names, num_clusters=2,
                                          affinity='precomputed')
    metrics_df = calc_clustering_metrics(sim_matrix, sim_labels, datasets_and_seeds_str)
    plot_clusters(sim_labels, sim_matrix, datasets_and_seeds_str, num_dims=2, reduction_method='tsne',
                  matrix_method='cosine similarity', title_add='', metrics_df=metrics_df)


def run_analysis(depended_expr_dirs, base_model_name, desired_datasets, weights_reference_point,
                 weights_normalization_type, norm_thr, matrix_type, dim_reduction_method, save_matrix_flag,
                 with_base_model:bool, model_paths=None, is_classification=True, to_cluster=True, matrix_file_name=None,
                 return_matrix=False, clearml_task=None, dataset_group='GLUE_AND_SUPER_GLUE'):
    """

    :param matrix_file_name: file_name of ready matrix to load, or None
    :param
    :return:
    """
    if model_paths is None:
        model_paths = []
    print('\nGetting data')
    if matrix_file_name: # load ready matrix
        print(f'\nLoading matrix from {matrix_file_name}')
        matrix, models_names = load_matrix(matrix_file_name)
        data_num2name = {num.split('_')[-1]: DATA_NUM2NAME[dataset_group][num.split('_')[-1]] for num in models_names}
        short_models_names = [s.replace('seed_', 's').replace('data', 'd') for s in models_names]
        short_models_names = [s.split('d_')[0] + data_num2name[s.split('d_')[-1]] for s in short_models_names]
        matrix_type = 'cosine similarity' if matrix_file_name.split('/')[-1].split('_')[0]=='sim' else 'euclidean distance' if matrix_file_name.split('/')[-1].split('_')[0]=='euc' else 'mutual information' if matrix_file_name.split('/')[-1].split('_')[0]=='mi' else None
    else: # get data
        print('\nCalculating matrix')
        if type(base_model_name)==str: # needs to load model
            base_model_name = load_model(ModelLoadingInfo(classification=is_classification, name=base_model_name, from_tf=False, tokenizer_name=base_model_name))

        models_weights, models_names = get_models_weights(depended_expr_dirs, base_model_name, desired_datasets,
                                                          is_classification, model_paths)

        short_models_names = [s.replace('seed_', 's').replace('data', 'd') for s in models_names] # seed_0_data_1 -> s0_d_1
        if models_names[0].split('_')[-1].isdigit():
            data_num2name = {name.split('_')[-1]: DATA_NUM2NAME[dataset_group][name.split('_')[-1]] for name in models_names}
            short_models_names = [s.split('d_')[0] + data_num2name[s.split('d_')[-1]] for s in short_models_names] # s0_mnli

        # add base model as a model
        if with_base_model:
            # assert weights_reference_point != 'base_model', 'reference point is base_model, but base model is one of the models'
            models_weights.append(tuple(base_model_name.base_model.parameters()))
            short_models_names.append('s0_basemodel')
            models_names.append('base_model')

        flat_models_weights_diff = analyze_weights_n_norms(models_weights, short_models_names, base_model_name, weights_reference_point, weights_normalization_type, norm_thr)
        matrix = create_matrix(flat_models_weights_diff, matrix_type)
        matrix = matrix[0]
        matrix = [[float(d) for d in data] for data in matrix]
        if save_matrix_flag:
            s_matrix_type = 'sim' if matrix_type == 'cosine similarity' else 'euc' if matrix_type == 'euclidean distance' else 'mi' if matrix_type == 'mutual information' else matrix_type
            s_models_names = 'glue_super_glue_70' if not desired_datasets else '_'.join(desired_datasets)
            num_seeds = len(set([name.split('_')[0] for name in short_models_names])) -1 * with_base_model
            save_matrix_file_name = f'{s_matrix_type}_matrix_{s_models_names}_seeds_{num_seeds}.csv'
            save_matrix(matrix, save_matrix_file_name, models_names)

    if clearml_task:
        clearml_task.upload_artifact('matrix', matrix)
        clearml_task.upload_artifact('models names', models_names)
        if save_matrix_flag:
            clearml_task.upload_artifact('matrix save file path', save_matrix_file_name)

    if to_cluster:
        print('\nCluster models')
        is_sim = True if (matrix_type  in ['cosine similarity', 'mutual information', 'cka']) else False
        analyze_matrix(matrix, short_models_names, is_sim)
        analyze_clustering(matrix, is_sim, short_models_names, dim_reduction_method, matrix_type, with_base_model=with_base_model)

    if return_matrix:
        try:
            return matrix, models_names
        except Exception as e:
            print('missing one of the return objects: matrix, save_matrix_file_name, models_names in run_analysis()')
            raise e


def run_find_num_bins(bins_range, depended_expr_dirs, base_model_name, desired_datasets, is_classification, model_paths,
                      weights_reference_point, weights_normalization_type, norm_thr, dataset_group='GLUE_AND_SUPER_GLUE'):

    models_weights, models_names = get_models_weights(depended_expr_dirs, base_model_name, desired_datasets,
                                                      is_classification, model_paths)
    data_num2name = {name.split('_')[-1]: DATA_NUM2NAME[dataset_group][name.split('_')[-1]] for name in models_names}
    short_models_names = [s.replace('seed_', 's').replace('data', 'd') for s in models_names]
    short_models_names = [s.split('d_')[0] + data_num2name[s.split('d_')[-1]] for s in short_models_names]
    flat_models_weights_diff = analyze_weights_n_norms(models_weights, short_models_names, base_model_name,
                                                       weights_reference_point, weights_normalization_type, norm_thr)

    best_num_bins = find_num_bins(flat_models_weights_diff, short_models_names, bins_range)
    print(best_num_bins)

if __name__ == '__main__':

    # get parser args
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--is_local_run", default=False, type=bool)
    args, _ = parser.parse_known_args()

    if args.debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        debug_ip = None  # '9.148.203.20'  # os.environ.get('SSH_CONNECTION', None) # this is the default value for debug_ip
        debug_port = 12345  # this is the default value for debug_port
        set_remote_debugger(debug_ip, debug_port)

    # Define depended experiment dirs
    if args.is_local_run:
        glue_template_expr_dir = '/Users/almoggueta/data/fusion/outputs_almog/TrainIterateOverDataset_'
    else:
        glue_template_expr_dir = '/dccstor/fuse/outputs_almog/TrainIterateOverDataset_'

    seeds = range(1, 6) # range(1, 6)    1-5 is for roberta with weight_decay=0.01,  6-10 for roberta weight_decay=0.0
    glue_all_seeds_expr_dir = [glue_template_expr_dir + str(seed) for seed in seeds]
    depended_expr_dirs = glue_all_seeds_expr_dir

    # get base model
    base_model_name = 'roberta-base'
    base_model = load_model(
        ModelLoadingInfo(classification=True, name=base_model_name, from_tf=False, tokenizer_name=base_model_name))

    model_paths = []

    desired_datasets = ['mnli', 'sst2']  # , 'cola', 'qqp']

    matrix_type = 'cosine similarity' # 'euclidean distance' #  # 'mutual information'
    # run analysis
    # analyze_models(depended_expr_dirs,
    #                model_paths, base_model, weights_reference_point='base_model', weights_normalization_type=None,
    #                is_classification=True, desired_datasets=desired_datasets)


    run_analysis_kwargs = {
                            'depended_expr_dirs': depended_expr_dirs,
                            'base_model_name': base_model,
                            'desired_datasets': desired_datasets,
                            'weights_reference_point': 'avg weights', #'base_model',
                            'weights_normalization_type': None,
                            'norm_thr': 0,
                            'matrix_type': matrix_type,
                            'dim_reduction_method': 'tsne',
                            'save_matrix_flag': False,
                            'with_base_model': False,
                            'model_paths': model_paths,
                            'is_classification': True,
                            'to_cluster': True,
                            'dataset_group': 'GLUE_AND_SUPER_GLUE',
                            'matrix_file_name': 'analysis_utils/saved_matrices/sim_matrix_glue_super_glue_70.csv', #None, #'analysis_utils/saved_matrices/euc_dist_matrix_glue_super_glue_70.csv', #None, # 'analysis_utils/saved_matrices/mi_matrix_mnli_sst2.csv',
                            }
    matrix_name = run_analysis_kwargs['matrix_file_name'].split('/')[-1] if run_analysis_kwargs['matrix_file_name'] is not None else desired_datasets if desired_datasets is not None else 'glue super glue'
    task = Task.init(project_name='fusion', tags=[matrix_type, 'knn'],
                     task_name=f'analysis_clustering_{matrix_type}_on_{matrix_name}')
    task.upload_artifact('run_analysis_kwargs', run_analysis_kwargs)
    # task = None
    run_analysis(**run_analysis_kwargs, clearml_task=task)


    # MUTUAL INFORMATION FIND NUM BINS EXP
    # run_find_bins_kwargs = {}
    # for arg in ['depended_expr_dirs', 'base_model_name', 'desired_datasets', 'is_classification', 'model_paths',
    #                   'weights_reference_point', 'weights_normalization_type', 'norm_thr']:
    #     run_find_bins_kwargs[arg] = run_analysis_kwargs[arg]
    # run_find_bins_kwargs['bins_range'] = range(100, 20000, int((20000-100)/15))
    # run_find_num_bins(**run_find_bins_kwargs)

