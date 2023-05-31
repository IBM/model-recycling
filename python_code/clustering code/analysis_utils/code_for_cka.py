import argparse
import os
from typing import List

import numpy as np
import torch
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from tqdm import tqdm

from analysis_utils import calcs as calcs
from analysis_utils.analyze_utils import analyze_clustering, analyze_matrix
from analysis_utils.calcs import create_matrix
from analysis_utils.constants import DATA_NUM2NAME
from analysis_utils.data_utils import get_model_loading_infos_from_dirs, save_matrix
from utils.files_utils import get_root_model_dir
from utils.model_loading_utils import load_model


def calculate_CKA_for_two_matrices(activationA, activationB, debiased=False):
    # code from: https://towardsdatascience.com/do-different-neural-networks-learn-the-same-things-ac215f2103c3
    '''Takes two activations A and B and computes the linear CKA to measure their similarity'''

    # calculate the CKA score
    cka_score = calcs.feature_space_linear_cka(activationA, activationB, debiased)

    # remove space from memory
    del activationA
    del activationB

    return cka_score


def get_all_layers_activations(model, data, desired_layers: List[str]):
    # code from: https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter

    mid_getter = MidGetter(model, return_layers=desired_layers, keep_output=True)
    mid_outputs, model_output = mid_getter(data)
    """ 
    print(model_output)
    >> tensor([[0.3219]], grad_fn=<AddmmBackward>)
    print(mid_outputs)
    >> OrderedDict([('fc2', tensor([[-1.5125,  0.9334]], grad_fn=<AddmmBackward>)),
      ('interaction', tensor([[-0.0687, -0.1462]], grad_fn=<MulBackward0>)),
      ('nested', tensor([[-0.1697,  0.1432,  0.2959]], grad_fn=<AddmmBackward>))])

    # model_output is None if keep_ouput is False
    # if keep_output is True the model_output contains the final model's output
    """

    return mid_outputs


def flatten_layers_activations(activations):
    flatten = []
    for acts in activations: # acts = tensor [num_layers, num_samples, hid_size]
        num_layers, num_samples, hid_size = acts.shape
        flat_acts = torch.flatten(acts, (-1, hid_size)) # [num_layers*num_samples, hid_size]
        flatten.append(flat_acts)

    return flatten




# ------ USED FUNCTIONS:


# def get_activations_from_dirs(depended_expr_dirs, desired_datasets):
#     activations = []
#     models_names = []
#     for depended_expr_dir in depended_expr_dirs: # e.g. '/dccstor/fuse/outputs_almog/TrainIterateOverDataset_1/'
#         root, dirs, _ = next(os.walk(get_root_model_dir(depended_expr_dir))) # root: '/dccstor/fuse/outputs_almog/TrainIterateOverDataset_1/models', dirs: [train_0,...,train_13]
#         for model_dir_name in dirs: # 'train_i'
#             model_path = os.path.join(root, model_dir_name) # '/dccstor/fuse/outputs_almog/TrainIterateOverDataset_1/models/train_i'
#
#             model_name = DATA_NUM2NAME[model_dir_name.split("_")[-1]]
#             acts_torch = torch.load(os.path.join(model_path, f'hidden_states_{COMPLETE}')) # [num_layers, num_samples, hid_size]  should be the dataset name i ran infer on, not finetuned on
#             if desired_datasets is None or model_name in desired_datasets:
#                 models_names.append(model_name)
#                 activations.append(acts_torch)
#
#     #  - somehow have seed num in addition to model name
#     # maybe use: datasets_and_seeds_str = [
#     #         model_loading_infos[i].name.split('Over')[-1].replace('/models/', '_').replace('Dataset', 'seed').replace(
#     #             'train', 'data') for i in range(len(model_loading_infos))]
#
#     return activations, models_names


def get_models_activations(depended_expr_dirs, desired_datasets):
    activations, models_names = get_activations_from_dirs(depended_expr_dirs, desired_datasets)
    flatten_models_activations = flatten_layers_activations(activations)
    return flatten_models_activations, models_names


def run_cka_analysis(depended_expr_dirs, save_matrix_flag: bool, desired_datasets, dim_reduction_method, debiased:bool, with_base_model):
    flatten_models_activations, models_names = get_models_activations(depended_expr_dirs, desired_datasets)
    data_num2name = {name.split('_')[-1]: DATA_NUM2NAME['GLUE_AND_SUPER_GLUE'][name.split('_')[-1]] for name in models_names}
    short_models_names = [s.replace('seed_', 's').replace('data', 'd') for s in models_names]
    short_models_names = [s.split('d_')[0] + data_num2name[s.split('d_')[-1]] for s in short_models_names]
    matrix = create_matrix(flatten_models_activations, 'cka', debiased)
    matrix = matrix[0]
    matrix = [[float(d) for d in data] for data in matrix]
    if save_matrix_flag:
        s_matrix_type = 'cka'
        s_models_names = 'glue_super_glue_70' if not desired_datasets else '_'.join(desired_datasets)
        save_matrix_file_name = f'{s_matrix_type}_matrix_{s_models_names}.csv'
        save_matrix(matrix, save_matrix_file_name, models_names)

    print('\nCluster models')
    is_sim = True
    analyze_matrix(matrix, short_models_names, is_sim)
    analyze_clustering(matrix, is_sim, short_models_names, dim_reduction_method, 'cka', with_base_model=with_base_model)





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
        glue_template_expr_dir = '/Users/almoggueta/data/fusion/outputs_almog/InferAndGetActivationsOverDataset_'
    else:
        glue_template_expr_dir = '/dccstor/fuse/outputs_almog/InferAndGetActivationsOverDataset_'

    seeds = range(1,6)
    glue_all_seeds_expr_dir = [glue_template_expr_dir + str(seed) for seed in seeds]
    depended_expr_dirs = glue_all_seeds_expr_dir

    desired_datasets = ['mnli', 'sst2']  # , 'cola', 'qqp']

    cka_analysis_kwargs = {'depended_expr_dirs': depended_expr_dirs,
                           'desired_datasets': desired_datasets,
                           'dim_reduction_method': 'tsne',
                           'save_matrix_flag': False,
                           'with_base_model': True,
                           'debiased': True} # If the number of examples is small, it might help to compute a "debiased" form of CKA. The resulting estimator of CKA is still generally biased, but the bias is reduced.

    run_cka_analysis(**cka_analysis_kwargs)