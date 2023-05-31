import csv
import json
import os
from typing import List

import torch

from run_configuration.inter_paper.cross_datasets_largre_matrix import load_dataset_by_dataset_name
from utils.files_utils import get_root_model_dir
from utils.model_loading_utils import ModelLoadingInfo, load_model


def normalize_weights(model_weights: List[torch.TensorType], normalization_type: str):
    """
    Normalize models' weights according to the given normalization type- no normalization, normalize by model's weight,
    or normalize all model's weights together.
    Return output at the same shape as the input
    """
    if normalization_type is None or normalization_type == 'none':
        return model_weights

    if normalization_type == 'layers':
        normalized_model_weights = []
        for model_w in model_weights:
            norm = torch.linalg.norm(model_w)
            normalized_model_weights.append(model_w / norm)
        return normalized_model_weights

    elif normalization_type == 'all':
        norm = torch.linalg.norm(flatten_model(model_weights))
        normalized_model = [w / norm for w in model_weights]
        return normalized_model

    else:
        print(f"Unknown model's weights normalization type: {normalization_type}")
        return -1


def load_matrix(file_name: str):
    """
    Load similarity matrix or distance matrix
    """
    # possible file names: 'sim_matrix_glue_super_glue_70.csv', 'euc_dist_matrix_glue_super_glue_70.csv'

    try:
        with open(file_name, 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = csv.reader(read_obj, quoting=csv.QUOTE_NONNUMERIC)
            # Pass reader object to list() to get a list of lists
            matrix = list(csv_reader)
            header = matrix[0][0].split(', ')
            matrix = matrix[1:]

            print(f'Loading matrix from path: {file_name}')
            print(f'Cols names: {header}')
            print(f'Matrix num rows: {len(matrix)}, num cols: {len(matrix[0])}')

            return matrix, header

    except:
        pass

    with open(file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)
        matrix = []
        for row in csv_reader:
            row = [float(v) for v in row]
            matrix.append(row)

        return matrix, header


def save_matrix(matrix: [[]], file_name: str, cols_names: list):
    """
    Save similarity matrix or distance matrix given as list of lists, saved as csv
    """
    if type(matrix[1][0]) == str:
        rows = []
        rows.append(matrix[0])
        for r in matrix[1:]:
            curr_row = []
            for num in r.split():
                curr_row.append(num)
            rows.append(curr_row)

        if len(rows) != len(rows[0]):
            print('Error: wrong dimension of given matrix!')
        matrix = rows

    full_path = os.path.join('analysis_utils/saved_matrices', file_name)
    with open(f"{full_path}", "w") as f:
        wr = csv.writer(f)
        wr.writerow(cols_names)
        wr.writerows(matrix)
    print(f'Saved matrix in path: {full_path}')


def get_dataset_sizes(datasets_names: list) -> dict:
    dataname2size = {}
    for data_name in datasets_names:
        dataset, data_info = load_dataset_by_dataset_name(data_name=data_name, split="train")
        dataname2size[data_info.name] = dataset.num_rows
    return dataname2size


def get_model_loading_infos_from_dirs(depended_expr_dirs, tokenizer_name, desired_datasets: List[str],
                                      classification=False, model_paths=None):
    if model_paths is None:
        model_paths = []
    model_loading_infos = []
    datasets_names = []
    for depended_expr_dir in depended_expr_dirs:
        root, dirs, _ = next(os.walk(get_root_model_dir(depended_expr_dir)))
        for model_dir_name in dirs:
            model_path = os.path.join(root, model_dir_name)
            with open(os.path.join(model_path, 'results.json')) as f:
                result = json.load(f)
            dataset_name = next(iter(result['results'].keys()))
            if depended_expr_dir.split('/')[-1].split('_')[0]=='WeightedFuseAndTrainIterateOverChosenIndices':
                dataset_name = 'avg'
            if desired_datasets is None or dataset_name in desired_datasets:
                datasets_names.append(dataset_name)
                model_paths.append(model_path)

    for model_path in model_paths:
        model_loading_infos.append(
            ModelLoadingInfo(classification=classification, name=model_path, from_tf=False,
                             tokenizer_name=tokenizer_name))
    return model_loading_infos, datasets_names


def extract_models_weights(model_loading_infos):
    models_weights = []
    names = None
    for i, model_loading_info in enumerate(model_loading_infos):
        model = load_model(model_loading_info)
        names, weights = list(zip(*model.base_model.named_parameters()))
        models_weights.append(weights)
        del model
    return names, models_weights


def flatten_model(weights):
    return torch.cat([torch.flatten(layer) for layer in weights])


def get_models_weights(depended_expr_dirs, base_model_name, desired_datasets, is_classification, model_paths):
    model_loading_infos, datasets_names = get_model_loading_infos_from_dirs(depended_expr_dirs, base_model_name,
                                                                            desired_datasets, is_classification,
                                                                            model_paths)

    print('\nExtracting models weights')
    weights_names, models_weights = extract_models_weights(model_loading_infos)
    datasets_and_seeds_str = [
        model_loading_infos[i].name.split('Over')[-1].replace('/models/', '_').replace('Dataset', 'seed').replace(
            'train', 'data') for i in range(len(model_loading_infos))] # 'seed_{i}_data_{j}
    # following condition is to correct datasets names for glue_and_fused_cross_lp- single seed for all glue & super glue tasks
    if (datasets_and_seeds_str[0].split('_data_')[-1]==datasets_and_seeds_str[-1].split('_data_')[-1]) and (datasets_names[0]!=datasets_names[-1]):
        datasets_and_seeds_str = [f'seed_0_data_{d}' for d in datasets_names]
    return models_weights, datasets_and_seeds_str
