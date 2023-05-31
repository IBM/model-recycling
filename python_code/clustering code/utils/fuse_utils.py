from typing import List
from copy import deepcopy

def fuse_models(model, model_loading_infos, config=None):
    model = model.base_model
    models_generator = models_loader(model_loading_infos)
    new_weights = average_models(models_generator, config=config)
    model = plant_weights(model, new_weights)
    return model

def model_diff(base_weights: List, new_weights: List) -> List:
    weights_diff = []  # num of models x model layers
    for j, (model_weight, base_model_weight) in enumerate(zip(new_weights, base_weights)):
        weights_diff.append(model_weight.detach() - base_model_weight.detach())
    return weights_diff


def model_norm(weights: List) -> float:
    import torch
    flatten_weights = torch.cat([torch.flatten(weight) for weight in weights])
    norm = torch.linalg.norm(flatten_weights)
    return norm


def get_model_from_model_direction_and_size(model, direction_model_loading_info, config):
    norm_scale = config["norm_scale"] # size
    state_dict = model.state_dict()  # replace the original state_dict - model.state_dict()
    direction_model = average_models(direction_model_loading_info)
    norm = model_norm(direction_model)
    for (name, weight), new_weight in zip(model.named_parameters(), direction_model):
        state_dict[name] += new_weight * norm_scale / norm
    model.load_state_dict(state_dict=state_dict)


def predefined_global_norm_fuse_models(model, model_loading_infos, config):
    norm_scale = config["norm_scale"]
    state_dict = model.state_dict()  # replace the original state_dict - model.state_dict()
    new_model = average_models(model_loading_infos)
    diff = model_diff(model.parameters(), new_model)
    norm = model_norm(diff)
    for (name, weight), new_weight in zip(model.named_parameters(), diff):
        state_dict[name] += new_weight * norm_scale / norm
    model.load_state_dict(state_dict=state_dict)


def predefined_local_norm_fuse_models(model, model_loading_infos, config):
    norm_scale = config["norm_scale"]
    state_dict = model.state_dict()  # replace the original state_dict - model.state_dict()
    for (name, weight), diff_weight in zip(model.named_parameters(),
                                           average_normalized_diff(model, model_loading_infos, norm_scale)):
        state_dict[name] += diff_weight
    model.load_state_dict(state_dict=state_dict)


def average_normalized_diff(model, model_loading_infos: List, norm_scale=None) -> List:
    """
    gets a list of models and returns the average of their weights
    :param norm_scale:
    :param model:
    :param model_loading_infos: List of model
    :return:
    """
    from utils.model_loading_utils import load_model
    
    sum_diffs = []

    for i, model_loading_info in enumerate(model_loading_infos):
        new_model = load_model(model_loading_info)
        diff = model_diff(model.parameters(), new_model)
        norm = model_norm(diff)
        for diff_weight_num, diff_weight in enumerate(diff):
            normalize = norm_scale / norm if norm_scale else 1
            diff_weight = diff_weight.detach() / len(model_loading_infos) * normalize
            if i == 0:
                sum_diffs.append(diff_weight)
            else:
                try:
                    sum_diffs[diff_weight_num] += diff_weight
                except RuntimeError as e:
                    raise ValueError(
                        f"Models do not fit for fusing: {model_loading_info} "
                        f"\n base model for fusing: {model_loading_infos[0]}")
        del new_model
    return sum_diffs


def models_loader(model_loading_infos):
    from utils.model_loading_utils import load_model
    for model_loading_info in model_loading_infos:
        yield load_model(model_loading_info).base_model

def average_models(models_generator, config=None, return_model=False):
    """
    gets a list of models and returns the average of their weights
    :param config:
    :param model_loading_infos: List of model
    :return:
    """
    sum_weights = None
    
    for i, model in enumerate(models_generator):
        
        if config is not None and "average_weights" in config:
            average_weight = config["average_weights"][i]
        else:
            average_weight = 1
        
        weights = model.parameters()
        
        if sum_weights is None:
            sum_weights = [weight * average_weight for weight in weights]
        else:
            for idx, weight in enumerate(weights):
                sum_weights[idx] += (weight * average_weight)
                        
    if config is None or "average_weights" not in config:
        sum_weights = [weight / (i + 1) for weight in sum_weights]
    else:
        assert len(config["average_weights"]) == (i + 1), \
            f"Number of average weights ({len(config['average_weights'])}) " \
            f"does not match number of models ({i + 1})"
    
    if return_model:
        return plant_weights(deepcopy(model), sum_weights)
        
    return sum_weights

def plant_weights(model, weights):
    state_dict = model.state_dict()
    for (name, weight), new_weight in zip(model.named_parameters(), weights):
        state_dict[name] = new_weight
    model.load_state_dict(state_dict=state_dict)
    return model
