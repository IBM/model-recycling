# (C) Copyright IBM Corp. 2022.

from dataclasses import dataclass
from itertools import repeat

import os
from typing import Union, List

from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, PretrainedConfig


@dataclass
class ModelLoadingInfo:
    name: str
    tokenizer_name: str
    classification: bool
    model_class: object = None
    from_tf: bool = False
    from_flax: bool = False
    config: Union[PretrainedConfig, str, os.PathLike] = None

    def __post_init__(self):
        if self.model_class is None:
            self.model_class = AutoModelForSequenceClassification if self.classification else AutoModelForSeq2SeqLM


def load_model(model_loading_info: ModelLoadingInfo):
    model = model_loading_info.model_class.from_pretrained(model_loading_info.name, config=model_loading_info.config,
                                                           from_tf=model_loading_info.from_tf,
                                                           from_flax=model_loading_info.from_flax,
                                                           ignore_mismatched_sizes=True)
    if not model_loading_info.classification and model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    return model


def average_models(model_loading_infos: List[ModelLoadingInfo], config=None):
    """
    gets a list of models and returns the average of their weights
    :param config:
    :param model_loading_infos: List of model
    :return:
    """
    sum_weights = []
    if config and "average_weights" in config:
        average_weights = config["average_weights"]
        assert len(average_weights) == len(model_loading_infos), \
            f"Number of average weights ({len(average_weights)}) " \
            f"does not match number of models ({len(model_loading_infos)})"
    else:
        average_weights = repeat(1)
    for i, (model_loading_info, average_weight) in enumerate(zip(model_loading_infos, average_weights)):
        # try:
        model = load_model(model_loading_info).base_model
        weights = model.parameters()
        for weight_num, weight in enumerate(weights):
            weight = weight.detach() / len(model_loading_infos)
            if i == 0:
                sum_weights.append(weight)
            else:
                try:
                    sum_weights[weight_num] += weight * average_weight
                except RuntimeError as e:
                    raise ValueError(
                        f"Models do not fit for fusing: {model_loading_info} "
                        f"\n base model for fusing: {model_loading_infos[0]}")
        del model
    return sum_weights


def fuse_models(model, model_loading_infos, config=None):
    model = model.base_model
    state_dict = model.state_dict()  # replace the original state_dict - model.state_dict()
    for (name, weight), new_weight in zip(model.named_parameters(),
                                          average_models(model_loading_infos, config=config)):
        state_dict[name] = new_weight
    model.load_state_dict(state_dict=state_dict)
    return model


if __name__ == '__main__':
    bert = ModelLoadingInfo(name="bert-base-uncased", tokenizer_name="bert-base-uncased", classification=True)
    models_to_fuse = [bert, bert]
    base_model = load_model(bert)
    # fuse bert with itself, which should return the same original weights
    fused = fuse_models(base_model, models_to_fuse)

    # a more interesting examples
    roberta = ModelLoadingInfo(name="roberta-base", tokenizer_name="roberta-base", classification=True)
    models_to_fuse = ['cross-encoder/stsb-roberta-base', 'textattack/roberta-base-STS-B']
    models_to_fuse = [ModelLoadingInfo(name=model, tokenizer_name=model, classification=True) for model in models_to_fuse]
    base_model = load_model(roberta)
    fused = fuse_models(base_model, models_to_fuse)
