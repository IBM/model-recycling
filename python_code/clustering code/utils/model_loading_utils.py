import os
from dataclasses import dataclass
from typing import Union

from transformers import AutoTokenizer, PretrainedConfig, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from utils.DatasetEncoder import DatasetEncoder, DatasetEncoderForClassifier, DatasetEncoderForRegresor
from utils.dataset_info import Task, DatasetInfo


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
    # TODO is needed:?
    # model.resize_token_embeddings(len(tokenizer))
    if not model_loading_info.classification and model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    return model


def load_encoder(model_info: ModelLoadingInfo, dataset_info: DatasetInfo, dataset=None, labels_set=None, use_prefix=False):
    tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer_name, use_fast=True)
    post_input = f"<{dataset_info.name}"
    prefix = ""
    if use_prefix:
        prefix = f"{dataset_info.name}>"
    if dataset_info.name.endswith("_mapped"):
        post_input = ""
    if model_info.classification:
        if dataset_info.task == Task.REGRESSION:
            encoder = DatasetEncoderForRegresor(tokenizer=tokenizer, extract_labels=dataset_info.extract_label,
                                                post_input=post_input, sentence_names=dataset_info.sentence_names,
                                                prefix=prefix
                                                )
        else:
            if labels_set is None:
                labels_set = sorted(set(dataset_info.extract_label(dataset)))
            encoder = DatasetEncoderForClassifier(tokenizer=tokenizer, extract_labels=dataset_info.extract_label,
                                                  post_input=post_input, sentence_names=dataset_info.sentence_names,
                                                  labels_set=labels_set,
                                                  prefix=prefix
                                                  )
    else:
        encoder = DatasetEncoder(tokenizer=tokenizer, extract_labels=dataset_info.extract_label,
                                 post_input=post_input, sentence_names=dataset_info.sentence_names, prefix=prefix)
    return encoder


def local_model_loadable(model_info: ModelLoadingInfo):
    path = model_info.name
    needed_files = ['config.json',
                    'pytorch_model.bin',
                    ]
    return all([os.path.isfile(os.path.join(path, needed_file)) for needed_file in needed_files])
