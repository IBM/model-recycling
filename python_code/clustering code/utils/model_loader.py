import os
from typing import List, Tuple

import torch
from transformers import AutoConfig

from utils.files_utils import get_results_model_dir_for_iteration, get_root_model_dir
from utils.fuse_utils import fuse_models
from utils.logger_util import create_logger
from utils.model_loading_utils import ModelLoadingInfo, load_model, local_model_loadable
from peft import get_peft_model, LoraConfig, TaskType

class ModelLoader:
    def __init__(self, model_name='google/t5-v1_1-small', tokenizer=None, classification=False, num_labels=None,
                 from_tf=False, from_flax=False,
                 logger=None):
        self.model_name = model_name
        self.tokenizer_name = tokenizer if tokenizer is not None else self.model_name
        if self.tokenizer_name and 'google/t5-v1_1' in self.tokenizer_name:
            self.tokenizer_name = self.tokenizer_name.replace('google/t5-v1_1', 't5')
        self.classification = classification
        self.num_labels = num_labels
        self.from_tf = from_tf
        self.from_flax = from_flax
        self.logger = logger
        if logger is None:
            self.logger = create_logger(__name__ + f".{self.__class__.__name__}")
        self.logger.info(f"Loading {model_name}")

    def get_model_loading_info(self):
        if self.classification:
            problem_type = 'single_label_classification'
            if self.num_labels == 1:
                problem_type = 'regression'
            config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, problem_type=problem_type)
        else:
            config = None
        loading_info = ModelLoadingInfo(classification=self.classification,
                                        name=self.model_name, from_tf=self.from_tf, from_flax=self.from_flax,
                                        tokenizer_name=self.tokenizer_name, config=config)
        return loading_info

    def set_num_labels(self, num_labels):
        self.num_labels = num_labels

    def get_model(self) -> Tuple[torch.nn.Module, ModelLoadingInfo]:
        loading_info = self.get_model_loading_info()
        model = load_model(loading_info)
        return model, loading_info


class LoraModelLoader(ModelLoader):
    def get_model(self):
        model, loading_info = super().get_model()
        model.enable_input_require_grads()
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS if self.classification else TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16, # lora_alpha=8 till 23/5
            lora_dropout=0 # lora_dropout=0.05 till 23/5
        )
        # task_type = TaskType.SEQ_2_SEQ_LM, inference_mode = False, r = 8, lora_alpha = 328
        # lora_dropout = 0.1
        model = get_peft_model(model, peft_config)
        return model, loading_info


class LinearProbingModelLoader(ModelLoader):
    def get_model(self):
        model, loading_info = super().get_model()
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
        return model, loading_info


class BitFitMiniModelLoader(ModelLoader):
    # https://arxiv.org/pdf/2106.10199.pdf
    # freeze entire base model (encoder) except for 2 bias terms
    def get_model(self):
        model, loading_info = super().get_model()
        trainable_biases = ['attention.self.query.bias', 'intermediate.dense.bias']
        for name, param in model.base_model.named_parameters():
            for trainable_param in trainable_biases:
                if trainable_param not in name:
                    param.requires_grad = False
        return model, loading_info

class LinearProbeT5ModelLoader(ModelLoader):
    def get_model(self):
        model, loading_info = super().get_model()
        trainable_biases = 'lm_head.weight'
        # the lm_head is not part of the base model, so it remains unfrozen.
        for name, param in model.base_model.named_parameters():
            if trainable_biases not in name:
                param.requires_grad = False
        return model, loading_info

class FuseModelLoader(ModelLoader):
    """Fuses models locally trained and found in the depended experiment dir into model_name"""

    def __init__(self, depended_expr_dir, logger, model_names_or_paths="", tokenizer=None, classification=False,
                 model_name='google/t5-v1_1-small', from_tf=False, from_flax=False, fuse_config=None,
                 fuse_models_func=fuse_models, selected_models_indices=None):
        """

        :param depended_expr_dir:
        :param logger:
        :param model_names_or_paths: models to fuse
        :param model_name: basic model which all of the rest should be similar to
        :param from_tf:
        """
        super().__init__(model_name=model_name, tokenizer=tokenizer, classification=classification, from_tf=from_tf,
                         from_flax=from_flax)
        assert not (
                from_tf or from_flax), "Behaviour undefined. (without this assert all fused models would be loaded with flax/tf) "
        if depended_expr_dir == 'no_dependent':
            self.depended_expr_dir = ""
        else:
            self.depended_expr_dir = depended_expr_dir.split(',')
        self.logger = logger
        self.selected_models_indices = selected_models_indices
        self.model_names_or_paths = [path for path in model_names_or_paths.split(',') if path]
        self.fuse_models_func = fuse_models_func
        self.fuse_config = fuse_config

    def get_model(self):
        model, loading_info = super().get_model()
        model_loading_infos = self.get_model_loading_infos()
        self.logger.info(f"Fusing {len(model_loading_infos)} models...")
        self.fuse_models_func(model, model_loading_infos, self.fuse_config)
        return model, loading_info

    def get_model_loading_infos(self) -> List[ModelLoadingInfo]:
        model_loading_infos = []
        for dirpath in self.depended_expr_dir:
            dir_model_loading_infos = self.get_models_from_dir(dirpath)
            self.logger.info(f"Adding {len(dir_model_loading_infos)} models from {dirpath} for fusing")
            model_loading_infos += dir_model_loading_infos
        for path in self.model_names_or_paths:
            model_loading_infos.append(self.path_to_info(path))
        self.logger.info(f"Adding {len(self.model_names_or_paths)} models specified explicitly for fusing")
        return model_loading_infos

    def path_to_info(self, name_or_path):
        return ModelLoadingInfo(classification=self.classification, name=name_or_path, from_tf=self.from_tf,
                                from_flax=self.from_flax,
                                tokenizer_name=self.tokenizer_name)

    def get_models_from_dir(self, dirpath):
        model_loading_infos = []
        if self.selected_models_indices:
            dirs = [get_results_model_dir_for_iteration(dirpath, i) for i in self.selected_models_indices]
        else:
            root, dirs, _ = next(os.walk(get_root_model_dir(dirpath)))
            dirs = [os.path.join(root, model_dir_name) for model_dir_name in dirs]
        for model_path in dirs:
            self.logger.info(f"Chosen model path for fusing: {model_path}")
            model_loading_infos.append(self.path_to_info(model_path))
        return model_loading_infos


class FuseRecursiveModelLoader(FuseModelLoader):
    """
    Looks for all models under depended expr dirs recursively (except fused models) and fuses them
    """

    def __init__(self, depended_expr_dir, logger,
                 model_name='google/t5-v1_1-small', tokenizer=None, classification=False, from_tf=False,
                 from_flax=False):
        super().__init__(model_name=model_name, classification=classification, tokenizer=tokenizer,
                         depended_expr_dir=depended_expr_dir, from_tf=from_tf, from_flax=from_flax, logger=logger,
                         )

    def get_model_loading_infos(self) -> List[ModelLoadingInfo]:
        model_loading_infos = []
        for base_dir in self.depended_expr_dir:
            for root, dirs, _ in os.walk(base_dir):
                for dirname in dirs:
                    dirpath = os.path.join(root, dirname)
                    model_dir_path = get_root_model_dir(dirpath)
                    if "fuse" not in dirname.lower() and os.path.isdir(model_dir_path):
                        all_dir_model_loading_infos = self.get_models_from_dir(dirpath)
                        dir_model_loading_infos = []
                        # Make sure all models can be loaded, remove failed ones
                        for model_info in all_dir_model_loading_infos:
                            if local_model_loadable(model_info):
                                dir_model_loading_infos.append(model_info)
                            else:
                                self.logger.warning(
                                    f"Skipping invalid model path, model may be corrupted: {model_info.name}")
                        self.logger.info(f"Adding {len(dir_model_loading_infos)} models from {dirpath} for fusing")
                        model_loading_infos += dir_model_loading_infos

        return model_loading_infos


class LooFuseModelLoader(FuseModelLoader):
    def __init__(self, depended_expr_dir, number_of_iterations, experiment_num, logger, classification=False,
                 model_name='google/t5-v1_1-small', tokenizer=None, from_tf=False, from_flax=False, fuse_config=None,
                 fuse_models_func=fuse_models):
        super().__init__(depended_expr_dir=depended_expr_dir, classification=classification, from_tf=from_tf,
                         from_flax=from_flax,
                         logger=logger,
                         model_name=model_name, tokenizer=tokenizer, fuse_config=fuse_config,
                         fuse_models_func=fuse_models_func)
        self.indices = tuple([i for i in range(0, number_of_iterations) if i != experiment_num])

        assert len(self.depended_expr_dir) == 1, "LOO only allows one depended dir at the time"
        self.depended_expr_dir = self.depended_expr_dir[0]

    def get_model_loading_infos(self) -> List[ModelLoadingInfo]:
        self.logger.info(f"Running on indices: {self.indices}")
        model_loading_infos = []
        for i in self.indices:
            model_path = get_results_model_dir_for_iteration(self.depended_expr_dir, i)
            model_loading_infos.append(
                ModelLoadingInfo(classification=self.classification, name=model_path, from_tf=self.from_tf,
                                 from_flax=self.from_flax,
                                 tokenizer_name=self.tokenizer_name))
        return model_loading_infos


class RandomModelLoader(ModelLoader):
    def __init__(self, save_path, logger, classification=False,
                 model_name='google/t5-v1_1-small', tokenizer=None, from_tf=False, from_flax=False):
        """

        :param depended_expr_dir:
        :param logger:
        :param model_names_or_paths: models to fuse
        :param model_name: basic model which all of the rest should be similar to
        :param from_tf:
        """
        super().__init__(model_name=model_name, tokenizer=tokenizer, classification=classification, from_tf=from_tf,
                         from_flax=from_flax,
                         logger=logger)
        self.save_path = save_path

    def get_model(self):
        model, loading_info = super(RandomModelLoader, self).get_model()
        model = loading_info.model_class.from_config(model.config)
        model.save_pretrained(self.save_path)
        return model, loading_info

