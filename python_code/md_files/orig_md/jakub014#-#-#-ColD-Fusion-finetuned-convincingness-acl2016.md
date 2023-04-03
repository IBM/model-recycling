---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: ColD-Fusion-finetuned-convincingness-acl2016
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ColD-Fusion-finetuned-convincingness-acl2016

This model is a fine-tuned version of [ibm/ColD-Fusion](https://huggingface.co/ibm/ColD-Fusion) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4112
- Accuracy: 0.9275

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.4318        | 1.0   | 583  | 0.2267          | 0.9047   |
| 0.2063        | 2.0   | 1166 | 0.1945          | 0.9142   |
| 0.1647        | 3.0   | 1749 | 0.3107          | 0.9155   |
| 0.1179        | 4.0   | 2332 | 0.3730          | 0.9215   |
| 0.0669        | 5.0   | 2915 | 0.4112          | 0.9275   |


### Framework versions

- Transformers 4.27.2
- Pytorch 1.13.1+cu116
- Datasets 2.10.1
- Tokenizers 0.13.2
