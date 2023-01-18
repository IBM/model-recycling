---
license: mit
tags:
- generated_from_trainer
datasets:
- squad_v2
model-index:
- name: muppet-roberta-base-finetuned-squad
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# muppet-roberta-base-finetuned-squad

This model is a fine-tuned version of [facebook/muppet-roberta-base](https://huggingface.co/facebook/muppet-roberta-base) on the squad_v2 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9017

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
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 0.7007        | 1.0   | 8239  | 0.7905          |
| 0.4719        | 2.0   | 16478 | 0.9017          |


### Framework versions

- Transformers 4.20.0
- Pytorch 1.11.0+cu113
- Datasets 2.3.2
- Tokenizers 0.12.1
