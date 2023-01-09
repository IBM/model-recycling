---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: deberta-v3-base__sst2__all-train
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base__sst2__all-train

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6964
- Accuracy: 0.49

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
- num_epochs: 50
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 7    | 0.6964          | 0.49     |
| No log        | 2.0   | 14   | 0.7010          | 0.49     |
| No log        | 3.0   | 21   | 0.7031          | 0.49     |
| No log        | 4.0   | 28   | 0.7054          | 0.49     |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.2+cu102
- Datasets 1.18.2
- Tokenizers 0.10.3
