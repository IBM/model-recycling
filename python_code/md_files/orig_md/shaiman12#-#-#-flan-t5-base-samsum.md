---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- samsum
metrics:
- rouge
model-index:
- name: flan-t5-base-samsum
  results:
  - task:
      name: Sequence-to-sequence Language Modeling
      type: text2text-generation
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
      args: samsum
    metrics:
    - name: Rouge1
      type: rouge
      value: 47.5929
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-samsum

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3776
- Rouge1: 47.5929
- Rouge2: 23.8272
- Rougel: 40.1493
- Rougelsum: 43.7798
- Gen Len: 17.2503

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 1.4416        | 1.0   | 1842 | 1.3837          | 46.6013 | 23.125  | 39.4894 | 42.9943   | 17.0684 |
| 1.3581        | 2.0   | 3684 | 1.3730          | 47.3142 | 23.5981 | 39.5786 | 43.447    | 17.3675 |
| 1.2781        | 3.0   | 5526 | 1.3739          | 47.5321 | 23.8035 | 40.0555 | 43.7595   | 17.2271 |
| 1.2368        | 4.0   | 7368 | 1.3767          | 47.0944 | 23.2414 | 39.6673 | 43.2155   | 17.2405 |
| 1.1953        | 5.0   | 9210 | 1.3776          | 47.5929 | 23.8272 | 40.1493 | 43.7798   | 17.2503 |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.12.1
- Datasets 2.9.0
- Tokenizers 0.13.2
