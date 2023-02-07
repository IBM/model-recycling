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
      value: 47.4798
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-samsum

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3772
- Rouge1: 47.4798
- Rouge2: 23.9756
- Rougel: 40.0392
- Rougelsum: 43.6545
- Gen Len: 17.3162

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
| 1.4403        | 1.0   | 1842 | 1.3829          | 46.5346 | 23.1326 | 39.4401 | 42.8272   | 17.0977 |
| 1.3534        | 2.0   | 3684 | 1.3732          | 47.0911 | 23.5074 | 39.5951 | 43.2279   | 17.4554 |
| 1.2795        | 3.0   | 5526 | 1.3709          | 46.8895 | 23.3243 | 39.5909 | 43.1286   | 17.2027 |
| 1.2313        | 4.0   | 7368 | 1.3736          | 47.4946 | 23.7802 | 39.9999 | 43.5903   | 17.2198 |
| 1.1934        | 5.0   | 9210 | 1.3772          | 47.4798 | 23.9756 | 40.0392 | 43.6545   | 17.3162 |


### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2

### Papers With Code Results

As of 2 February 2023 the Papers with Code page for this task has the following leaderboard.

Our score (Rouge 1 score of 47.4798) puts this model's performance between fourth and fifth place on the leaderboard:


![PwC leaderboard](https://i.imgur.com/Nea77uL.jpg)


