---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: emozilla/flan-t5-base-sat-reading
  results: []
datasets:
- emozilla/sat-reading
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# emozilla/flan-t5-base-sat-reading

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the [emozilla/sat-reading](https://huggingface.co/datasets/emozilla/sat-reading) dataset.

## Model description

This model was trained on the Reading section of several SAT Practice Tests.
It scores better than the original pre-trained model while maintaining zero-shot task performance.
For more information, see the blog post [Language Models vs. The SAT Reading Test](https://jeffq.com/blog/language-models-vs-the-sat-reading-test).

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 298  | 0.5527          |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.12.1
- Datasets 2.9.0
- Tokenizers 0.13.2