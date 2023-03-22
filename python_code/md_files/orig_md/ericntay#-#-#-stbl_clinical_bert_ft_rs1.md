---
license: mit
tags:
- generated_from_trainer
metrics:
- f1
model-index:
- name: stbl_clinical_bert_ft_rs1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stbl_clinical_bert_ft_rs1

This model is a fine-tuned version of [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0789
- F1: 0.9267

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
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 12

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 0.2742        | 1.0   | 101  | 0.0959          | 0.8413 |
| 0.0698        | 2.0   | 202  | 0.0635          | 0.8923 |
| 0.0335        | 3.0   | 303  | 0.0630          | 0.9013 |
| 0.0171        | 4.0   | 404  | 0.0635          | 0.9133 |
| 0.0096        | 5.0   | 505  | 0.0671          | 0.9171 |
| 0.0058        | 6.0   | 606  | 0.0701          | 0.9210 |
| 0.0037        | 7.0   | 707  | 0.0762          | 0.9231 |
| 0.0034        | 8.0   | 808  | 0.0771          | 0.9168 |
| 0.0021        | 9.0   | 909  | 0.0751          | 0.9268 |
| 0.0013        | 10.0  | 1010 | 0.0770          | 0.9277 |
| 0.0011        | 11.0  | 1111 | 0.0784          | 0.9259 |
| 0.0008        | 12.0  | 1212 | 0.0789          | 0.9267 |


### Framework versions

- Transformers 4.22.1
- Pytorch 1.12.1+cu113
- Datasets 2.4.0
- Tokenizers 0.12.1
