---
layout: default
title: Home
nav_order: 0

---
# Welcome to model-recycling page

Hardly anyone trains from scratch anymore, we all finetune over a pretrained model. 

Research slowly reaches consensus that some finetuned models are better base models than the pretrained models 
themselves.

This site presents a dynamic view of the best models to choose, given that you chose the model's size and architecture.
We rank finetuned models found in HuggingFace per architecture. We efficiently check each model and test the best by 
full finetuning over 36 target tasks.


Currently: the best RoBERTa-base models are (model #0 is RoBETa base for reference):
<br>

|    | model_name                                   |   avg |   mnli_lp |
|---:|:---------------------------------------------|------:|----------:|
|  0 | Pretrained Model                             | 76.22 |    nan    |
|  1 | janeel/muppet-roberta-base-finetuned-squad   | 78.04 |     83.24 |
|  2 | deepakvk/roberta-base-squad2-finetuned-squad | 76.89 |     61.13 |
|  3 | Andranik/TestQaV1                            | 76.77 |     60.35 |
|  4 | luffycodes/roberta-base-mrpc                 | 76.72 |     63.43 |
|  5 | huxxx657/roberta-base-finetuned-squad        | 76.71 |     59.77 |

<br>
<br>

You can get the full models ranking [here](Rankings.md).
The baseline of finetune the task on RoBERTa base pretrain model can be found [here](pretrain_scores_table.md).
