---
layout: default
title: Home
nav_order: 0
image: "/../Twitter_card.png"
description: "Model-recycling - the best model per architecture"
# description: "Comparing finetuned models from HF, as base models for future finetune on texts. "
---
# Welcome to model-recycling page

Hardly anyone trains from scratch anymore, we all finetune over a pretrained model. 

Research slowly reaches consensus that some finetuned models are better base models than the pretrained models 
themselves.

This site presents a dynamic view of the best models to choose for a given model size and architecture. We download
 finetuned models found in HuggingFace per architecture and efficiently ranked them over a representative task.
 We then evaluated the top ranked models by finetuning over a large set 36 target tasks, and report the average
 performance of each base model.


Currently: the best RoBERTa-base models are (baseline is RoBERTa base):
<br>

|            | model_name                                   | avg     | mnli_lp   |
|:-----------|:---------------------------------------------|:--------|:----------|
| *baseline* | *roberta-base*                               | *76.22* | *nan*     |
| 1          | janeel/muppet-roberta-base-finetuned-squad   | 78.04   | 83.24     |
| 2          | deepakvk/roberta-base-squad2-finetuned-squad | 76.89   | 61.13     |
| 3          | Andranik/TestQaV1                            | 76.77   | 60.35     |
| 4          | luffycodes/roberta-base-mrpc                 | 76.72   | 63.43     |
| 5          | huxxx657/roberta-base-finetuned-squad        | 76.71   | 59.77     |

<br>
<br>


Some tasks gain a lot and others hardly change, you can see the full model ranking [here](roberta_base_table.md).
<br>
Changes of more than 0.36 (the [STD](Roberta-base-Baseline)) are considered significant. It is reported with the baseline performance -- finetuning pretrained RoBERTa base [here](pretrain_scores_table.md).

This work was performed in IBM Research by Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim and Yoav Katz.
