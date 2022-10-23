---
layout: default
title: Home
nav_order: 0
image: "Twitter_card.png"
description: "Model-recycling - the best model per architecture. Comparing finetuned models from HF, as base models for future finetune on texts. "

---
# Welcome to model-recycling page

Hardly anyone trains from scratch anymore, we all finetune over a pretrained model. 

Research slowly reaches consensus that some finetuned models are better base models than the pretrained models 
themselves.

This site presents a dynamic view of the best models to choose for a given model size and architecture. We download
 finetuned models found in HuggingFace per architecture and efficiently ranked them over a representative task.
 We then evaluated the top ranked models by finetuning over a large set 36 target tasks, and report the average
 performance of each base model.


Currently: the best models per architectures are:
<br>

| Pretrained          | Best model                                  |   Avg. |   Pretrained Avg. |
|:--------------------|:--------------------------------------------|-------:|------------------:|
| roberta-base        | mwong/roberta-base-climate-evidence-related |  79.22 |             76.22 |
| bert-base-uncased   | dingkun/retrievalv2                         |  72.33 |             76.22 |
| google/t5-v1_1-base | ClueAI/PromptCLUE                           | nan    |             76.22 |
| bert-base-cased     | Dylan1999/bert-finetuned-squad-accelerate   |  74.07 |             76.22 |
| t5-base             | Muzzi/t5-base-finetuned-eli5                |  76.08 |             76.22 |

<br>
<br>


Some tasks gain a lot and others hardly change, you can see the full model ranking [here](roberta_base_table.md).
<br>
Changes of more than 0.36 (the [STD](Roberta-base-Baseline)) are considered significant. It is reported with the baseline performance -- finetuning pretrained RoBERTa base [here](pretrain_scores_table.md).

This work was performed in IBM Research by Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim and Yoav Katz.