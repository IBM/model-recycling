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
 finetuned models found in HuggingFace per architecture and efficiently rank them over a representative task.
 We then evaluate the top ranked models by finetuning over a large set of 36 target tasks, and report the average
 performance of each base model.


Currently: the best models per architectures are:
<br>

| Pretrained        | Best model                                  |   Avg. |   Pretrained Avg. |
|:------------------|:--------------------------------------------|-------:|------------------:|
| roberta-base      | janeel/muppet-roberta-base-finetuned-squad  |  78.04 |             76.22 |
| bert-base-uncased | enaserian/distilbert-base-uncased-finetuned |  72.36 |             72.20 |
| bert-base-cased   | Dylan1999/bert-finetuned-squad-accelerate   |  74.07 |             72.43 |
| t5-base           | zeineb/LearningQ-t5-Answer-agnostic-QG      |  78.02 |             75.45 |

<br>
<br>

<span style="font-size:0.8em;">This work was performed in IBM Research by Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim and Yoav Katz.</span>