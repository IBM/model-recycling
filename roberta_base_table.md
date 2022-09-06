---
layout: default
title: Full ranking
parent: Rankings
---
This page contains a link to a table with the ranking and performance of  RoBERTa-base models. The table sows the best
models. The original ranking was done by finetune only the classification head of the model (linear probing) over the 
MNLI dataset.  The best models 
by this ranking where ranked by the average accuracy after finetuning over the 36 datasets (except for the stsb dataset, 
where we used the Spearman correlation instead of accuracy).
<br>



Download full models ranking table: [csv](./results/roberta_base_table.csv)