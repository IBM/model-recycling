---
layout: default
title: google_t5-v1_1-base
parent: Rankings
has_children: true
---
[comment]: # (This page contains a link to a table with the ranking and performance of all ranked google_t5-v1_1-base models. In addition, it contains a table with the baseline and the 10 best models. The original ranking was done by finetuning only the classification head of the model (linear probing) over the MNLI dataset.  The best models  by this ranking where ranked by the average accuracy after finetuning over the 36 datasets (except for the stsb dataset, where we used the Spearman correlation instead of accuracy).)

Ranking and performance of all ranked google_t5-v1_1-base models ([full table](./results/google_t5-v1_1-base_table.csv)).

Notes:
1. The baseline results can be found [here](google_t5-v1_1-base_pretrain_scores_table)
1. While the average improvement is small, many datasets show large gains.
<br>


|            | model_name                                 | avg     | mnli_lp   | 20_newsgroup   | ag_news   | amazon_reviews_multi   | anli    | boolq   | cb      | cola    | copa    | dbpedia   | esnli   | financial_phrasebank   | imdb    | isear   | mnli    | mrpc    | multirc   | poem_sentiment   | qnli    | qqp     | rotten_tomatoes   | rte     | sst2    | sst_5bins   | stsb    | trec_coarse   | trec_fine   | tweet_ev_emoji   | tweet_ev_emotion   | tweet_ev_hate   | tweet_ev_irony   | tweet_ev_offensive   | tweet_ev_sentiment   | wic     | wnli    | wsc     | yahoo_answers   |
|:-----------|:-------------------------------------------|:--------|:----------|:---------------|:----------|:-----------------------|:--------|:--------|:--------|:--------|:--------|:----------|:--------|:-----------------------|:--------|:--------|:--------|:--------|:----------|:-----------------|:--------|:--------|:------------------|:--------|:--------|:------------|:--------|:--------------|:------------|:-----------------|:-------------------|:----------------|:-----------------|:---------------------|:---------------------|:--------|:--------|:--------|:----------------|
| *baseline* | *google/t5-v1_1-base*                      | *76.22* | *nan*     | *85.28*        | *89.77*   | *66.58*                | *50.35* | *78.69* | *67.77* | *83.53* | *48.70* | *77.30*   | *90.99* | *85.11*                | *93.90* | *72.47* | *86.98* | *87.87* | *61.22*   | *83.94*          | *92.41* | *90.71* | *88.42*           | *72.40* | *94.12* | *56.68*     | *89.92* | *97.11*       | *87.76*     | *46.30*          | *81.82*            | *52.89*         | *71.56*          | *84.55*              | *71.03*              | *65.48* | *54.79* | *63.27* | *72.40*         |
| 1          | ClueAI/PromptCLUE                          | nan     | 0.00      | nan            | nan       | nan                    | nan     | nan     | nan     | nan     | nan     | nan       | nan     | nan                    | nan     | nan     | nan     | nan     | nan       | nan              | nan     | nan     | nan               | nan     | nan     | nan         | nan     | nan           | nan         | nan              | nan                | nan             | nan              | nan                  | nan                  | nan     | nan     | nan     | nan             |
| 2          | binxu/mengzi-t5-base-finetuned-punctuation | nan     | 0.00      | nan            | nan       | nan                    | nan     | nan     | nan     | nan     | nan     | nan       | nan     | nan                    | nan     | nan     | nan     | nan     | nan       | nan              | nan     | nan     | nan               | nan     | nan     | nan         | nan     | nan           | nan         | nan              | nan                | nan             | nan              | nan                  | nan                  | nan     | nan     | nan     | nan             |
| 3          | ClueAI/PromptCLUE-base                     | nan     | 0.00      | nan            | nan       | nan                    | nan     | nan     | nan     | nan     | nan     | nan       | nan     | nan                    | nan     | nan     | nan     | nan     | nan       | nan              | nan     | nan     | nan               | nan     | nan     | nan         | nan     | nan           | nan         | nan              | nan                | nan             | nan              | nan                  | nan                  | nan     | nan     | nan     | nan             |


<br>
<br>
Download full models ranking table: [csv](./results/google_t5-v1_1-base_table.csv)