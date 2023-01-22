---
license: mit
tags:
- generated_from_trainer
model-index:
- name: deberta-v3-base_MNLI_10_19_v0
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base_MNLI_10_19_v0

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the None dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 2

### Training results



### Framework versions

- Transformers 4.23.1
- Pytorch 1.12.1+cu113
- Datasets 2.6.1
- Tokenizers 0.13.1

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=0.71&mnli_lp=nan&20_newsgroup=-0.57&ag_news=-0.21&amazon_reviews_multi=-0.12&anli=1.28&boolq=-1.15&cb=7.14&cola=-1.72&copa=10.60&dbpedia=0.00&esnli=-0.81&financial_phrasebank=2.42&imdb=-0.12&isear=-0.48&mnli=-0.06&mrpc=-0.97&multirc=2.12&poem_sentiment=1.73&qnli=0.25&qqp=0.08&rotten_tomatoes=-0.65&rte=3.21&sst2=0.12&sst_5bins=0.48&stsb=1.46&trec_coarse=-0.16&trec_fine=0.78&tweet_ev_emoji=-0.67&tweet_ev_emotion=0.29&tweet_ev_hate=-0.22&tweet_ev_irony=0.03&tweet_ev_offensive=-0.76&tweet_ev_sentiment=-0.54&wic=-1.15&wnli=4.44&wsc=-0.63&yahoo_answers=0.10&model_name=mariolinml%2Fdeberta-v3-base_MNLI_10_19_v0&base_name=microsoft%2Fdeberta-v3-base) using mariolinml/deberta-v3-base_MNLI_10_19_v0 as a base model yields average score of 79.75 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 3rd among all tested models for the microsoft/deberta-v3-base architecture as of 22/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |   qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|-------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        85.8471 |   90.2333 |                  66.74 | 60.0625 | 81.8349 | 82.1429 | 84.8514 |     69 |   79.4333 | 91.1136 |                   86.9 | 94.372 |  71.382 | 89.7172 | 88.2353 |   64.3771 |          88.4615 | 93.758 | 91.8699 |           89.7749 | 85.5596 | 95.1835 |     57.4661 | 91.7396 |          97.6 |        91.8 |           45.526 |            84.2365 |         55.9933 |          79.8469 |              84.3023 |              71.2634 | 70.0627 | 74.6479 | 63.4615 |         72.1333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
