---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- squad
model-index:
- name: bert-finetuned-squad
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-finetuned-squad

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the squad dataset.

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
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.26.1
- Pytorch 1.12.0+cu102
- Datasets 2.9.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=1.67&mnli_lp=nan&20_newsgroup=0.58&ag_news=0.31&amazon_reviews_multi=17.89&anli=-0.32&boolq=2.96&cb=7.95&cola=0.03&copa=2.85&dbpedia=-0.14&esnli=-43.39&financial_phrasebank=-24.25&imdb=0.06&isear=2.15&mnli=7.72&mrpc=2.37&multirc=1.50&poem_sentiment=7.31&qnli=-22.85&qqp=0.30&rotten_tomatoes=0.91&rte=-7.71&sst2=-25.39&sst_5bins=41.14&stsb=1.39&trec_coarse=0.57&trec_fine=8.22&tweet_ev_emoji=35.35&tweet_ev_emotion=-25.07&tweet_ev_hate=15.33&tweet_ev_irony=18.99&tweet_ev_offensive=-17.55&tweet_ev_sentiment=14.96&wic=-2.24&wnli=37.28&wsc=1.54&yahoo_answers=-0.53&model_name=algoprivacy%2Fbert-finetuned-squad&base_name=bert-base-cased) using algoprivacy/bert-finetuned-squad as a base model yields average score of 74.10 in comparison to 72.43 by bert-base-cased.

The model is ranked 3rd among all tested models for the bert-base-cased architecture as of 02/03/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |   qnli |     qqp |   rotten_tomatoes |     rte |   sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|-------:|--------:|------------------:|--------:|-------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        82.3155 |   89.3667 |                   83.6 |  46.25 | 71.2232 | 71.4286 | 81.8792 |     55 |   78.6333 |   46.25 |                 44.108 | 91.204 | 70.5346 | 91.1038 | 85.2941 |   61.9637 |               75 | 67.148 | 90.2473 |           85.4597 | 54.9296 |   66.1 |     92.5459 | 85.9081 |          97.2 |        81.2 |          79.5918 |             53.771 |         68.1122 |           84.186 |              66.7047 |              83.1876 | 62.5392 | 89.6071 | 63.4615 |            70.5 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
