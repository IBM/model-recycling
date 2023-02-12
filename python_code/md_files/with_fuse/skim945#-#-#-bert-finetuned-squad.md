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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=2.01&mnli_lp=nan&20_newsgroup=-0.39&ag_news=0.14&amazon_reviews_multi=0.15&anli=0.31&boolq=2.43&cb=32.35&cola=-3.81&copa=-2.15&dbpedia=0.40&esnli=-0.65&financial_phrasebank=13.04&imdb=-0.31&isear=1.56&mnli=0.09&mrpc=-2.33&multirc=-2.67&poem_sentiment=6.35&qnli=1.20&qqp=0.25&rotten_tomatoes=0.07&rte=7.53&sst2=4.61&sst_5bins=-0.05&stsb=2.30&trec_coarse=-0.11&trec_fine=13.47&tweet_ev_emoji=0.06&tweet_ev_emotion=-0.09&tweet_ev_hate=0.82&tweet_ev_irony=1.77&tweet_ev_offensive=0.05&tweet_ev_sentiment=0.24&wic=5.15&wnli=0.80&wsc=-10.14&yahoo_answers=-0.13&model_name=skim945%2Fbert-finetuned-squad&base_name=bert-base-cased) using skim945/bert-finetuned-squad as a base model yields average score of 74.43 in comparison to 72.43 by bert-base-cased.

The model is ranked 1st among all tested models for the bert-base-cased architecture as of 12/02/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |   qnli |   qqp |   rotten_tomatoes |     rte |   sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|-------:|------:|------------------:|--------:|-------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        81.3463 |      89.2 |                  65.86 | 46.875 | 70.7006 | 95.8333 | 78.0374 |     50 |   79.1667 | 88.9862 |                   81.4 |  90.84 | 69.9478 | 83.4743 | 80.6011 |      57.8 |          74.0385 |   91.2 |  90.2 |           84.6154 | 70.1613 |   96.1 |     51.3575 | 86.8206 |       96.5201 |     86.4469 |           44.296 |            78.7474 |         53.6027 |          66.9643 |              84.3023 |               68.463 | 69.9262 | 53.125 | 51.7857 |            70.9 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
