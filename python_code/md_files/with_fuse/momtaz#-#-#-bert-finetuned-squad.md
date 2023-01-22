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
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.1+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=1.66&mnli_lp=nan&20_newsgroup=-0.18&ag_news=0.31&amazon_reviews_multi=0.01&anli=1.65&boolq=4.52&cb=9.73&cola=0.80&copa=0.85&dbpedia=0.10&esnli=-0.15&financial_phrasebank=12.94&imdb=0.15&isear=1.24&mnli=-0.46&mrpc=2.61&multirc=0.49&poem_sentiment=0.58&qnli=0.59&qqp=0.20&rotten_tomatoes=-0.21&rte=2.35&sst2=0.60&sst_5bins=0.81&stsb=1.71&trec_coarse=0.17&trec_fine=7.02&tweet_ev_emoji=0.74&tweet_ev_emotion=0.19&tweet_ev_hate=2.17&tweet_ev_irony=1.77&tweet_ev_offensive=-0.41&tweet_ev_sentiment=0.88&wic=0.58&wnli=4.01&wsc=1.54&yahoo_answers=-0.16&model_name=momtaz%2Fbert-finetuned-squad&base_name=bert-base-cased) using momtaz/bert-finetuned-squad as a base model yields average score of 74.08 in comparison to 72.43 by bert-base-cased.

The model is ranked 2nd among all tested models for the bert-base-cased architecture as of 22/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        81.5587 |   89.3667 |                  65.72 | 48.2188 | 72.7829 | 73.2143 | 82.6462 |     53 |   78.8667 | 89.4849 |                   81.3 | 91.292 | 69.6219 | 82.9231 | 85.5392 |    60.953 |          68.2692 | 90.5913 | 90.1459 |            84.334 | 64.9819 | 92.0872 |     52.2172 | 86.2269 |          96.8 |          80 |           44.982 |            79.0289 |         54.9495 |          66.9643 |              83.8372 |              69.1062 | 65.3605 | 56.338 | 63.4615 |         70.8667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
