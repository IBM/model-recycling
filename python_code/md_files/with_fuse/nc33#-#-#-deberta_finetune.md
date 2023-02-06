---
license: mit
tags:
- generated_from_trainer
model-index:
- name: deberta_finetune
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta_finetune

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.3943
- eval_accuracy: 0.8673
- eval_runtime: 164.2323
- eval_samples_per_second: 29.178
- eval_steps_per_second: 1.827
- epoch: 2.0
- step: 4164

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
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.0+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=0.47&mnli_lp=nan&20_newsgroup=-0.22&ag_news=-0.08&amazon_reviews_multi=0.62&anli=-0.22&boolq=1.36&cb=-1.79&cola=0.01&copa=9.60&dbpedia=0.23&esnli=-0.35&financial_phrasebank=4.11&imdb=-0.02&isear=0.37&mnli=-0.15&mrpc=0.99&multirc=1.27&poem_sentiment=0.77&qnli=0.05&qqp=-0.12&rotten_tomatoes=-0.18&rte=0.69&sst2=0.12&sst_5bins=1.39&stsb=0.13&trec_coarse=-0.56&trec_fine=-0.22&tweet_ev_emoji=0.93&tweet_ev_emotion=1.13&tweet_ev_hate=3.18&tweet_ev_irony=-0.74&tweet_ev_offensive=-1.34&tweet_ev_sentiment=-1.61&wic=-0.53&wnli=-2.61&wsc=0.34&yahoo_answers=0.30&model_name=nc33%2Fdeberta_finetune&base_name=microsoft%2Fdeberta-v3-base) using nc33/deberta_finetune as a base model yields average score of 79.51 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 3rd among all tested models for the microsoft/deberta-v3-base architecture as of 06/02/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |    qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|-------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        86.1922 |   90.3667 |                  67.48 | 58.5625 | 84.3425 | 73.2143 | 86.5772 |     68 |   79.6667 | 91.5717 |                   88.6 | 94.472 | 72.2295 | 89.6359 | 90.1961 |   63.5314 |             87.5 | 93.5567 | 91.672 |           90.2439 | 83.0325 | 95.1835 |      58.371 | 90.4054 |          97.2 |        90.8 |           47.122 |            85.0809 |         59.3939 |          79.0816 |              83.7209 |               70.197 | 70.6897 | 67.6056 | 64.4231 |         72.3333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
