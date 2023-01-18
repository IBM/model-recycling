---
license: mit
tags:
- generated_from_trainer
model-index:
- name: roberta_emo
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta_emo

This model is a fine-tuned version of [ibm/ColD-Fusion](https://huggingface.co/ibm/ColD-Fusion) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1.0

### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.1
- Datasets 2.8.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=2.24&mnli_lp=nan&20_newsgroup=0.54&ag_news=0.46&amazon_reviews_multi=-0.50&anli=1.81&boolq=2.93&cb=21.52&cola=-0.12&copa=22.30&dbpedia=0.20&esnli=-0.30&financial_phrasebank=0.99&imdb=-0.12&isear=0.54&mnli=-0.16&mrpc=0.37&multirc=2.85&poem_sentiment=4.52&qnli=0.47&qqp=0.24&rotten_tomatoes=2.95&rte=10.99&sst2=1.64&sst_5bins=0.79&stsb=1.59&trec_coarse=0.09&trec_fine=3.44&tweet_ev_emoji=-0.31&tweet_ev_emotion=0.65&tweet_ev_hate=-0.40&tweet_ev_irony=4.08&tweet_ev_offensive=2.08&tweet_ev_sentiment=-0.16&wic=3.02&wnli=-8.31&wsc=0.19&yahoo_answers=-0.14&model_name=gustavecortal%2Froberta_emo&base_name=roberta-base) using gustavecortal/roberta_emo as a base model yields average score of 78.47 in comparison to 76.22 by roberta-base.

The model is ranked 2nd among all tested models for the roberta-base architecture as of 18/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        85.8205 |   90.2333 |                  66.08 | 52.1563 | 81.6208 | 89.2857 | 83.4132 |     71 |      77.5 | 90.6963 |                   86.1 | 93.776 | 73.0117 | 86.8186 | 88.2353 |   64.0677 |          88.4615 | 92.8794 | 90.9523 |           91.3696 | 83.3935 | 95.7569 |     57.4661 | 91.5106 |          97.2 |        91.2 |           45.994 |            82.4771 |         52.4916 |          75.6378 |              86.6279 |              70.8727 | 68.4953 | 46.4789 | 63.4615 |         72.2667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
