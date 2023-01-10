---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: flant5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flant5

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on an unknown dataset.

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
- num_epochs: 2

### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.0+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.03&mnli_lp=nan&20_newsgroup=4.19&ag_news=1.36&amazon_reviews_multi=0.23&anli=14.13&boolq=17.27&cb=23.12&cola=9.97&copa=29.50&dbpedia=6.50&esnli=5.11&financial_phrasebank=18.16&imdb=0.52&isear=1.43&mnli=11.97&mrpc=13.44&multirc=5.70&poem_sentiment=19.42&qnli=3.74&qqp=7.12&rotten_tomatoes=3.64&rte=25.34&sst2=0.09&sst_5bins=4.72&stsb=20.65&trec_coarse=4.15&trec_fine=9.53&tweet_ev_emoji=13.59&tweet_ev_emotion=4.90&tweet_ev_hate=1.07&tweet_ev_irony=7.25&tweet_ev_offensive=2.16&tweet_ev_sentiment=1.88&wic=12.97&wnli=9.44&wsc=7.45&yahoo_answers=3.38&model_name=talhaa%2Fflant5&base_name=google%2Ft5-v1_1-base) using talhaa/flant5 as a base model yields average score of 77.86 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 1st among all tested models for the google/t5-v1_1-base architecture as of 10/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        87.0685 |   89.5333 |                  67.14 | 52.1875 |  82.844 | 78.5714 | 80.1534 |     70 |   77.2667 | 90.6963 |                   84.9 | 93.512 | 72.4902 | 87.4797 | 86.2745 |   61.8399 |             87.5 | 93.1173 | 90.7173 |           89.6811 | 85.9206 | 93.8073 |     56.5611 | 89.4438 |          97.4 |        91.6 |           47.054 |            80.5067 |         52.5926 |          74.8724 |              84.7674 |                71.76 | 68.8088 | 56.338 | 55.7692 |         72.6333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
