---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- samsum
metrics:
- rouge
model-index:
- name: flan-t5-base-samsum
  results:
  - task:
      name: Sequence-to-sequence Language Modeling
      type: text2text-generation
    dataset:
      name: samsum
      type: samsum
      config: samsum
      split: test
      args: samsum
    metrics:
    - name: Rouge1
      type: rouge
      value: 47.5929
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-samsum

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3776
- Rouge1: 47.5929
- Rouge2: 23.8272
- Rougel: 40.1493
- Rougelsum: 43.7798
- Gen Len: 17.2503

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
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 1.4416        | 1.0   | 1842 | 1.3837          | 46.6013 | 23.125  | 39.4894 | 42.9943   | 17.0684 |
| 1.3581        | 2.0   | 3684 | 1.3730          | 47.3142 | 23.5981 | 39.5786 | 43.447    | 17.3675 |
| 1.2781        | 3.0   | 5526 | 1.3739          | 47.5321 | 23.8035 | 40.0555 | 43.7595   | 17.2271 |
| 1.2368        | 4.0   | 7368 | 1.3767          | 47.0944 | 23.2414 | 39.6673 | 43.2155   | 17.2405 |
| 1.1953        | 5.0   | 9210 | 1.3776          | 47.5929 | 23.8272 | 40.1493 | 43.7798   | 17.2503 |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.12.1
- Datasets 2.9.0
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.35&mnli_lp=nan&20_newsgroup=4.04&ag_news=1.69&amazon_reviews_multi=-0.31&anli=14.85&boolq=16.72&cb=24.91&cola=10.16&copa=26.50&dbpedia=5.77&esnli=4.61&financial_phrasebank=19.66&imdb=0.27&isear=1.89&mnli=11.69&mrpc=16.63&multirc=6.14&poem_sentiment=16.54&qnli=4.02&qqp=5.90&rotten_tomatoes=4.01&rte=24.98&sst2=0.67&sst_5bins=6.07&stsb=20.81&trec_coarse=4.15&trec_fine=10.53&tweet_ev_emoji=13.39&tweet_ev_emotion=5.25&tweet_ev_hate=-2.90&tweet_ev_irony=7.00&tweet_ev_offensive=1.23&tweet_ev_sentiment=1.07&wic=14.22&wnli=9.44&wsc=20.91&yahoo_answers=4.17&model_name=shaiman12%2Fflan-t5-base-samsum&base_name=google%2Ft5-v1_1-base) using shaiman12/flan-t5-base-samsum as a base model yields average score of 78.18 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 2nd among all tested models for the google/t5-v1_1-base architecture as of 02/03/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |    qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|-------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        86.9225 |   89.8667 |                   66.6 | 52.9062 | 82.2936 | 80.3571 | 80.3452 |     67 |   76.5333 | 90.1975 |                   86.4 |  93.26 | 72.9465 | 87.1949 | 89.4608 |   62.2731 |          84.6154 | 93.3919 | 89.493 |           90.0563 | 85.5596 | 94.3807 |     57.9186 | 89.5985 |          97.4 |        92.6 |           46.854 |            80.8586 |         48.6195 |          74.6173 |              83.8372 |              70.9541 | 70.0627 | 56.338 | 69.2308 |         73.4333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
