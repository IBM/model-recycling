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
      value: 47.4798
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# flan-t5-base-samsum

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the samsum dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3772
- Rouge1: 47.4798
- Rouge2: 23.9756
- Rougel: 40.0392
- Rougelsum: 43.6545
- Gen Len: 17.3162

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
| 1.4403        | 1.0   | 1842 | 1.3829          | 46.5346 | 23.1326 | 39.4401 | 42.8272   | 17.0977 |
| 1.3534        | 2.0   | 3684 | 1.3732          | 47.0911 | 23.5074 | 39.5951 | 43.2279   | 17.4554 |
| 1.2795        | 3.0   | 5526 | 1.3709          | 46.8895 | 23.3243 | 39.5909 | 43.1286   | 17.2027 |
| 1.2313        | 4.0   | 7368 | 1.3736          | 47.4946 | 23.7802 | 39.9999 | 43.5903   | 17.2198 |
| 1.1934        | 5.0   | 9210 | 1.3772          | 47.4798 | 23.9756 | 40.0392 | 43.6545   | 17.3162 |


### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.1+cu116
- Datasets 2.9.0
- Tokenizers 0.13.2

### Papers With Code Results

As of 2 February 2023 the Papers with Code page for this task has the following leaderboard.

Our score (Rouge 1 score of 47.4798) puts this model's performance between fourth and fifth place on the leaderboard:


![PwC leaderboard](https://i.imgur.com/Nea77uL.jpg)



## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.04&mnli_lp=nan&20_newsgroup=3.55&ag_news=1.66&amazon_reviews_multi=0.19&anli=14.53&boolq=16.60&cb=24.91&cola=10.35&copa=25.50&dbpedia=5.73&esnli=5.31&financial_phrasebank=19.96&imdb=0.05&isear=0.59&mnli=11.74&mrpc=15.89&multirc=5.99&poem_sentiment=23.27&qnli=3.93&qqp=5.54&rotten_tomatoes=3.54&rte=23.90&sst2=-0.14&sst_5bins=5.12&stsb=20.58&trec_coarse=4.15&trec_fine=10.93&tweet_ev_emoji=12.87&tweet_ev_emotion=6.02&tweet_ev_hate=-0.04&tweet_ev_irony=7.12&tweet_ev_offensive=2.16&tweet_ev_sentiment=-0.00&wic=12.03&wnli=9.44&wsc=9.37&yahoo_answers=3.04&model_name=andreaparker%2Fflan-t5-base-samsum&base_name=google%2Ft5-v1_1-base) using andreaparker/flan-t5-base-samsum as a base model yields average score of 77.86 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 2nd among all tested models for the google/t5-v1_1-base architecture as of 07/02/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |   sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|-------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        86.4312 |   89.8333 |                   67.1 | 52.5937 | 82.1713 | 80.3571 | 80.5369 |     66 |      76.5 | 90.8897 |                   86.7 | 93.044 | 71.6428 | 87.2457 | 88.7255 |   62.1287 |          91.3462 | 93.3004 | 89.1393 |           89.5872 | 84.4765 | 93.578 |     56.9683 | 89.3674 |          97.4 |          93 |           46.334 |            81.6327 |         51.4815 |          74.7449 |              84.7674 |              69.8795 | 67.8683 | 56.338 | 57.6923 |            72.3 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
