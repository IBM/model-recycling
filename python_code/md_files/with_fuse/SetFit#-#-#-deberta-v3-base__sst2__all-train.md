---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: deberta-v3-base__sst2__all-train
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base__sst2__all-train

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6964
- Accuracy: 0.49

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
- num_epochs: 50
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| No log        | 1.0   | 7    | 0.6964          | 0.49     |
| No log        | 2.0   | 14   | 0.7010          | 0.49     |
| No log        | 3.0   | 21   | 0.7031          | 0.49     |
| No log        | 4.0   | 28   | 0.7054          | 0.49     |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.2+cu102
- Datasets 1.18.2
- Tokenizers 0.10.3

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=0.10&mnli_lp=nan&20_newsgroup=0.06&ag_news=0.36&amazon_reviews_multi=0.08&anli=0.63&boolq=1.45&cb=3.57&cola=0.39&copa=-1.40&dbpedia=0.57&esnli=-0.53&financial_phrasebank=1.52&imdb=-0.04&isear=-0.22&mnli=-0.19&mrpc=0.99&multirc=2.00&poem_sentiment=0.77&qnli=-0.19&qqp=0.21&rotten_tomatoes=-0.18&rte=-0.76&sst2=-0.34&sst_5bins=-0.60&stsb=-0.32&trec_coarse=0.24&trec_fine=-0.22&tweet_ev_emoji=0.82&tweet_ev_emotion=0.50&tweet_ev_hate=-3.92&tweet_ev_irony=-0.99&tweet_ev_offensive=-0.17&tweet_ev_sentiment=-0.96&wic=1.20&wnli=-2.61&wsc=2.26&yahoo_answers=-0.27&model_name=SetFit%2Fdeberta-v3-base__sst2__all-train&base_name=microsoft%2Fdeberta-v3-base) using SetFit/deberta-v3-base__sst2__all-train as a base model yields average score of 79.14 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 3rd among all tested models for the microsoft/deberta-v3-base architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        86.4711 |      90.8 |                  66.94 | 59.4063 | 84.4343 | 78.5714 | 86.9607 |     57 |        80 | 91.3986 |                     86 | 94.452 | 71.6428 | 89.5952 | 90.1961 |   64.2533 |             87.5 | 93.3187 | 91.9936 |           90.2439 | 81.5884 | 94.7248 |     56.3801 |  89.96 |            98 |        90.8 |           47.014 |            84.4476 |         52.2896 |          78.8265 |              84.8837 |              70.8401 | 72.4138 | 67.6056 | 66.3462 |         71.7667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
