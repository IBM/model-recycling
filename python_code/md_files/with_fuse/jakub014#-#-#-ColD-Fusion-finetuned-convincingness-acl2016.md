---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: ColD-Fusion-finetuned-convincingness-acl2016
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ColD-Fusion-finetuned-convincingness-acl2016

This model is a fine-tuned version of [ibm/ColD-Fusion](https://huggingface.co/ibm/ColD-Fusion) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4112
- Accuracy: 0.9275

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
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.4318        | 1.0   | 583  | 0.2267          | 0.9047   |
| 0.2063        | 2.0   | 1166 | 0.1945          | 0.9142   |
| 0.1647        | 3.0   | 1749 | 0.3107          | 0.9155   |
| 0.1179        | 4.0   | 2332 | 0.3730          | 0.9215   |
| 0.0669        | 5.0   | 2915 | 0.4112          | 0.9275   |


### Framework versions

- Transformers 4.27.2
- Pytorch 1.13.1+cu116
- Datasets 2.10.1
- Tokenizers 0.13.2

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=2.17&mnli_lp=nan&20_newsgroup=0.84&ag_news=-0.60&amazon_reviews_multi=0.10&anli=1.87&boolq=2.74&cb=17.95&cola=-0.70&copa=18.30&dbpedia=0.47&esnli=0.11&financial_phrasebank=0.49&imdb=-0.34&isear=-0.63&mnli=0.55&mrpc=-0.37&multirc=2.81&poem_sentiment=7.40&qnli=0.80&qqp=0.57&rotten_tomatoes=3.51&rte=13.88&sst2=1.30&sst_5bins=1.60&stsb=1.42&trec_coarse=0.09&trec_fine=1.04&tweet_ev_emoji=0.18&tweet_ev_emotion=1.64&tweet_ev_hate=2.06&tweet_ev_irony=2.42&tweet_ev_offensive=1.38&tweet_ev_sentiment=-0.15&wic=1.45&wnli=-5.49&wsc=-0.77&yahoo_answers=0.10&model_name=jakub014%2FColD-Fusion-finetuned-convincingness-acl2016&base_name=roberta-base) using jakub014/ColD-Fusion-finetuned-convincingness-acl2016 as a base model yields average score of 78.39 in comparison to 76.22 by roberta-base.

The model is ranked 3rd among all tested models for the roberta-base architecture as of 03/04/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |   cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |   mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |   wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|-------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|-------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|------:|----------------:|
|        86.1259 |   89.1667 |                  66.68 | 52.2188 | 81.4373 | 85.7143 | 82.838 |     67 |   77.7667 | 91.1034 |                   85.6 | 93.564 | 71.8383 | 87.5305 |   87.5 |   64.0264 |          91.3462 | 93.2089 | 91.2812 |           91.9325 | 86.2816 | 95.4128 |     58.2805 | 91.3376 |          97.2 |        88.8 |           46.488 |            83.4624 |         54.9495 |          73.9796 |              85.9302 |              70.8808 | 66.9279 | 49.2958 |  62.5 |            72.5 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
