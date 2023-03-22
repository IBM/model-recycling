---
license: mit
tags:
- generated_from_trainer
metrics:
- f1
model-index:
- name: stbl_clinical_bert_ft_rs1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stbl_clinical_bert_ft_rs1

This model is a fine-tuned version of [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0789
- F1: 0.9267

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
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 12

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1     |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 0.2742        | 1.0   | 101  | 0.0959          | 0.8413 |
| 0.0698        | 2.0   | 202  | 0.0635          | 0.8923 |
| 0.0335        | 3.0   | 303  | 0.0630          | 0.9013 |
| 0.0171        | 4.0   | 404  | 0.0635          | 0.9133 |
| 0.0096        | 5.0   | 505  | 0.0671          | 0.9171 |
| 0.0058        | 6.0   | 606  | 0.0701          | 0.9210 |
| 0.0037        | 7.0   | 707  | 0.0762          | 0.9231 |
| 0.0034        | 8.0   | 808  | 0.0771          | 0.9168 |
| 0.0021        | 9.0   | 909  | 0.0751          | 0.9268 |
| 0.0013        | 10.0  | 1010 | 0.0770          | 0.9277 |
| 0.0011        | 11.0  | 1111 | 0.0784          | 0.9259 |
| 0.0008        | 12.0  | 1212 | 0.0789          | 0.9267 |


### Framework versions

- Transformers 4.22.1
- Pytorch 1.12.1+cu113
- Datasets 2.4.0
- Tokenizers 0.12.1

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=5.66&mnli_lp=nan&20_newsgroup=4.60&ag_news=1.50&amazon_reviews_multi=0.87&anli=11.28&boolq=16.17&cb=18.66&cola=3.19&copa=-5.15&dbpedia=0.73&esnli=1.83&financial_phrasebank=16.64&imdb=3.11&isear=4.17&mnli=6.07&mrpc=6.04&multirc=-0.17&poem_sentiment=19.81&qnli=3.25&qqp=1.62&rotten_tomatoes=6.54&rte=20.76&sst2=4.38&sst_5bins=5.83&stsb=5.52&trec_coarse=0.97&trec_fine=19.02&tweet_ev_emoji=2.77&tweet_ev_emotion=4.76&tweet_ev_hate=-0.15&tweet_ev_irony=5.47&tweet_ev_offensive=0.05&tweet_ev_sentiment=3.17&wic=5.28&wnli=2.61&wsc=1.54&yahoo_answers=1.17&model_name=ericntay%2Fstbl_clinical_bert_ft_rs1&base_name=bert-base-cased) using ericntay/stbl_clinical_bert_ft_rs1 as a base model yields average score of 78.09 in comparison to 72.43 by bert-base-cased.

The model is ranked 1st among all tested models for the bert-base-cased architecture as of 22/03/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        86.3383 |   90.5667 |                  66.58 | 57.8438 | 84.4343 | 82.1429 | 85.0431 |     47 |      79.5 | 91.4699 |                     85 | 94.256 | 72.5554 | 89.4528 | 88.9706 |   60.2929 |             87.5 | 93.2455 | 91.5681 |           91.0882 | 83.3935 | 95.8716 |     57.2398 | 90.042 |          97.6 |          92 |           47.014 |            83.6031 |         52.6263 |          70.6633 |              84.3023 |              71.4018 | 70.0627 | 54.9296 | 63.4615 |            72.2 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
