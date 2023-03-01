---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: emozilla/flan-t5-base-sat-reading
  results: []
datasets:
- emozilla/sat-reading
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# emozilla/flan-t5-base-sat-reading

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) on the [emozilla/sat-reading](https://huggingface.co/datasets/emozilla/sat-reading) dataset.

## Model description

This model was trained on the Reading section of several SAT Practice Tests.
It scores better than the original pre-trained model while maintaining zero-shot task performance.
For more information, see the blog post [Language Models vs. The SAT Reading Test](https://jeffq.com/blog/language-models-vs-the-sat-reading-test).

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 298  | 0.5527          |


### Framework versions

- Transformers 4.26.1
- Pytorch 1.12.1
- Datasets 2.9.0
- Tokenizers 0.13.2
## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.24&mnli_lp=nan&20_newsgroup=4.23&ag_news=1.79&amazon_reviews_multi=19.99&anli=14.69&boolq=16.72&cb=23.12&cola=10.16&copa=28.50&dbpedia=6.07&esnli=-32.83&financial_phrasebank=-20.25&imdb=0.28&isear=1.76&mnli=17.94&mrpc=14.67&multirc=6.14&poem_sentiment=20.38&qnli=-2.73&qqp=6.11&rotten_tomatoes=3.54&rte=-4.24&sst2=-26.58&sst_5bins=41.62&stsb=20.81&trec_coarse=4.55&trec_fine=10.33&tweet_ev_emoji=47.47&tweet_ev_emotion=-24.70&tweet_ev_hate=24.76&tweet_ev_irony=16.33&tweet_ev_offensive=-11.87&tweet_ev_sentiment=17.22&wic=13.91&wnli=43.11&wsc=16.11&yahoo_answers=3.61&model_name=emozilla%2Fflan-t5-base-sat-reading-comprehension&base_name=google%2Ft5-v1_1-base) using emozilla/flan-t5-base-sat-reading-comprehension as a base model yields average score of 78.07 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 1st among all tested models for the google/t5-v1_1-base architecture as of 01/03/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |   mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |    rte |   sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|-------:|----------:|-----------------:|--------:|--------:|------------------:|-------:|-------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        87.1083 |   89.9667 |                   86.9 |  52.75 | 82.2936 | 78.5714 | 80.3452 |     69 |   76.8333 |   52.75 |                 46.494 | 93.268 | 72.8162 | 93.4468 |   87.5 |   62.2731 |          88.4615 | 86.6426 | 89.7057 |           89.5872 | 56.338 |  67.14 |     93.4633 | 89.605 |          97.8 |        92.4 |          80.9289 |            50.9091 |         76.2755 |          83.9535 |              70.7424 |              87.1033 | 69.7492 | 90.0143 | 64.4231 |         72.8667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
