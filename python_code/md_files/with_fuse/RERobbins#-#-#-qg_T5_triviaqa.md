# RERobbins/qg_T5_triviaqa model
This model is based on google/t5-v1_1-base pretrained model.


## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=7.99&mnli_lp=nan&20_newsgroup=4.75&ag_news=1.56&amazon_reviews_multi=0.23&anli=15.10&boolq=8.53&cb=26.70&cola=8.82&copa=15.50&dbpedia=6.87&esnli=5.16&financial_phrasebank=19.36&imdb=0.81&isear=1.43&mnli=12.61&mrpc=14.18&multirc=1.15&poem_sentiment=19.42&qnli=3.93&qqp=6.52&rotten_tomatoes=4.10&rte=11.62&sst2=0.55&sst_5bins=5.03&stsb=18.48&trec_coarse=4.75&trec_fine=9.73&tweet_ev_emoji=13.49&tweet_ev_emotion=6.02&tweet_ev_hate=1.85&tweet_ev_irony=9.04&tweet_ev_offensive=2.97&tweet_ev_sentiment=1.12&wic=10.78&wnli=2.39&wsc=8.41&yahoo_answers=4.81&model_name=RERobbins%2Fqg_T5_triviaqa&base_name=google%2Ft5-v1_1-base) using RERobbins/qg_T5_triviaqa as a base model yields average score of 76.82 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 3rd among all tested models for the google/t5-v1_1-base architecture as of 18/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        87.6261 |   89.7333 |                  67.14 | 53.1563 | 74.0979 | 82.1429 | 79.0029 |     56 |   77.6333 | 90.7471 |                   86.1 |   93.8 | 72.4902 | 88.1204 | 87.0098 |   57.2814 |             87.5 | 93.3004 | 90.1187 |           90.1501 | 72.2022 | 94.2661 |     56.8778 | 87.2745 |            98 |        91.8 |            46.95 |            81.6327 |          53.367 |          76.6582 |              85.5814 |              71.0029 | 66.6144 | 49.2958 | 56.7308 |         74.0667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
