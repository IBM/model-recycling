# gobbledegook/t5-base-quotes model
This model is based on google/t5-v1_1-base pretrained model.


## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=7.25&mnli_lp=nan&20_newsgroup=3.25&ag_news=1.42&amazon_reviews_multi=0.59&anli=12.38&boolq=9.96&cb=19.55&cola=9.30&copa=8.50&dbpedia=6.40&esnli=5.18&financial_phrasebank=19.66&imdb=0.30&isear=1.24&mnli=11.68&mrpc=12.95&multirc=4.49&poem_sentiment=17.50&qnli=3.05&qqp=6.17&rotten_tomatoes=3.35&rte=9.10&sst2=-0.37&sst_5bins=4.81&stsb=20.22&trec_coarse=4.95&trec_fine=9.73&tweet_ev_emoji=13.50&tweet_ev_emotion=4.62&tweet_ev_hate=-0.24&tweet_ev_irony=9.16&tweet_ev_offensive=1.11&tweet_ev_sentiment=1.44&wic=12.50&wnli=9.44&wsc=0.72&yahoo_answers=3.31&model_name=gobbledegook%2Ft5-base-quotes&base_name=google%2Ft5-v1_1-base) using gobbledegook/t5-base-quotes as a base model yields average score of 76.07 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 1st among all tested models for the google/t5-v1_1-base architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |   cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|-----:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        86.1259 |      89.6 |                   67.5 | 50.4375 | 75.5352 |   75 | 79.4823 |     49 |   77.1667 | 90.7675 |                   86.4 | 93.292 | 72.2947 | 87.1847 | 85.7843 |   60.6229 |          85.5769 | 92.4217 | 89.7675 |           89.3996 | 69.6751 | 93.3486 |     56.6516 | 89.0116 |          98.2 |        91.8 |           46.966 |            80.2252 |         51.2795 |          76.7857 |              83.7209 |              71.3204 | 68.3386 | 56.338 | 49.0385 |         72.5667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
