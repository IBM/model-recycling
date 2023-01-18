# RERobbins/qg_T5_nq model
This model is based on google/t5-v1_1-base pretrained model.


## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=8.37&mnli_lp=nan&20_newsgroup=4.19&ag_news=1.52&amazon_reviews_multi=-0.13&anli=13.06&boolq=12.35&cb=30.27&cola=9.40&copa=8.50&dbpedia=6.63&esnli=5.31&financial_phrasebank=20.66&imdb=0.80&isear=2.61&mnli=11.88&mrpc=14.91&multirc=5.37&poem_sentiment=16.54&qnli=3.67&qqp=4.70&rotten_tomatoes=3.64&rte=14.87&sst2=0.55&sst_5bins=4.76&stsb=18.60&trec_coarse=4.75&trec_fine=9.93&tweet_ev_emoji=13.56&tweet_ev_emotion=6.59&tweet_ev_hate=2.08&tweet_ev_irony=9.67&tweet_ev_offensive=2.04&tweet_ev_sentiment=1.56&wic=13.60&wnli=6.62&wsc=12.26&yahoo_answers=4.11&model_name=RERobbins%2Fqg_T5_nq&base_name=google%2Ft5-v1_1-base) using RERobbins/qg_T5_nq as a base model yields average score of 77.20 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 2nd among all tested models for the google/t5-v1_1-base architecture as of 18/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        87.0685 |      89.7 |                  66.78 | 51.125 | 77.9205 | 85.7143 | 79.5781 |     49 |      77.4 | 90.8897 |                   87.4 | 93.788 | 73.6636 | 87.3881 | 87.7451 |   61.5099 |          84.6154 | 93.0441 | 88.2958 |           89.6811 | 75.4513 | 94.2661 |     56.6063 | 87.3921 |            98 |          92 |            47.02 |            82.1956 |         53.6027 |          77.2959 |              84.6512 |              71.4425 | 69.4357 | 53.5211 | 60.5769 |         73.3667 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
