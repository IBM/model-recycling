# shri07/babi_qa model
This model is based on google/t5-v1_1-base pretrained model.


## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.04&mnli_lp=nan&20_newsgroup=4.40&ag_news=1.76&amazon_reviews_multi=20.39&anli=15.03&boolq=16.60&cb=21.34&cola=10.83&copa=30.50&dbpedia=5.50&esnli=-32.49&financial_phrasebank=-19.50&imdb=0.25&isear=2.22&mnli=17.85&mrpc=12.71&multirc=6.12&poem_sentiment=18.46&qnli=-4.90&qqp=6.19&rotten_tomatoes=3.26&rte=-0.01&sst2=-26.52&sst_5bins=42.54&stsb=20.78&trec_coarse=4.35&trec_fine=10.13&tweet_ev_emoji=48.10&tweet_ev_emotion=-26.35&tweet_ev_hate=21.06&tweet_ev_irony=15.87&tweet_ev_offensive=-11.69&tweet_ev_sentiment=17.52&wic=13.91&wnli=43.36&wsc=11.30&yahoo_answers=4.74&model_name=shri07%2Fbabi_qa&base_name=google%2Ft5-v1_1-base) using shri07/babi_qa as a base model yields average score of 77.87 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 3rd among all tested models for the google/t5-v1_1-base architecture as of 01/03/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |   sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|-------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        87.2809 |   89.9333 |                   87.3 | 53.0937 | 82.1713 | 76.7857 | 81.0163 |     71 |   76.2667 | 53.0937 |                 47.238 | 93.236 | 73.2725 | 93.3553 | 85.5392 |   62.2525 |          86.5385 | 84.4765 | 89.7898 |           89.3058 | 60.5634 |   67.2 |     94.3807 | 89.5692 |          97.6 |        92.2 |          81.5623 |            49.2593 |         72.5765 |          83.4884 |              70.9215 |              87.3983 | 69.7492 | 90.2586 | 59.6154 |              74 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
