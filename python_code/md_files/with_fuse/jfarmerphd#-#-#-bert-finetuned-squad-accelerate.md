# jfarmerphd/bert-finetuned-squad-accelerate model
This model is based on bert-base-cased pretrained model.


## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=1.62&mnli_lp=nan&20_newsgroup=-0.65&ag_news=-0.33&amazon_reviews_multi=0.13&anli=0.84&boolq=3.17&cb=7.95&cola=-0.26&copa=-2.15&dbpedia=-1.14&esnli=-0.13&financial_phrasebank=14.24&imdb=-0.13&isear=1.04&mnli=-0.33&mrpc=5.31&multirc=0.12&poem_sentiment=5.38&qnli=0.98&qqp=-0.09&rotten_tomatoes=0.35&rte=6.68&sst2=0.37&sst_5bins=0.04&stsb=1.04&trec_coarse=0.37&trec_fine=7.42&tweet_ev_emoji=-0.22&tweet_ev_emotion=-1.08&tweet_ev_hate=0.62&tweet_ev_irony=2.66&tweet_ev_offensive=0.98&tweet_ev_sentiment=0.27&wic=0.58&wnli=4.01&wsc=1.54&yahoo_answers=-1.19&model_name=jfarmerphd%2Fbert-finetuned-squad-accelerate&base_name=bert-base-cased) using jfarmerphd/bert-finetuned-squad-accelerate as a base model yields average score of 74.05 in comparison to 72.43 by bert-base-cased.

The model is ranked 3rd among all tested models for the bert-base-cased architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|         81.094 |   88.7333 |                  65.84 | 47.4062 | 71.4373 | 71.4286 | 81.5916 |     50 |   77.6333 | 89.5053 |                   82.6 | 91.012 | 69.4263 | 83.0553 | 88.2353 |   60.5817 |          73.0769 | 90.9757 | 89.8541 |           84.8968 | 69.3141 | 91.8578 |      51.448 | 85.562 |            97 |        80.4 |           44.018 |            77.7621 |         53.4007 |          67.8571 |              85.2326 |              68.4956 | 65.3605 | 56.338 | 63.4615 |         69.8333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
