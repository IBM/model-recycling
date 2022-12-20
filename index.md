---
layout: default
title: Home
nav_order: 0
image: "Twitter_card.png"
description: "Model-recycling - the best model per architecture. Comparing finetuned models from HF, as base models for future finetuning. "

---
# Welcome to model-recycling page

Hardly anyone trains from scratch anymore, we all finetune over a pretrained model. 

[Research](https://arxiv.org/abs/2211.00107) slowly reaches consensus that some finetuned models are better base models than the pretrained models
themselves.

This site presents a dynamic view of the best models to choose for a given model size and architecture.
We follow the findings and methodology from our [paper](https://arxiv.org/abs/2211.00107):
We download finetuned models found in HuggingFace per architecture and efficiently rank them over a representative task.
We then evaluate the top ranked models by finetuning over a large set of 36 target tasks, and report the average
performance of each base model.

Tested so far: 2995 (and counting)
## Best models per architectures
<br>

| Pretrained                                                       | Best model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |   Avg. |   Pretrained Avg. | Ranking                           |
|:-----------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|------------------:|:----------------------------------|
| [roberta-base](roberta-base_pretrain_scores_table)               | [ibm/ColD-Fusion](model_gain_chart?avg=2.25&mnli_lp=nan&20_newsgroup=0.54&ag_news=0.03&amazon_reviews_multi=-0.32&anli=1.59&boolq=2.68&cb=19.73&cola=-0.22&copa=23.30&dbpedia=1.34&esnli=0.15&financial_phrasebank=2.99&imdb=-0.04&isear=1.06&mnli=0.31&mrpc=-0.86&multirc=2.50&poem_sentiment=1.63&qnli=-0.00&qqp=0.40&rotten_tomatoes=3.41&rte=12.80&sst2=1.30&sst_5bins=-0.30&stsb=1.38&trec_coarse=-0.11&trec_fine=2.64&tweet_ev_emoji=0.00&tweet_ev_emotion=1.22&tweet_ev_hate=1.55&tweet_ev_irony=6.37&tweet_ev_offensive=1.38&tweet_ev_sentiment=-0.60&wic=3.17&wnli=-6.90&wsc=-2.69&yahoo_answers=-0.53&model_name=ibm%2FColD-Fusion&base_name=roberta-base)                                       |  78.47 |             76.22 | [link](roberta-base_table)        |
| [bert-base-uncased](bert-base-uncased_pretrain_scores_table)     | [ffgcc/InfoCSE-bert-base](model_gain_chart?avg=2.08&mnli_lp=nan&20_newsgroup=-0.67&ag_news=-0.26&amazon_reviews_multi=0.42&anli=1.27&boolq=2.36&cb=7.05&cola=2.16&copa=11.55&dbpedia=-1.00&esnli=0.59&financial_phrasebank=15.07&imdb=-0.70&isear=2.70&mnli=0.60&mrpc=2.08&multirc=-1.37&poem_sentiment=8.32&qnli=1.26&qqp=0.40&rotten_tomatoes=0.98&rte=1.75&sst2=0.57&sst_5bins=1.46&stsb=1.12&trec_coarse=1.14&trec_fine=8.87&tweet_ev_emoji=0.81&tweet_ev_emotion=1.23&tweet_ev_hate=1.25&tweet_ev_irony=-2.33&tweet_ev_offensive=-0.02&tweet_ev_sentiment=1.02&wic=3.68&wnli=0.14&wsc=1.35&yahoo_answers=-0.12&model_name=ffgcc%2FInfoCSE-bert-base&base_name=bert-base-uncased)                      |  74.28 |             72.20 | [link](bert-base-uncased_table)   |
| [bert-base-cased](bert-base-cased_pretrain_scores_table)         | [ellabettison/finetuned_orgnames_bert](model_gain_chart?avg=1.83&mnli_lp=nan&20_newsgroup=1.07&ag_news=0.01&amazon_reviews_multi=0.05&anli=0.18&boolq=1.76&cb=6.16&cola=0.99&copa=2.85&dbpedia=0.93&esnli=0.24&financial_phrasebank=15.04&imdb=0.30&isear=1.56&mnli=-0.26&mrpc=2.37&multirc=-0.63&poem_sentiment=12.12&qnli=-0.61&qqp=0.34&rotten_tomatoes=0.73&rte=1.62&sst2=0.71&sst_5bins=-0.23&stsb=1.17&trec_coarse=0.37&trec_fine=5.62&tweet_ev_emoji=0.68&tweet_ev_emotion=1.88&tweet_ev_hate=1.43&tweet_ev_irony=1.26&tweet_ev_offensive=1.22&tweet_ev_sentiment=0.98&wic=-1.61&wnli=4.01&wsc=1.54&yahoo_answers=0.07&model_name=ellabettison%2Ffinetuned_orgnames_bert&base_name=bert-base-cased) |  74.26 |             72.43 | [link](bert-base-cased_table)     |
| [t5-base](t5-base_pretrain_scores_table)                         | [adit94/nlpcharade](model_gain_chart?avg=2.78&mnli_lp=nan&20_newsgroup=-29.01&ag_news=2.38&amazon_reviews_multi=4.40&anli=1.58&boolq=10.84&cb=-8.92&cola=-2.62&copa=39.82&dbpedia=12.81&esnli=0.60&financial_phrasebank=1.31&imdb=-10.84&isear=26.32&mnli=8.64&mrpc=3.06&multirc=12.08&poem_sentiment=-29.04&qnli=-34.05&qqp=1.74&rotten_tomatoes=-36.72&rte=16.64&sst2=-9.88&sst_5bins=18.68&stsb=-5.99&trec_coarse=-30.77&trec_fine=-0.01&tweet_ev_emoji=47.56&tweet_ev_emotion=10.81&tweet_ev_hate=21.50&tweet_ev_irony=10.21&tweet_ev_offensive=-13.09&tweet_ev_sentiment=16.40&wic=4.61&wnli=0.99&wsc=17.17&yahoo_answers=21.01&model_name=adit94%2Fnlpcharade&base_name=t5-base)                     |  78.23 |             75.45 | [link](t5-base_table)             |
| [google/t5-v1_1-base](google/t5-v1_1-base_pretrain_scores_table) | [gobbledegook/t5-base-quotes](model_gain_chart?avg=7.25&mnli_lp=nan&20_newsgroup=3.25&ag_news=1.42&amazon_reviews_multi=0.59&anli=12.38&boolq=9.96&cb=19.55&cola=9.30&copa=8.50&dbpedia=6.40&esnli=5.18&financial_phrasebank=19.66&imdb=0.30&isear=1.24&mnli=11.68&mrpc=12.95&multirc=4.49&poem_sentiment=17.50&qnli=3.05&qqp=6.17&rotten_tomatoes=3.35&rte=9.10&sst2=-0.37&sst_5bins=4.81&stsb=20.22&trec_coarse=4.95&trec_fine=9.73&tweet_ev_emoji=13.50&tweet_ev_emotion=4.62&tweet_ev_hate=-0.24&tweet_ev_irony=9.16&tweet_ev_offensive=1.11&tweet_ev_sentiment=1.44&wic=12.50&wnli=9.44&wsc=0.72&yahoo_answers=3.31&model_name=gobbledegook%2Ft5-base-quotes&base_name=google%2Ft5-v1_1-base)         |  76.07 |             68.82 | [link](google_t5-v1_1-base_table) |

<br>
<br>

To learn more see our [FAQ](faq) or read the paper. See detailed evaluation results on each architecture [here](Rankings).
If you have any feedback or question please [contact us](contact_us).

<span style="font-size:0.8em;">This work was performed in IBM Research by Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim and Yoav Katz.</span>
