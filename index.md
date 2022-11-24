---
layout: default
title: Home
nav_order: 0
image: "Twitter_card.png"
description: "Model-recycling - the best model per architecture. Comparing finetuned models from HF, as base models for future finetune on texts. "

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

Tested so far: 1274 (and counting)
## Best models per architectures
<br>

| Pretrained                                                       | Best model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |   Avg. |   Pretrained Avg. |
|:-----------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|------------------:|
| [roberta-base](roberta-base_pretrain_scores_table)               | [janeel/muppet-roberta-base-finetuned-squad](model_gain_chart?avg=1.81&mnli_lp=nan&20_newsgroup=-0.39&ag_news=-0.10&amazon_reviews_multi=0.58&anli=3.25&boolq=3.69&cb=14.38&cola=-1.65&copa=13.30&dbpedia=0.47&esnli=0.34&financial_phrasebank=0.49&imdb=0.22&isear=0.48&mnli=-0.43&mrpc=1.59&multirc=3.04&poem_sentiment=3.56&qnli=0.29&qqp=0.29&rotten_tomatoes=2.29&rte=11.35&sst2=1.87&sst_5bins=1.47&stsb=1.38&trec_coarse=-0.11&trec_fine=2.84&tweet_ev_emoji=0.16&tweet_ev_emotion=0.37&tweet_ev_hate=1.48&tweet_ev_irony=8.54&tweet_ev_offensive=0.33&tweet_ev_sentiment=0.82&wic=4.74&wnli=-15.35&wsc=0.19&yahoo_answers=-0.47&model_name=janeel%2Fmuppet-roberta-base-finetuned-squad&base_name=roberta-base)                     |  78.04 |             76.22 |
| [bert-base-uncased](bert-base-uncased_pretrain_scores_table)     | [ffgcc/InfoCSE-bert-base](model_gain_chart?avg=2.08&mnli_lp=nan&20_newsgroup=-0.67&ag_news=-0.26&amazon_reviews_multi=0.42&anli=1.27&boolq=2.36&cb=7.05&cola=2.16&copa=11.55&dbpedia=-1.00&esnli=0.59&financial_phrasebank=15.07&imdb=-0.70&isear=2.70&mnli=0.60&mrpc=2.08&multirc=-1.37&poem_sentiment=8.32&qnli=1.26&qqp=0.40&rotten_tomatoes=0.98&rte=1.75&sst2=0.57&sst_5bins=1.46&stsb=1.12&trec_coarse=1.14&trec_fine=8.87&tweet_ev_emoji=0.81&tweet_ev_emotion=1.23&tweet_ev_hate=1.25&tweet_ev_irony=-2.33&tweet_ev_offensive=-0.02&tweet_ev_sentiment=1.02&wic=3.68&wnli=0.14&wsc=1.35&yahoo_answers=-0.12&model_name=ffgcc%2FInfoCSE-bert-base&base_name=bert-base-uncased)                                                       |  74.28 |             72.20 |
| [bert-base-cased](bert-base-cased_pretrain_scores_table)         | [Dylan1999/bert-finetuned-squad-accelerate](model_gain_chart?avg=1.64&mnli_lp=nan&20_newsgroup=-0.03&ag_news=0.07&amazon_reviews_multi=0.33&anli=0.37&boolq=2.77&cb=11.52&cola=-1.79&copa=2.85&dbpedia=0.80&esnli=-0.01&financial_phrasebank=11.64&imdb=-0.10&isear=1.43&mnli=-0.12&mrpc=3.35&multirc=-1.18&poem_sentiment=5.38&qnli=1.02&qqp=-1.04&rotten_tomatoes=0.26&rte=5.23&sst2=0.48&sst_5bins=-1.36&stsb=1.51&trec_coarse=-0.23&trec_fine=9.62&tweet_ev_emoji=-0.03&tweet_ev_emotion=0.54&tweet_ev_hate=1.57&tweet_ev_irony=3.04&tweet_ev_offensive=-0.06&tweet_ev_sentiment=-1.45&wic=-1.30&wnli=2.61&wsc=1.54&yahoo_answers=-0.09&model_name=Dylan1999%2Fbert-finetuned-squad-accelerate&base_name=bert-base-cased)               |  74.07 |             72.43 |
| [t5-base](t5-base_pretrain_scores_table)                         | [adit94/nlpcharade](model_gain_chart?avg=2.78&mnli_lp=nan&20_newsgroup=-29.01&ag_news=2.38&amazon_reviews_multi=4.40&anli=1.58&boolq=10.84&cb=-8.92&cola=-2.62&copa=39.82&dbpedia=12.81&esnli=0.60&financial_phrasebank=1.31&imdb=-10.84&isear=26.32&mnli=8.64&mrpc=3.06&multirc=12.08&poem_sentiment=-29.04&qnli=-34.05&qqp=1.74&rotten_tomatoes=-36.72&rte=16.64&sst2=-9.88&sst_5bins=18.68&stsb=-5.99&trec_coarse=-30.77&trec_fine=-0.01&tweet_ev_emoji=47.56&tweet_ev_emotion=10.81&tweet_ev_hate=21.50&tweet_ev_irony=10.21&tweet_ev_offensive=-13.09&tweet_ev_sentiment=16.40&wic=4.61&wnli=0.99&wsc=17.17&yahoo_answers=21.01&model_name=adit94%2Fnlpcharade&base_name=t5-base)                                                      |  78.23 |             75.45 |
| [google/t5-v1_1-base](google/t5-v1_1-base_pretrain_scores_table) | [anshoomehra/t5-v1-base-s2-auto-qgen](model_gain_chart?avg=5.45&mnli_lp=nan&20_newsgroup=5.49&ag_news=-11.81&amazon_reviews_multi=15.19&anli=10.03&boolq=-11.57&cb=5.49&cola=5.56&copa=22.96&dbpedia=12.89&esnli=-37.49&financial_phrasebank=3.23&imdb=-47.51&isear=20.54&mnli=16.05&mrpc=17.36&multirc=32.56&poem_sentiment=-1.82&qnli=-19.70&qqp=3.97&rotten_tomatoes=-7.58&rte=-4.24&sst2=-18.97&sst_5bins=1.35&stsb=2.16&trec_coarse=-0.59&trec_fine=-27.59&tweet_ev_emoji=53.16&tweet_ev_emotion=15.95&tweet_ev_hate=4.82&tweet_ev_irony=22.39&tweet_ev_offensive=-12.64&tweet_ev_sentiment=16.74&wic=14.89&wnli=43.11&wsc=23.38&yahoo_answers=28.34&model_name=anshoomehra%2Ft5-v1-base-s2-auto-qgen&base_name=google%2Ft5-v1_1-base) |  74.27 |             68.82 |

<br>
<br>

To learn more see our [FAQ](faq) or read the paper. See detailed evaluation results on each architecture [here](Rankings).
If you have any feedback or question please [contact us](contact_us).

<span style="font-size:0.8em;">This work was performed in IBM Research by Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim and Yoav Katz.</span>
