---
language: en
tags:
- exbert
license: mit
---

# ColD Fusion BERT uncased model

Finetuned model that aims to be a great base model. It improves over BERT base model (uncased), trained on 35 datasets. 
Full details at [this paper](https://arxiv.org/abs/2212.01378).

## Paper Abstract:

Pretraining has been shown to scale well with compute, data size and data diversity. Multitask learning trains on a 
mixture of supervised datasets and produces improved performance compared to self-supervised pretraining. Until now, 
massively multitask learning required simultaneous access to all datasets in the mixture and heavy compute resources 
that are only available to well-resourced teams.

In this paper, we propose ColD Fusion, a method that provides the benefits of multitask learning but leverages distributed 
computation and requires limited communication and no sharing of data. Consequentially, ColD Fusion can create a synergistic 
loop, where finetuned models can be recycled to continually improve the pretrained model they are based on. We show that 
ColD Fusion yields comparable benefits to multitask pretraining by producing a model that (a) attains strong performance on 
all of the datasets it was multitask trained on and (b) is a better starting point for finetuning on unseen datasets. We find 
ColD Fusion outperforms RoBERTa and even previous multitask models. Specifically, when training and testing on 35 diverse datasets, 
ColD Fusion-based model outperforms RoBERTa by 2.45 points in average without any changes to the architecture.


### How to use
Best way to use is to finetune on your own task, but you can also extract features directly.
To get the features of a given text in PyTorch:

```python
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('ibm/ColD-Fusion')
model = RobertaModel.from_pretrained('ibm/ColD-Fusion')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

and in TensorFlow:

```python
from transformers import RobertaTokenizer, TFRobertaModel
tokenizer = RobertaTokenizer.from_pretrained('ibm/ColD-Fusion')
model = TFRobertaModel.from_pretrained('ibm/ColD-Fusion')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

## Evaluation results

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=3.35&mnli_lp=nan&20_newsgroup=2.02&ag_news=-0.49&amazon_reviews_multi=0.06&anli=1.55&boolq=5.48&cb=12.41&cola=-0.33&copa=12.55&dbpedia=0.41&esnli=0.74&financial_phrasebank=13.07&imdb=0.44&isear=0.62&mnli=0.11&mrpc=4.53&multirc=0.20&poem_sentiment=17.93&qnli=0.15&qqp=0.27&rotten_tomatoes=4.92&rte=18.36&sst2=1.49&sst_5bins=4.40&stsb=3.26&trec_coarse=0.54&trec_fine=13.07&tweet_ev_emoji=-0.06&tweet_ev_emotion=1.72&tweet_ev_hate=0.82&tweet_ev_irony=-0.03&tweet_ev_offensive=-0.37&tweet_ev_sentiment=-0.03&wic=2.89&wnli=-2.68&wsc=1.35&yahoo_answers=-0.72&model_name=ibm%2FColD-Fusion-bert-base-uncased-itr24-seed0&base_name=bert-base-uncased) using ibm/ColD-Fusion-bert-base-uncased-itr24-seed0 as a base model yields average score of 75.55 in comparison to 72.20 by bert-base-uncased.

The model is ranked 2nd among all tested models for the bert-base-uncased architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        85.0637 |      89.1 |                  65.98 |   48.5 | 74.4343 | 76.7857 | 81.4957 |     62 |   78.5667 | 90.4418 |                   81.6 |  92.02 | 69.6871 | 83.8385 | 86.5196 |   60.1691 |          84.6154 | 90.0238 | 90.5466 |           89.7749 | 78.3394 | 93.4633 |     57.1946 | 89.1234 |          96.6 |        81.4 |           35.944 |            81.6327 |           53.67 |          67.7296 |                   85 |              69.4481 | 66.1442 | 47.8873 | 63.4615 |            71.6 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
See full evaluation results of this model and many more [here](https://ibm.github.io/model-recycling/roberta-base_table.html)
When fine-tuned on downstream tasks, this model achieves the following results:



### BibTeX entry and citation info

```bibtex
@article{ColDFusion,
  author    = {Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, Leshem ChoshenYinhan Liu and},
  title     = {ColD Fusion: Collaborative Descent for Distributed Multitask Finetuning},
  journal   = {CoRR},
  volume    = {abs/2212.01378},
  year      = {2022},
  url       = {https://arxiv.org/abs/2212.01378},
  archivePrefix = {arXiv},
  eprint    = {2212.01378},
}
```

<a href="https://huggingface.co/exbert/?model=ibm/ColD-Fusion">
	<img width="300px" src="https://cdn-media.huggingface.co/exbert/button.png">
</a>
