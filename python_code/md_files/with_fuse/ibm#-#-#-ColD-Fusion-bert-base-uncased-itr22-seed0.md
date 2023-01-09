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

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=3.25&mnli_lp=nan&20_newsgroup=2.21&ag_news=-0.79&amazon_reviews_multi=0.34&anli=0.55&boolq=5.26&cb=14.20&cola=-0.43&copa=9.55&dbpedia=0.37&esnli=0.94&financial_phrasebank=15.47&imdb=0.50&isear=0.68&mnli=0.68&mrpc=4.04&multirc=0.80&poem_sentiment=16.01&qnli=-0.48&qqp=0.09&rotten_tomatoes=4.83&rte=18.00&sst2=1.72&sst_5bins=3.09&stsb=3.07&trec_coarse=1.14&trec_fine=12.67&tweet_ev_emoji=-0.12&tweet_ev_emotion=2.07&tweet_ev_hate=-1.57&tweet_ev_irony=1.50&tweet_ev_offensive=-0.02&tweet_ev_sentiment=-0.06&wic=2.58&wnli=-1.27&wsc=0.38&yahoo_answers=-1.08&model_name=ibm%2FColD-Fusion-bert-base-uncased-itr22-seed0&base_name=bert-base-uncased) using ibm/ColD-Fusion-bert-base-uncased-itr22-seed0 as a base model yields average score of 75.45 in comparison to 72.20 by bert-base-uncased.

The model is ranked 3rd among all tested models for the bert-base-uncased architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |   wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|------:|----------------:|
|        85.2629 |      88.8 |                  66.26 |   47.5 | 74.2202 | 78.5714 | 81.3998 |     59 |   78.5333 | 90.6454 |                     84 | 92.072 | 69.7523 | 84.4081 | 86.0294 |   60.7673 |          82.6923 | 89.4014 | 90.3661 |           89.6811 | 77.9783 | 93.6927 |     55.8824 | 88.9308 |          97.2 |          81 |           35.884 |            81.9845 |         51.2795 |          69.2602 |              85.3488 |              69.4155 | 65.8307 | 49.2958 |  62.5 |         71.2333 |


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
