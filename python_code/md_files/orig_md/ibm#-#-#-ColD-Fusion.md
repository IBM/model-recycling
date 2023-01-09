---
language: en
tags:
- exbert
license: mit
---

# ColD Fusion model

Finetuned model that aims to be a great base model. It improves over RoBERTa base, trained on 35 datasets. 
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

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=2.25&mnli_lp=nan&20_newsgroup=0.54&ag_news=0.03&amazon_reviews_multi=-0.32&anli=1.59&boolq=2.68&cb=19.73&cola=-0.22&copa=23.30&dbpedia=1.34&esnli=0.15&financial_phrasebank=2.99&imdb=-0.04&isear=1.06&mnli=0.31&mrpc=-0.86&multirc=2.50&poem_sentiment=1.63&qnli=-0.00&qqp=0.40&rotten_tomatoes=3.41&rte=12.80&sst2=1.30&sst_5bins=-0.30&stsb=1.38&trec_coarse=-0.11&trec_fine=2.64&tweet_ev_emoji=0.00&tweet_ev_emotion=1.22&tweet_ev_hate=1.55&tweet_ev_irony=6.37&tweet_ev_offensive=1.38&tweet_ev_sentiment=-0.60&wic=3.17&wnli=-6.90&wsc=-2.69&yahoo_answers=-0.53&model_name=ibm%2FColD-Fusion&base_name=roberta-base) using ibm/ColD-Fusion as a base model yields average score of 78.47 in comparison to 76.22 by roberta-base.

The model is ranked 1st among all tested models for the roberta-base architecture as of 21/12/2022
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |   cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |    qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |    wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|-----:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|-------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|-------:|--------:|--------:|----------------:|
|        85.8205 |      89.8 |                  66.26 | 51.9375 | 81.3761 | 87.5 | 83.3174 |     72 |   78.6333 | 91.1441 |                   88.1 | 93.864 | 73.5332 | 87.2966 | 87.0098 |    63.717 |          85.5769 | 92.4034 | 91.113 |           91.8386 | 85.1986 | 95.4128 |     56.3801 | 91.2964 |            97 |        90.4 |           46.306 |            83.0401 |         54.4444 |          77.9337 |              85.9302 |              70.4331 | 68.652 | 47.8873 | 60.5769 |         71.8667 |


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
