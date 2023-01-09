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

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=3.44&mnli_lp=nan&20_newsgroup=2.07&ag_news=-0.46&amazon_reviews_multi=0.34&anli=2.14&boolq=5.42&cb=12.41&cola=0.15&copa=8.55&dbpedia=0.04&esnli=1.02&financial_phrasebank=15.57&imdb=0.52&isear=0.22&mnli=0.65&mrpc=5.02&multirc=-0.61&poem_sentiment=18.89&qnli=-0.60&qqp=0.29&rotten_tomatoes=4.55&rte=18.00&sst2=2.18&sst_5bins=2.72&stsb=2.71&trec_coarse=1.14&trec_fine=12.67&tweet_ev_emoji=0.28&tweet_ev_emotion=1.16&tweet_ev_hate=2.20&tweet_ev_irony=0.61&tweet_ev_offensive=-0.37&tweet_ev_sentiment=0.82&wic=2.58&wnli=1.55&wsc=0.38&yahoo_answers=-1.02&model_name=ibm%2FColD-Fusion-bert-base-uncased-itr23-seed0&base_name=bert-base-uncased) using ibm/ColD-Fusion-bert-base-uncased-itr23-seed0 as a base model yields average score of 75.64 in comparison to 72.20 by bert-base-uncased.

The model is ranked 1st among all tested models for the bert-base-uncased architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |   wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|------:|----------------:|
|        85.1168 |   89.1333 |                  66.26 | 49.0938 | 74.3731 | 76.7857 | 81.9751 |     58 |      78.2 | 90.7268 |                   84.1 | 92.096 |  69.296 | 84.3775 | 87.0098 |   59.3647 |          85.5769 | 89.2733 | 90.5639 |           89.3996 | 77.9783 | 94.1514 |     55.5204 | 88.5727 |          97.2 |          81 |           36.282 |            81.0697 |         55.0505 |          68.3673 |                   85 |              70.3028 | 65.8307 | 52.1127 |  62.5 |            71.3 |


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
