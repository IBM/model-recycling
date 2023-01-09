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
