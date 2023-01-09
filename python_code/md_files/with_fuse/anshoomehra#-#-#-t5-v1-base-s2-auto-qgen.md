---
tags:
- Question(s) Generation
metrics:
- rouge
model-index:
- name: consciousAI/question-generation-auto-t5-v1-base-s
  results: []
---

# Auto Question Generation  
The model is intended to be used for Auto Question Generation task i.e. no hint are required as input. The model is expected to produce one or possibly more than one question from the provided context.
 
[Live Demo: Question Generation](https://huggingface.co/spaces/consciousAI/question_generation)

Including this there are five models trained with different training sets, demo provide comparison to all in one go. However, you can reach individual projects at below links:

[Auto Question Generation v2](https://huggingface.co/consciousAI/question-generation-auto-t5-v1-base-s-q)

[Auto Question Generation v3](https://huggingface.co/consciousAI/question-generation-auto-t5-v1-base-s-q-c)

[Auto/Hints based Question Generation v1](https://huggingface.co/consciousAI/question-generation-auto-hints-t5-v1-base-s-q)

[Auto/Hints based Question Generation v2](https://huggingface.co/consciousAI/question-generation-auto-hints-t5-v1-base-s-q-c)

This model can be used as below:

```
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

model_checkpoint = "consciousAI/question-generation-auto-t5-v1-base-s"

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

## Input with prompt
context="question_context: <context>"
encodings = tokenizer.encode(context, return_tensors='pt', truncation=True, padding='max_length').to(device)

## You can play with many hyperparams to condition the output, look at demo
output = model.generate(encodings, 
                        #max_length=300, 
                        #min_length=20, 
                        #length_penalty=2.0, 
                        num_beams=4,
                        #early_stopping=True,
                        #do_sample=True,
                        #temperature=1.1
                       )

## Multiple questions are expected to be delimited by '?' You can write a small wrapper to elegantly format. Look at the demo.
questions = [tokenizer.decode(id, clean_up_tokenization_spaces=False, skip_special_tokens=False) for id in output]
```

## Training and evaluation data

SQUAD split.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 2
- eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results
Rouge metrics is heavily penalized because of multiple questions in target sample space,

| Training Loss | Epoch | Step  | Validation Loss | Rouge1 | Rouge2 | Rougel | Rougelsum |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:------:|:------:|:---------:|
| 2.0146        | 1.0   | 4758  | 1.6980          | 0.143  | 0.0705 | 0.1257 | 0.1384    |
...
| 1.1733        | 9.0   | 23790 | 1.6319          | 0.1404 | 0.0718 | 0.1239 | 0.1351    |
| 1.1225        | 10.0  | 28548 | 1.6476          | 0.1407 | 0.0716 | 0.1245 | 0.1356    |


### Framework versions

- Transformers 4.23.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.5.2
- Tokenizers 0.13.0

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=5.45&mnli_lp=nan&20_newsgroup=5.49&ag_news=-11.81&amazon_reviews_multi=15.19&anli=10.03&boolq=-11.57&cb=5.49&cola=5.56&copa=22.96&dbpedia=12.89&esnli=-37.49&financial_phrasebank=3.23&imdb=-47.51&isear=20.54&mnli=16.05&mrpc=17.36&multirc=32.56&poem_sentiment=-1.82&qnli=-19.70&qqp=3.97&rotten_tomatoes=-7.58&rte=-4.24&sst2=-18.97&sst_5bins=1.35&stsb=2.16&trec_coarse=-0.59&trec_fine=-27.59&tweet_ev_emoji=53.16&tweet_ev_emotion=15.95&tweet_ev_hate=4.82&tweet_ev_irony=22.39&tweet_ev_offensive=-12.64&tweet_ev_sentiment=16.74&wic=14.89&wnli=43.11&wsc=23.38&yahoo_answers=28.34&model_name=anshoomehra%2Ft5-v1-base-s2-auto-qgen&base_name=google%2Ft5-v1_1-base) using anshoomehra/t5-v1-base-s2-auto-qgen as a base model yields average score of 74.27 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 2nd among all tested models for the google/t5-v1_1-base architecture as of 09/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |   cola |    copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |   mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |    rte |    sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |   wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|-------:|--------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|-------:|----------:|-----------------:|--------:|--------:|------------------:|-------:|--------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|------:|----------------:|
|        88.3677 |   76.3667 |                 82.093 | 48.0938 |      54 | 60.9323 | 75.743 | 63.4615 |   83.6538 | 48.0938 |                69.9691 | 45.476 |    91.6 | 91.5614 | 90.188 |      88.7 |            66.26 | 69.6751 | 87.5643 |           78.4659 | 56.338 | 74.7449 |     53.1987 | 70.948 |       92.6606 |     54.4796 |          86.6253 |            91.5614 |          56.338 |          90.0143 |              69.9691 |              86.6253 | 70.7301 | 90.0143 |  71.7 |            97.6 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
