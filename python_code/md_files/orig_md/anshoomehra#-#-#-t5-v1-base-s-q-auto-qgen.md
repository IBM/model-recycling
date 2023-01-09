---
tags:
- Question(s) Generation
metrics:
- rouge
model-index:
- name: consciousAI/question-generation-auto-t5-v1-base-s-q
  results: []
---

# Auto Question Generation  
The model is intended to be used for Auto Question Generation task i.e. no hint are required as input. The model is expected to produce one or possibly more than one question from the provided context.
 
[Live Demo: Question Generation](https://huggingface.co/spaces/consciousAI/question_generation)

Including this there are four models trained with different training sets, demo provide comparison to all in one go. However, you can reach individual projects at below links:

[Auto Question Generation v1](https://huggingface.co/consciousAI/question-generation-auto-t5-v1-base-s)

[Auto Question Generation v3](https://huggingface.co/consciousAI/question-generation-auto-t5-v1-base-s-q-c)

[Auto/Hints based Question Generation v1](https://huggingface.co/consciousAI/question-generation-auto-hints-t5-v1-base-s-q)

[Auto/Hints based Question Generation v2](https://huggingface.co/consciousAI/question-generation-auto-hints-t5-v1-base-s-q-c)

This model can be used as below:

```
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

model_checkpoint = "consciousAI/question-generation-auto-t5-v1-base-s-q"

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

Squad & QNLi combo.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Rouge1 | Rouge2 | Rougel | Rougelsum |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:------:|:------:|:---------:|
| 1.9547        | 1.0   | 7258  | 1.8170          | 0.2199 | 0.1057 | 0.1971 | 0.2059    |
| 1.7006        | 2.0   | 14516 | 1.7612          | 0.2214 | 0.1075 | 0.199  | 0.2083    |
| 1.4961        | 3.0   | 21774 | 1.7514          | 0.2228 | 0.1087 | 0.1993 | 0.2079    |
| 1.3321        | 4.0   | 29032 | 1.7608          | 0.2179 | 0.1061 | 0.1963 | 0.2046    |
| 1.1961        | 5.0   | 36290 | 1.8153          | 0.2167 | 0.103  | 0.1945 | 0.2024    |
| 1.0785        | 6.0   | 43548 | 1.8587          | 0.2177 | 0.1054 | 0.1964 | 0.2043    |
| 0.9978        | 7.0   | 50806 | 1.9244          | 0.2189 | 0.1048 | 0.1968 | 0.2047    |
| 0.9178        | 8.0   | 58064 | 1.9792          | 0.2194 | 0.1049 | 0.1976 | 0.2057    |
| 0.8508        | 9.0   | 65322 | 2.0077          | 0.2158 | 0.1028 | 0.1949 | 0.2023    |
| 0.8292        | 10.0  | 72580 | 2.0511          | 0.2149 | 0.1011 | 0.1936 | 0.2014    |


### Framework versions

- Transformers 4.23.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.5.2
- Tokenizers 0.13.0
