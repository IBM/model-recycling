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
