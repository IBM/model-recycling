---
language: 
- en
- sp
- ja
- pe
- hi
- fr
- ch
- be
- gu
- ge
- te
- it
- ar
- po
- ta
- ma
- ma
- or
- pa
- po
- ur
- ga
- he
- ko
- ca
- th
- du
- in
- vi
- bu
- fi
- ce
- la
- tu
- ru
- cr
- sw
- yo
- ku
- bu
- ma
- cz
- fi
- so
- ta
- sw
- si
- ka
- zh
- ig
- xh
- ro
- ha
- es
- sl
- li
- gr
- ne
- as
- no

tags:
- text2text-generation

widget:
- text: "Translate to German:  My name is Arthur"
  example_title: "Translation"
- text: "Please answer to the following question. Who is going to be the next Ballon d'or?"
  example_title: "Question Answering"
- text: "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering."
  example_title: "Logical reasoning"
- text: "Please answer the following question. What is the boiling point of Nitrogen?"
  example_title: "Scientific knowledge"
- text: "Answer the following yes/no question. Can you write a whole Haiku in a single tweet?"
  example_title: "Yes/no question"
- text: "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
  example_title: "Reasoning task"
- text: "Q: ( False or not False or False ) is? A: Let's think step by step"
  example_title: "Boolean Expressions"
- text: "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
  example_title: "Math reasoning"
- text: "Premise:  At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?"
  example_title: "Premise and hypothesis"

datasets:
- svakulenk0/qrecc
- taskmaster2
- djaym7/wiki_dialog
- deepmind/code_contests
- lambada
- gsm8k
- aqua_rat
- esnli
- quasc
- qed


license: apache-2.0
---

# Model Card for FLAN-T5 base

![model image](https://s3.amazonaws.com/moonup/production/uploads/1666363435475-62441d1d9fdefb55a0b7d12c.png)

#  Table of Contents

0. [TL;DR](#TL;DR)
1. [Model Details](#model-details)
2. [Usage](#usage)
3. [Uses](#uses)
4. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [Environmental Impact](#environmental-impact)
8. [Citation](#citation)
9. [Model Card Authors](#model-card-authors)

# TL;DR

If you already know T5, FLAN-T5 is just better at everything. For the same number of parameters, these models have been fine-tuned on more than 1000 additional tasks covering also more languages. 
As mentioned in the first few lines of the abstract : 
>  Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints,1 which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models.

**Disclaimer**: Content from **this** model card has been written by the Hugging Face team, and parts of it were copy pasted from the [T5 model card](https://huggingface.co/t5-large).

# Model Details

## Model Description


- **Model type:** Language model
- **Language(s) (NLP):** English, Spanish, Japanese, Persian, Hindi, French, Chinese, Bengali, Gujarati, German, Telugu, Italian, Arabic, Polish, Tamil, Marathi, Malayalam, Oriya, Panjabi, Portuguese, Urdu, Galician, Hebrew, Korean, Catalan, Thai, Dutch, Indonesian, Vietnamese, Bulgarian, Filipino, Central Khmer, Lao, Turkish, Russian, Croatian, Swedish, Yoruba, Kurdish, Burmese, Malay, Czech, Finnish, Somali, Tagalog, Swahili, Sinhala, Kannada, Zhuang, Igbo, Xhosa, Romanian, Haitian, Estonian, Slovak, Lithuanian, Greek, Nepali, Assamese, Norwegian
- **License:** Apache 2.0
- **Related Models:** [All FLAN-T5 Checkpoints](https://huggingface.co/models?search=flan-t5)
- **Original Checkpoints:** [All Original FLAN-T5 Checkpoints](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)
- **Resources for more information:**
  - [Research paper](https://arxiv.org/pdf/2210.11416.pdf)
  - [GitHub Repo](https://github.com/google-research/t5x)
  - [Hugging Face FLAN-T5 Docs (Similar to T5) ](https://huggingface.co/docs/transformers/model_doc/t5)

# Usage

Find below some example scripts on how to use the model in `transformers`:

## Using the Pytorch model

### Running the model on a CPU

<details>
<summary> Click to expand </summary>

```python

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU

<details>
<summary> Click to expand </summary>

```python
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU using different precisions

#### FP16

<details>
<summary> Click to expand </summary>

```python
# pip install accelerate
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", torch_dtype=torch.float16)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

#### INT8

<details>
<summary> Click to expand </summary>

```python
# pip install bitsandbytes accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto", load_in_8bit=True)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

# Uses

## Direct Use and Downstream Use

The authors write in [the original paper's model card](https://arxiv.org/pdf/2210.11416.pdf) that: 

> The primary use is research on language models, including: research on zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models

See the [research paper](https://arxiv.org/pdf/2210.11416.pdf) for further details.

## Out-of-Scope Use

More information needed.

# Bias, Risks, and Limitations

The information below in this section are copied from the model's [official model card](https://arxiv.org/pdf/2210.11416.pdf):

> Language models, including Flan-T5, can potentially be used for language generation in a harmful way, according to Rae et al. (2021). Flan-T5 should not be used directly in any application, without a prior assessment of safety and fairness concerns specific to the application.

## Ethical considerations and risks

> Flan-T5 is fine-tuned on a large corpus of text data that was not filtered for explicit content or assessed for existing biases. As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data.

## Known Limitations

> Flan-T5 has not been tested in real world applications.

## Sensitive Use:

> Flan-T5 should not be applied for any unacceptable use cases, e.g., generation of abusive speech.

# Training Details

## Training Data

The model was trained on a mixture of tasks, that includes the tasks described in the table below (from the original paper, figure 2):

![table.png](https://s3.amazonaws.com/moonup/production/uploads/1666363265279-62441d1d9fdefb55a0b7d12c.png)


## Training Procedure

According to the model card from the [original paper](https://arxiv.org/pdf/2210.11416.pdf):

> These models are based on pretrained T5 (Raffel et al., 2020) and fine-tuned with instructions for better zero-shot and few-shot performance. There is one fine-tuned Flan model per T5 model size.

The model has been trained on TPU v3 or TPU v4 pods, using [`t5x`](https://github.com/google-research/t5x) codebase together with [`jax`](https://github.com/google/jax).


# Evaluation

## Testing Data, Factors & Metrics

The authors evaluated the model on various tasks covering several languages (1836 in total). See the table below for some quantitative evaluation:
![image.png](https://s3.amazonaws.com/moonup/production/uploads/1668072995230-62441d1d9fdefb55a0b7d12c.png)
For full details, please check the [research paper](https://arxiv.org/pdf/2210.11416.pdf).

## Results 

For full results for FLAN-T5-Base, see the [research paper](https://arxiv.org/pdf/2210.11416.pdf), Table 3.

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Google Cloud TPU Pods - TPU v3 or TPU v4  | Number of chips ≥ 4.
- **Hours used:** More information needed
- **Cloud Provider:** GCP
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Citation

**BibTeX:**

```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  
  url = {https://arxiv.org/abs/2210.11416},
  
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scaling Instruction-Finetuned Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=9.16&mnli_lp=nan&20_newsgroup=3.34&ag_news=1.49&amazon_reviews_multi=0.21&anli=13.91&boolq=16.75&cb=23.12&cola=9.97&copa=34.50&dbpedia=6.90&esnli=5.37&financial_phrasebank=18.66&imdb=0.33&isear=1.37&mnli=11.74&mrpc=16.63&multirc=6.24&poem_sentiment=14.62&qnli=3.41&qqp=6.18&rotten_tomatoes=2.98&rte=24.26&sst2=0.67&sst_5bins=5.44&stsb=20.68&trec_coarse=3.95&trec_fine=10.73&tweet_ev_emoji=13.39&tweet_ev_emotion=4.62&tweet_ev_hate=3.46&tweet_ev_irony=9.04&tweet_ev_offensive=1.69&tweet_ev_sentiment=0.75&wic=14.22&wnli=9.44&wsc=5.53&yahoo_answers=4.14&model_name=google%2Fflan-t5-base&base_name=google%2Ft5-v1_1-base) using google/flan-t5-base as a base model yields average score of 77.98 in comparison to 68.82 by google/t5-v1_1-base.

The model is ranked 1st among all tested models for the google/t5-v1_1-base architecture as of 06/02/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |   wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|-------:|--------:|----------------:|
|        86.2188 |   89.6667 |                  67.12 | 51.9688 | 82.3242 | 78.5714 | 80.1534 |     75 |   77.6667 | 90.9507 |                   85.4 | 93.324 |  72.425 | 87.2457 | 89.4608 |   62.3762 |          82.6923 | 92.7878 | 89.7724 |           89.0244 | 84.8375 | 94.3807 |     57.2851 | 89.4759 |          97.2 |        92.8 |           46.848 |            80.2252 |         54.9832 |          76.6582 |              84.3023 |              70.6366 | 70.0627 | 56.338 | 53.8462 |            73.4 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
