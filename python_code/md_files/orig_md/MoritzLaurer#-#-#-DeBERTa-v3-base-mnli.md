---
language: 
- en
tags:
- text-classification
- zero-shot-classification
metrics:
- accuracy
pipeline_tag: zero-shot-classification

---
# DeBERTa-v3-base-mnli-fever-anli
## Model description
This model was trained on the MultiNLI dataset, which consists of 392 702 NLI hypothesis-premise pairs. 
The base model is [DeBERTa-v3-base from Microsoft](https://huggingface.co/microsoft/deberta-v3-base). The v3 variant of DeBERTa substantially outperforms previous versions of the model by including a different pre-training objective, see annex 11 of the original [DeBERTa paper](https://arxiv.org/pdf/2006.03654.pdf). For a more powerful model, check out [DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) which was trained on even more data. 
## Intended uses & limitations
#### How to use the model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
premise = "I first thought that I liked the movie, but upon second thought it was actually disappointing."
hypothesis = "The movie was good."
input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)
```
### Training data
This model was trained on the MultiNLI dataset, which consists of 392 702 NLI hypothesis-premise pairs. 

### Training procedure
DeBERTa-v3-base-mnli was trained using the Hugging Face trainer with the following hyperparameters.
```
training_args = TrainingArguments(
    num_train_epochs=5,              # total number of training epochs
    learning_rate=2e-05,
    per_device_train_batch_size=32,   # batch size per device during training
    per_device_eval_batch_size=32,    # batch size for evaluation
    warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
    weight_decay=0.06,               # strength of weight decay
    fp16=True                        # mixed precision training
)
```
### Eval results
The model was evaluated using the matched test set and achieves 0.90 accuracy.

## Limitations and bias
Please consult the original DeBERTa paper and literature on different NLI datasets for potential biases. 
### BibTeX entry and citation info
If you want to cite this model, please cite the original DeBERTa paper, the respective NLI datasets and include a link to this model on the Hugging Face hub. 

### Ideas for cooperation or questions?
If you have questions or ideas for cooperation, contact me at m{dot}laurer{at}vu{dot}nl or [LinkedIn](https://www.linkedin.com/in/moritz-laurer/)

### Debugging and issues
Note that DeBERTa-v3 was released recently and older versions of HF Transformers seem to have issues running the model (e.g. resulting in an issue with the tokenizer). Using Transformers==4.13 might solve some issues. 

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=0.97&mnli_lp=nan&20_newsgroup=-0.39&ag_news=0.19&amazon_reviews_multi=0.10&anli=1.31&boolq=0.81&cb=8.93&cola=0.01&copa=13.60&dbpedia=-0.23&esnli=-0.51&financial_phrasebank=0.61&imdb=-0.26&isear=-0.35&mnli=-0.34&mrpc=1.24&multirc=1.50&poem_sentiment=-0.19&qnli=0.30&qqp=0.13&rotten_tomatoes=-0.55&rte=3.57&sst2=0.35&sst_5bins=0.39&stsb=1.10&trec_coarse=-0.36&trec_fine=-0.02&tweet_ev_emoji=1.11&tweet_ev_emotion=-0.35&tweet_ev_hate=1.43&tweet_ev_irony=-2.65&tweet_ev_offensive=-1.69&tweet_ev_sentiment=-1.51&wic=0.57&wnli=-2.61&wsc=9.95&yahoo_answers=-0.33&model_name=MoritzLaurer%2FDeBERTa-v3-base-mnli&base_name=microsoft%2Fdeberta-v3-base) using MoritzLaurer/DeBERTa-v3-base-mnli as a base model yields average score of 80.01 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 1st among all tested models for the microsoft/deberta-v3-base architecture as of 09/01/2023.

Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |   stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|-------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        86.0196 |   90.6333 |                  66.96 | 60.0938 |  83.792 | 83.9286 | 86.5772 |     72 |      79.2 |  91.419 |                   85.1 | 94.232 | 71.5124 | 89.4426 | 90.4412 |   63.7583 |          86.5385 | 93.8129 | 91.9144 |           89.8687 | 85.9206 | 95.4128 |     57.3756 | 91.377 |          97.4 |          91 |           47.302 |            83.6031 |         57.6431 |          77.1684 |              83.3721 |              70.2947 | 71.7868 | 67.6056 | 74.0385 |            71.7 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
