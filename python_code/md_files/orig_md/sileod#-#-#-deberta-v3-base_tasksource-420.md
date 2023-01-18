
# Model Card for DeBERTa-base-v3_tasksource-420

<span style="color:red">DEPRECIATED: Please see better version</span>: <https://huggingface.co/sileod/deberta-v3-base-tasksource-nli>

DeBERTa model jointly fine-tuned on 420 tasks of the tasksource collection https://github.com/sileod/tasksource/
This is the model with the MNLI classifier on top. Its encoder was trained on many datasets including bigbench, Anthropic/hh-rlhf... alongside many NLI and classification tasks with a SequenceClassification heads while using only one shared encoder.

### Software

https://github.com/sileod/tasknet/
Training took 3 days on 24GB gpu.

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=1.41&mnli_lp=nan&20_newsgroup=0.63&ag_news=0.46&amazon_reviews_multi=-0.40&anli=0.94&boolq=2.55&cb=10.71&cola=0.49&copa=10.60&dbpedia=0.10&esnli=-0.25&financial_phrasebank=1.31&imdb=-0.17&isear=0.63&mnli=0.42&mrpc=-0.23&multirc=1.73&poem_sentiment=0.77&qnli=0.12&qqp=-0.05&rotten_tomatoes=0.67&rte=2.13&sst2=0.01&sst_5bins=-0.02&stsb=1.39&trec_coarse=0.24&trec_fine=0.18&tweet_ev_emoji=0.62&tweet_ev_emotion=0.43&tweet_ev_hate=1.84&tweet_ev_irony=1.43&tweet_ev_offensive=0.17&tweet_ev_sentiment=0.08&wic=-1.78&wnli=3.03&wsc=9.95&yahoo_answers=0.17&model_name=sileod%2Fdeberta-v3-base_tasksource-420&base_name=microsoft%2Fdeberta-v3-base) using sileod/deberta-v3-base_tasksource-420 as a base model yields average score of 80.45 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 1st among all tested models for the microsoft/deberta-v3-base architecture as of 10/01/2023
Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|         87.042 |      90.9 |                  66.46 | 59.7188 | 85.5352 | 85.7143 | 87.0566 |     69 |   79.5333 | 91.6735 |                   85.8 | 94.324 | 72.4902 | 90.2055 | 88.9706 |   63.9851 |             87.5 | 93.6299 | 91.7363 |           91.0882 | 84.4765 | 95.0688 |     56.9683 | 91.6654 |            98 |        91.2 |           46.814 |            84.3772 |         58.0471 |            81.25 |              85.2326 |              71.8821 | 69.4357 | 73.2394 | 74.0385 |            72.2 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)

# Citation [optional]

**BibTeX:**

```bib
@misc{sileod23-tasksource,
  author = {Sileo, Damien},
  doi = {10.5281/zenodo.7473446},
  month = {01},
  title = {{tasksource: preprocessings for reproducibility and multitask-learning}},
  url = {https://github.com/sileod/tasksource},
  version = {1.5.0},
  year = {2023}}
```


# Model Card Contact

damien.sileo@inria.fr


</details>

