---
layout: default
nav_order: 3
---

# FAQ

## Why should I use a different base model than the vanilla pretrained model?

It improves results significantly in most cases, and in average. The best base RoBERTa-besa models improve results in 75% of the tasks we evaluated, with a median gain of 2.5 accuracy points. So if you had have to choose one base model, it would be best to use these top ranked models.

## Can I get worse results from training over the top ranked base model when compared to the vanilla model?

Yes. For example, in RoBERTa-base, about 1 in 4 tasks perform slightly better on the pretrained model. Furthermore, difference in seed randomization can yield variance in results. The best approach is to assess multiple models and evaluate on dev data. 

## When shouldn't I use one of the recommended base models?

You should always review the base model license and fact sheet to ensure they meet your requirement for the particular 
use case. you should always only  download models and datasets from sources that you trust.  Downloading of models and 
datasets can run code your machine (see for example [HuggingFace](https://huggingface.co/docs/transformers/autoclass_tutorial) warning). 
We do not certify the quality and usability of  models listed.

## Which architectures are supported?

In the initial version, Roberta-base models are tracked. Other architectures will be added soon. Want us to add a specific model? Please [contact us](contact_us.md) and say so. If you have recommended training parameters, it is even better, send them too. 

## Could you test my model?

Sure. If the architecture is not supported, see the above question. You can add it to [HuggingFace](https://huggingface.co/docs/transformers/model_sharing#use-the-pushtohub-function)  and wait.
<!--    Really impatient? You can [contact us](contact_us.md) we don't make any promise.-->


## How frequently do you update the leaderboard?

We will update the results monthly.

## How do you assess the models?

We train a linear probing classification head for the MNLI on each candidate model.  We take each of the top 5 ranking models, and we fine-tune them on the 36 classification tasks (Consisting of sentiment, NLI, Twitter, topic classification and other general classification tasks).   We compare to the baseline of the vanilla model which is also trained and assessed on 5 seeds.
We use the following hyperparameters:
>model name: roberta-base,
tokenizer: roberta-base,
train size: inf,
val size: inf,
test size: inf,
epochs: 10,
learning rate: 5e-5,linear,0.0006,
early stop epsilon: 0.001,
batch size: 256,
patience: 20 * 50 * 256,
validate every: 50 * 256,
seed: 0,
l2 reg: 0.0,
classification model: ,
optimizer: adamw,
weight decay: 0.01

## Which datasets are used?

We use the following datasets:
1. Entailments: [MNLI](https://cims.nyu.edu/~sbowman/multinli/), [ESNLI](https://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations), [QNLI](https://rajpurkar.github.io/SQuAD-explorer/), [QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs), [RTE](https://aclweb.org/anthology/W14-3110), [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html), [ANLI](https://aclanthology.org/2020.acl-main.441/) 
2. Sentiment:[SST-2](https://nlp.stanford.edu/sentiment/index.html)
[SST-5](https://nlp.stanford.edu/sentiment/index.html),
[POEM SENTIMENT](https://arxiv.org/abs/2011.02686),
[IMDB](https://ai.stanford.edu/~amaas/data/sentiment/),
[Rotten Tomatoes](https://aclanthology.org/P05-1015),
[Amazon reviews](https://jmcauley.ucsd.edu/data/amazon/),
[Financial phrasebank](https://arxiv.org/abs/1307.5336)
3. Topic Classification:
[AG NEWS](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html),
[ISEAR](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/),
[Yahoo answers](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf),
[DBpedia](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf),
[20 NEWSGROUP](http://qwone.com/~jason/20Newsgroups/),
[TREC fine](https://www.aclweb.org/anthology/C02-1150),
[TREC coarse](https://www.aclweb.org/anthology/H01-1069)
4. Twitter:
[Tweet Emoji](https://aclanthology.org/S18-1003),
[Tweet Emotion](https://aclanthology.org/S18-1001/),
[Tweet Hate](https://aclanthology.org/S19-2007/),
[Tweet Irony](https://aclanthology.org/S18-1005),
[Tweet Offensive](https://aclanthology.org/S19-2010/),
[Tweet Sentiment](https://aclanthology.org/S17-2088)

5. Others:
[CoLA](https://nyu-mll.github.io/CoLA/),
[STS-B](http://www.aclweb.org/anthology/S/S17/S17-2001),
[QQP](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs),
[QNLI](https://aclanthology.org/W18-5446/),
[RTE](https://aclweb.org/aclwiki/Recognizing_Textual_Entailment),
[WNLI](https://aclanthology.org/W18-5446/),
[MRPC](https://aclanthology.org/I05-5002),
[BoolQ](https://aclanthology.org/N19-1300/),
[CB](https://paperswithcode.com/dataset/ml-cb),
[COPA](https://paperswithcode.com/dataset/copa),
[WIC](https://aclanthology.org/N19-1128/),
[WSC](https://paperswithcode.com/dataset/wsc)

## I have another question.

Please [contact us](contact_us.md)
