---
license: apache-2.0
language: en
tags:
- deberta-v3-base
- text-classification
- nli
- natural-language-inference
- multitask
- multi-task
- pipeline
- extreme-multi-task
- extreme-mtl
- deberta-v3-base
- tasksource
- zero-shot
- rlhf
pipeline_tag: zero-shot-classification
datasets:
- hellaswag
- ag_news
- pietrolesci/nli_fever
- numer_sense
- go_emotions
- Ericwang/promptProficiency
- poem_sentiment
- pietrolesci/robust_nli_is_sd
- sileod/probability_words_nli
- social_i_qa
- trec
- imppres
- pietrolesci/gen_debiased_nli
- snips_built_in_intents
- metaeval/imppres
- metaeval/crowdflower
- tals/vitaminc
- dream
- metaeval/babi_nli
- Ericwang/promptSpoke
- metaeval/ethics
- art
- ai2_arc
- discovery
- Ericwang/promptGrammar
- code_x_glue_cc_clone_detection_big_clone_bench
- prajjwal1/discosense
- pietrolesci/joci
- Anthropic/model-written-evals
- utilitarianism
- emo
- tweets_hate_speech_detection
- piqa
- blog_authorship_corpus
- SpeedOfMagic/ontonotes_english
- circa
- app_reviews
- anli
- Ericwang/promptSentiment
- codah
- definite_pronoun_resolution
- health_fact
- tweet_eval
- hate_speech18
- glue
- hendrycks_test
- paws
- bigbench
- hate_speech_offensive
- blimp
- sick
- turingbench/TuringBench
- martn-nguyen/contrast_nli
- Anthropic/hh-rlhf
- openbookqa
- species_800
- alisawuffles/WANLI
- ethos
- pietrolesci/mpe
- wiki_hop
- pietrolesci/glue_diagnostics
- mc_taco
- quarel
- PiC/phrase_similarity
- strombergnlp/rumoureval_2019
- quail
- acronym_identification
- pietrolesci/robust_nli
- quora
- wnut_17
- dynabench/dynasent
- pietrolesci/gpt3_nli
- truthful_qa
- pietrolesci/add_one_rte
- pietrolesci/breaking_nli
- copenlu/scientific-exaggeration-detection
- medical_questions_pairs
- rotten_tomatoes
- scicite
- scitail
- pietrolesci/dialogue_nli
- code_x_glue_cc_defect_detection
- nightingal3/fig-qa
- pietrolesci/conj_nli
- liar
- sciq
- head_qa
- pietrolesci/dnc
- quartz
- wiqa
- code_x_glue_cc_code_refinement
- Ericwang/promptCoherence
- joey234/nan-nli
- hope_edi
- jnlpba
- yelp_review_full
- pietrolesci/recast_white
- swag
- banking77
- cosmos_qa
- financial_phrasebank
- hans
- pietrolesci/fracas
- math_qa
- conll2003
- qasc
- ncbi_disease
- mwong/fever-evidence-related
- YaHi/EffectiveFeedbackStudentWriting
- ade_corpus_v2
- amazon_polarity
- pietrolesci/robust_nli_li_ts
- super_glue
- adv_glue
- Ericwang/promptNLI
- cos_e
- launch/open_question_type
- lex_glue
- has_part
- pragmeval
- sem_eval_2010_task_8
- imdb
- humicroedit
- sms_spam
- dbpedia_14
- commonsense_qa
- hlgd
- snli
- hyperpartisan_news_detection
- google_wellformed_query
- raquiba/Sarcasm_News_Headline
- metaeval/recast
- winogrande
- relbert/lexical_relation_classification
- metaeval/linguisticprobing
metrics:
- accuracy
library_name: transformers
---

# Model Card for DeBERTa-v3-base-tasksource-nli

DeBERTa-v3-base fine-tuned with multi-task learning on 444 tasks of the [tasksource collection](https://github.com/sileod/tasksource/)
You can further fine-tune this model to use it for any classification or multiple-choice task.
This checkpoint has strong zero-shot validation performance on many tasks (e.g. 70% on WNLI).
The untuned model CLS embedding also has strong linear probing performance (90% on MNLI), due to the multitask training.

This is the shared model with the MNLI classifier on top. Its encoder was trained on many datasets including bigbench, Anthropic rlhf, anli... alongside many NLI and classification tasks with a SequenceClassification heads while using only one shared encoder.
Each task had a specific CLS embedding, which is dropped 10% of the time to facilitate model use without it. All multiple-choice model used the same classification layers. For classification tasks, models shared weights if their labels matched.
The number of examples per task was capped to 64k. The model was trained for 20k steps with a batch size of 384, and a peak learning rate of 2e-5.

The list of tasks is available in tasks.md

tasksource training code: https://colab.research.google.com/drive/1iB4Oxl9_B5W3ZDzXoWJN-olUbqLBxgQS?usp=sharing

### Software
https://github.com/sileod/tasksource/ \
https://github.com/sileod/tasknet/ \
Training took 7 days on RTX6000 24GB gpu.

## Model Recycling
An earlier (weaker) version model is ranked 1st among all models with the microsoft/deberta-v3-base architecture as of 10/01/2023
Results:
[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=1.41&mnli_lp=nan&20_newsgroup=0.63&ag_news=0.46&amazon_reviews_multi=-0.40&anli=0.94&boolq=2.55&cb=10.71&cola=0.49&copa=10.60&dbpedia=0.10&esnli=-0.25&financial_phrasebank=1.31&imdb=-0.17&isear=0.63&mnli=0.42&mrpc=-0.23&multirc=1.73&poem_sentiment=0.77&qnli=0.12&qqp=-0.05&rotten_tomatoes=0.67&rte=2.13&sst2=0.01&sst_5bins=-0.02&stsb=1.39&trec_coarse=0.24&trec_fine=0.18&tweet_ev_emoji=0.62&tweet_ev_emotion=0.43&tweet_ev_hate=1.84&tweet_ev_irony=1.43&tweet_ev_offensive=0.17&tweet_ev_sentiment=0.08&wic=-1.78&wnli=3.03&wsc=9.95&yahoo_answers=0.17&model_name=sileod%2Fdeberta-v3-base_tasksource-420&base_name=microsoft%2Fdeberta-v3-base) using sileod/deberta-v3-base_tasksource-420 as a base model yields average score of 80.45 in comparison to 79.04 by microsoft/deberta-v3-base.



|   20_newsgroup |   ag_news |   amazon_reviews_multi |    anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |    qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|--------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|--------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|         87.042 |      90.9 |                  66.46 | 59.7188 | 85.5352 | 85.7143 | 87.0566 |     69 |   79.5333 | 91.6735 |                   85.8 | 94.324 | 72.4902 | 90.2055 | 88.9706 |   63.9851 |             87.5 | 93.6299 | 91.7363 |           91.0882 | 84.4765 | 95.0688 |     56.9683 | 91.6654 |            98 |        91.2 |           46.814 |            84.3772 |         58.0471 |            81.25 |              85.2326 |              71.8821 | 69.4357 | 73.2394 | 74.0385 |            72.2 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)


# Citation

More details on this [article:](https://arxiv.org/abs/2301.05948) 
```bib
@article{sileo2023tasksource,
  title={tasksource: Structured Dataset Preprocessing Annotations for Frictionless Extreme Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  url= {https://arxiv.org/abs/2301.05948},
  journal={arXiv preprint arXiv:2301.05948},
  year={2023}
}
```

# Loading a specific classifier

Classifiers for all tasks available.
```python
from torch import nn

TASK_NAME = "hh-rlhf"

class MultiTask(transformers.DebertaV2ForMultipleChoice):
   def __init__(self, *args, **kwargs):
        super().__init__(*args)
        n=len(self.config.tasks)
        cs=self.config.classifiers_size
        self.Z = nn.Embedding(n,768)
        self.classifiers = nn.ModuleList([torch.nn.Linear(*size) for size in cs])

model = MultiTask.from_pretrained("sileod/deberta-v3-base-tasksource-nli",ignore_mismatched_sizes=True)
task_index = {k:v for v,k in dict(enumerate(model.config.tasks)).items()}[TASK_NAME]
model.classifier = model.classifiers[task_index] # model is ready for $TASK_NAME ! (RLHF) ! 
```


# Model Card Contact

damien.sileo@inria.fr


</details>