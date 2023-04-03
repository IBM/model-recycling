---
license: apache-2.0
language: en
tags:
- deberta-v3-base
- deberta-v3
- deberta
- text-classification
- nli
- natural-language-inference
- multitask
- multi-task
- pipeline
- extreme-multi-task
- extreme-mtl
- tasksource
- zero-shot
- rlhf
pipeline_tag: zero-shot-classification
datasets:
- glue
- super_glue
- anli
- metaeval/babi_nli
- sick
- snli
- scitail
- hans
- alisawuffles/WANLI
- metaeval/recast
- sileod/probability_words_nli
- joey234/nan-nli
- pietrolesci/nli_fever
- pietrolesci/breaking_nli
- pietrolesci/conj_nli
- pietrolesci/fracas
- pietrolesci/dialogue_nli
- pietrolesci/mpe
- pietrolesci/dnc
- pietrolesci/gpt3_nli
- pietrolesci/recast_white
- pietrolesci/joci
- martn-nguyen/contrast_nli
- pietrolesci/robust_nli
- pietrolesci/robust_nli_is_sd
- pietrolesci/robust_nli_li_ts
- pietrolesci/gen_debiased_nli
- pietrolesci/add_one_rte
- metaeval/imppres
- pietrolesci/glue_diagnostics
- hlgd
- paws
- quora
- medical_questions_pairs
- conll2003
- Anthropic/hh-rlhf
- Anthropic/model-written-evals
- truthful_qa
- nightingal3/fig-qa
- tasksource/bigbench
- blimp
- cos_e
- cosmos_qa
- dream
- openbookqa
- qasc
- quartz
- quail
- head_qa
- sciq
- social_i_qa
- wiki_hop
- wiqa
- piqa
- hellaswag
- pkavumba/balanced-copa
- 12ml/e-CARE
- art
- tasksource/mmlu
- winogrande
- codah
- ai2_arc
- definite_pronoun_resolution
- swag
- math_qa
- metaeval/utilitarianism
- mteb/amazon_counterfactual
- SetFit/insincere-questions
- SetFit/toxic_conversations
- turingbench/TuringBench
- trec
- tals/vitaminc
- hope_edi
- strombergnlp/rumoureval_2019
- ethos
- tweet_eval
- discovery
- pragmeval
- silicone
- lex_glue
- papluca/language-identification
- imdb
- rotten_tomatoes
- ag_news
- yelp_review_full
- financial_phrasebank
- poem_sentiment
- dbpedia_14
- amazon_polarity
- app_reviews
- hate_speech18
- sms_spam
- humicroedit
- snips_built_in_intents
- banking77
- hate_speech_offensive
- yahoo_answers_topics
- pacovaldez/stackoverflow-questions
- zapsdcn/hyperpartisan_news
- zapsdcn/sciie
- zapsdcn/citation_intent
- go_emotions
- scicite
- liar
- relbert/lexical_relation_classification
- metaeval/linguisticprobing
- metaeval/crowdflower
- metaeval/ethics
- emo
- google_wellformed_query
- tweets_hate_speech_detection
- has_part
- wnut_17
- ncbi_disease
- acronym_identification
- jnlpba
- species_800
- SpeedOfMagic/ontonotes_english
- blog_authorship_corpus
- launch/open_question_type
- health_fact
- commonsense_qa
- mc_taco
- ade_corpus_v2
- prajjwal1/discosense
- circa
- YaHi/EffectiveFeedbackStudentWriting
- Ericwang/promptSentiment
- Ericwang/promptNLI
- Ericwang/promptSpoke
- Ericwang/promptProficiency
- Ericwang/promptGrammar
- Ericwang/promptCoherence
- PiC/phrase_similarity
- copenlu/scientific-exaggeration-detection
- quarel
- mwong/fever-evidence-related
- numer_sense
- dynabench/dynasent
- raquiba/Sarcasm_News_Headline
- sem_eval_2010_task_8
- demo-org/auditor_review
- medmcqa
- aqua_rat
- RuyuanWan/Dynasent_Disagreement
- RuyuanWan/Politeness_Disagreement
- RuyuanWan/SBIC_Disagreement
- RuyuanWan/SChem_Disagreement
- RuyuanWan/Dilemmas_Disagreement
- lucasmccabe/logiqa
- wiki_qa
- metaeval/cycic_classification
- metaeval/cycic_multiplechoice
- metaeval/sts-companion
- metaeval/commonsense_qa_2.0
- metaeval/lingnli
- metaeval/monotonicity-entailment
- metaeval/arct
- metaeval/scinli
- metaeval/naturallogic
- onestop_qa
- demelin/moral_stories
- corypaik/prost
- aps/dynahate
- metaeval/syntactic-augmentation-nli
- metaeval/autotnli
- lasha-nlp/CONDAQA
- openai/webgpt_comparisons
- Dahoas/synthetic-instruct-gptj-pairwise
- metaeval/scruples
- metaeval/wouldyourather
- sileod/attempto-nli
- metaeval/defeasible-nli
- metaeval/help-nli
- metaeval/nli-veridicality-transitivity
- metaeval/natural-language-satisfiability
- metaeval/lonli
- metaeval/dadc-limit-nli
- ColumbiaNLP/FLUTE
- metaeval/strategy-qa
- openai/summarize_from_feedback
- metaeval/folio
- metaeval/tomi-nli
- metaeval/avicenna
- stanfordnlp/SHP
- GBaker/MedQA-USMLE-4-options-hf
- sileod/wikimedqa
- declare-lab/cicero
- amydeng2000/CREAK
- metaeval/mutual
- inverse-scaling/NeQA
- inverse-scaling/quote-repetition
- inverse-scaling/redefine-math
- metaeval/puzzte
- metaeval/implicatures
- race
- metaeval/spartqa-yn
- metaeval/spartqa-mchoice
- metaeval/temporal-nli
metrics:
- accuracy
library_name: transformers
---

# Model Card for DeBERTa-v3-base-tasksource-nli

DeBERTa-v3-base fine-tuned with multi-task learning on 520 tasks of the [tasksource collection](https://github.com/sileod/tasksource/)
This checkpoint has strong zero-shot validation performance on many tasks (e.g. 70% on WNLI), and can be used for zero-shot NLI pipeline (similar to bart-mnli but better).
You can further fine-tune this model to use it for any classification or multiple-choice task.
The untuned model CLS embedding also has strong linear probing performance (90% on MNLI), due to the multitask training.

This is the shared model with the MNLI classifier on top. Its encoder was trained on many datasets including bigbench, Anthropic rlhf, anli... alongside many NLI and classification tasks with one shared encoder.
Each task had a specific CLS embedding, which is dropped 10% of the time to facilitate model use without it. All multiple-choice model used the same classification layers. For classification tasks, models shared weights if their labels matched.
The number of examples per task was capped to 64k. The model was trained for 45k steps with a batch size of 384, and a peak learning rate of 2e-5.

The list of tasks is available in model config.

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
Classifiers for all tasks available. See https://huggingface.co/sileod/deberta-v3-base-tasksource-adapters

<img src="https://www.dropbox.com/s/eyfw8i1ekzxj3fa/task_embeddings.png?dl=1"  width="1000" height="">


# Model Card Contact

damien.sileo@inria.fr


</details>