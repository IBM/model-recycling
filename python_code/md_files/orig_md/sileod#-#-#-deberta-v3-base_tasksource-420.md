---
license: apache-2.0
language: en
tags:
- deberta-v3-base
- text-classification
pipeline_tag: zero-shot-classification
datasets:
- commonsense_qa
- liar
- definite_pronoun_resolution
- cosmos_qa
- medical_questions_pairs
- pietrolesci/add_one_rte
- quarel
- ctu-aic/enfever_nli
- strombergnlp/rumoureval_2019
- has_part
- rotten_tomatoes
- metaeval/linguisticprobing
- acronym_identification
- google_wellformed_query
- pietrolesci/joci
- sileod/probability_words_nli
- tweets_hate_speech_detection
- circa
- pietrolesci/conj_nli
- math_qa
- snli
- art
- joey234/nan-nli
- openbookqa
- pietrolesci/breaking_nli
- quail
- Anthropic/model-written-evals
- Ericwang/promptProficiency
- raquiba/Sarcasm_News_Headline
- martn-nguyen/contrast_nli
- hendrycks_test
- sms_spam
- hellaswag
- blimp
- banking77
- launch/open_question_type
- pietrolesci/fracas
- PiC/phrase_similarity
- pietrolesci/recast_white
- code_x_glue_cc_clone_detection_big_clone_bench
- pragmeval
- pietrolesci/robust_nli_li_ts
- tweet_eval
- Ericwang/promptNLI
- winogrande
- tals/vitaminc
- trec
- pietrolesci/dnc
- quora
- bigbench
- imdb
- Anthropic/hh-rlhf
- ncbi_disease
- prajjwal1/discosense
- scitail
- metaeval/recast
- humicroedit
- sciq
- health_fact
- anli
- yelp_review_full
- ethos
- relbert/lexical_relation_classification
- SpeedOfMagic/ontonotes_english
- qasc
- ai2_arc
- sick
- swag
- lex_glue
- copenlu/scientific-exaggeration-detection
- metaeval/babi_nli
- go_emotions
- jnlpba
- hyperpartisan_news_detection
- hans
- mc_taco
- ag_news
- social_i_qa
- metaeval/ethics
- conll2003
- Ericwang/promptSentiment
- glue
- hope_edi
- pietrolesci/dialogue_nli
- paws
- cos_e
- scicite
- metaeval/crowdflower
- emo
- snips_built_in_intents
- pietrolesci/gpt3_nli
- pietrolesci/robust_nli
- code_x_glue_cc_code_refinement
- wiqa
- hlgd
- hate_speech_offensive
- mwong/fever-evidence-related
- numer_sense
- ade_corpus_v2
- poem_sentiment
- Ericwang/promptCoherence
- financial_phrasebank
- pietrolesci/gen_debiased_nli
- wnut_17
- adv_glue
- blog_authorship_corpus
- dream
- wiki_hop
- Ericwang/promptSpoke
- codah
- quartz
- hate_speech18
- Ericwang/promptGrammar
- app_reviews
- code_x_glue_cc_defect_detection
- alisawuffles/WANLI
- pietrolesci/glue_diagnostics
- amazon_polarity
- dynabench/dynasent
- super_glue
- discovery
- species_800
- piqa
- dbpedia_14
- pietrolesci/robust_nli_is_sd
- YaHi/EffectiveFeedbackStudentWriting
---

# Model Card for DeBERTa-base-v3_tasksource-420

DeBERTa model jointly fine-tuned on 420 tasks of the tasksource collection https://github.com/sileod/tasksource/
This is the model with the MNLI classifier on top. Its encoder was trained on many datasets including bigbench, Anthropic/hh-rlhf... alongside many NLI and classification tasks with a SequenceClassification heads while using only one shared encoder.

Each task had a specific CLS embedding, which is dropped 10% of the time to facilitate model use without it. All multiple-choice model used the same classification layers. For classification tasks, models shared weights if their labels matched.
The number of examples per task was capped to 30k. The model was trained for 10k steps with a batch size of 384, a peak learning rate of 2e-5.

You can fine-tune this model to use it for multiple-choice or any classification task (e.g. NLI) like any debertav2 model. 
This model has strong validation performance on many tasks (e.g. 70% on WNLI).

The list of tasks is available in tasks.md

code: https://colab.research.google.com/drive/1iB4Oxl9_B5W3ZDzXoWJN-olUbqLBxgQS?usp=sharing

### Software

https://github.com/sileod/tasknet/
Training took 3 days on 24GB gpu.

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