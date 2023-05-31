
GLUE_SUPER_DATA_NUM2NAME = {'9': 'cb', '7': 'wnli', '5': 'qnli', '10': 'copa', '8': 'boolq', '11': 'multirc', '3': 'qqp',
                 '1': 'sst2', '12': 'wic', '6': 'rte', '4': 'mnli', '2': 'mrpc', '0': 'cola', '13': 'wsc'}

NLI_DATA_NUM2NAME = {'0': 'mnli', '1': 'qnli', '2': 'rte', '3': 'wnli', '4': 'esnli', '5': 'anli'}

# FULL_CLASSIFICATION_NUM2NAME = {'0': 'cola', '1': 'mrpc', '2': 'qqp', '3': 'stsb', '4': 'boolq', '5': 'cb', '6': 'copa',
#                                 '7': 'multirc', '8': 'wic', '9': 'wsc', '10': 'ag_news', '11': 'isear',
#                                 '12': 'yahoo_answers', '13': 'dbpedia', '14': '20_newsgroup', '15': 'trec_fine',
#                                 '16': 'trec_coarse', '17': 'poem_sentiment', '18': 'imdb', '19': 'rotten_tomatoes',
#                                 '20': 'sst_5bins', '21': 'sst2', '22': 'amazon_reviews_multi',
#                                 '23': 'financial_phrasebank', '24': 'tweet_ev_emoji', '25': 'tweet_ev_emotion',
#                                 '26': 'tweet_ev_hate', '27': 'tweet_ev_irony', '28': 'tweet_ev_offensive',
#                                 '29': 'tweet_ev_sentiment', '30': 'mnli', '31': 'qnli', '32': 'rte', '33': 'wnli',
#                                 '34': 'esnli', '35': 'anli'}

FULL_CLASSIFICATION_NUM2NAME = {'0': 'glue', '1': 'glue', '2': 'glue', '3': 'glue', '4': 'glue', '5': 'glue',
                                '6': 'glue', '7': 'glue', '8': 'glue', '9': 'glue', '10': 'topic', '11': 'topic',
                                '12': 'topic', '13': 'topic', '14': 'topic', '15': 'topic', '16': 'topic', '17': 'sent',
                                '18': 'sent', '19': 'sent', '20': 'sent', '21': 'sent', '22': 'sent', '23': 'sent',
                                '24': 'tweet', '25': 'tweet', '26': 'tweet', '27': 'tweet', '28': 'tweet',
                                '29': 'tweet', '30': 'nli', '31': 'nli', '32': 'nli', '33': 'nli', '34': 'nli',
                                '35': 'nli'}



DATA_NUM2NAME = {'GLUE_AND_SUPER_GLUE': GLUE_SUPER_DATA_NUM2NAME,
                 'GLUE_AND_SUPER_GLUE_NO_OUTLIERS': GLUE_SUPER_DATA_NUM2NAME, 'NLI': NLI_DATA_NUM2NAME,
                 'FULL_CLASSIFICATION': FULL_CLASSIFICATION_NUM2NAME}

num2name_full = {0: ('cola', 'glue'), 1: ('mrpc', 'glue'), 2: ('qqp', 'glue'), 3: ('stsb', 'glue'),
                 4: ('boolq', 'glue'), 5: ('cb', 'glue'), 6: ('copa', 'glue'), 7: ('multirc', 'glue'),
                 8: ('wic', 'glue'), 9: ('wsc', 'glue'), 10: ('ag_news', 'topic'), 11: ('isear', 'topic'),
                 12: ('yahoo_answers', 'topic'), 13: ('dbpedia', 'topic'), 14: ('20_newsgroup', 'topic'),
                 15: ('trec_fine', 'topic'), 16: ('trec_coarse', 'topic'), 17: ('poem_sentiment', 'sent'),
                 18: ('imdb', 'sent'), 19: ('rotten_tomatoes', 'sent'), 20: ('sst_5bins', 'sent'),
                 21: ('sst2', 'sent'), 22: ('amazon_reviews_multi', 'sent'), 23: ('financial_phrasebank', 'sent'),
                 24: ('tweet_ev_emoji', 'tweet'), 25: ('tweet_ev_emotion', 'tweet'), 26: ('tweet_ev_hate', 'tweet'),
                 27: ('tweet_ev_irony', 'tweet'), 28: ('tweet_ev_offensive', 'tweet'),
                 29: ('tweet_ev_sentiment', 'tweet'), 30: ('mnli', 'nli'), 31: ('qnli', 'nli'),
                 32: ('rte', 'nli'), 33: ('wnli', 'nli'), 34: ('esnli', 'nli'), 35: ('anli', 'nli')}

# num2domain = {}
# for i, e in enumerate(dataset_groups['FULL_CLASSIFICATION']):
#     if e in topic:
#         num2domain[i] = (e, 'topic')
#     elif e in tweet:
#         num2domain[i] = (e, 'tweet')
#     elif e in sent:
#         num2domain[i] = (e, 'sent')
#     elif e in nli:
#         num2domain[i] = (e, 'nli')
#     elif e in glue:
#         num2domain[i] = (e, 'glue')
#     else:
#         print(f'didnt fined domain for {e}')


DATA_NAME2SIZE = {'cb': 226, 'wnli': 571, 'qnli': 103743, 'copa': 360, 'boolq': 8485, 'multirc': 26243, 'qqp': 362846,
                  'sst2': 66349, 'wic': 4886, 'rte': 2242, 'mnli': 392702, 'mrpc': 3302, 'cola': 7695, 'wsc': 498}

NUM_BINS = 18000