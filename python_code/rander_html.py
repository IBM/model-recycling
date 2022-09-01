import numpy as np
import pandas as pd
import math
from IPython.display import HTML

roberta_absolute_scores_html_file_path = '../roberta_absolute_scores_table.html'
roberta_absolute_scores_models_csv_path = '../results/models_results_roberta_base.csv'
roberta_pretrain_scores_csv_path = '../results/models_results_roberta_pretrain.csv'
dropped_columns = ['base_model', 'size', 'tokenizer', 'model_type', 'classification', 'layers',
                   'from_flax', 'from_tf', 'last_modified']
columns_to_avg = ['cola', 'mrpc', 'qqp', 'stsb', 'boolq', 'cb', 'copa', 'multirc', 'wic', 'wsc', 'ag_news', 'isear',
                  'yahoo_answers', 'dbpedia', '20_newsgroup', 'trec_fine', 'trec_coarse', 'poem_sentiment', 'imdb',
                  'rotten_tomatoes', 'sst_5bins', 'sst2', 'amazon_reviews_multi', 'financial_phrasebank',
                  'tweet_ev_emoji', 'tweet_ev_emotion', 'tweet_ev_hate', 'tweet_ev_irony', 'tweet_ev_offensive',
                  'tweet_ev_sentiment', 'mnli', 'qnli', 'rte', 'wnli', 'esnli', 'anli']

html_prefix = """<!DOCTYPE html>
<html>
<head>
</head>
<body>
  <h1>Welcome to model-recycling page</h1>
  <p>This page contains ranking of HF models.</p>
  """

html_suffix = """
</body>
</html>
  """

if __name__ == '__main__':
    models_df = pd.read_csv(roberta_absolute_scores_models_csv_path)
    models_df = models_df.drop(columns=dropped_columns)
    models_df['avg'] = models_df.apply(lambda row:
                                       np.average([row[column] for column in columns_to_avg
                                                   if not math.isnan(row[column])]),
                                       axis=1)
    models_df = models_df[sorted(models_df.columns, key=lambda st: (st!='model_name', st!='avg', st!='mnli_lp', st))]
    models_df = models_df.sort_values(by=['avg', 'mnli_lp'], ascending=[False, False])
    models_df = models_df.reset_index(drop=True)
    models_df = models_df.style.format(precision=3)
    style = models_df.set_table_styles(
        [{"selector": "", "props": [("border", "1px solid grey")]},
         {"selector": "tbody td", "props": [("border", "1px solid grey")]},
         {"selector": "th", "props": [("border", "1px solid grey")]}
         ]
    )
    with open(roberta_absolute_scores_html_file_path, 'w') as f:
        f.write(html_prefix)
        f.write(HTML(style.render()).__html__())
        f.write(html_suffix)

    pretrain_df = pd.read_csv(roberta_pretrain_scores_csv_path, sep='\t')
    print()
