import numpy as np
import pandas as pd
import math
import tabulate

roberta_absolute_scores_html_file_path = '../roberta_absolute_scores_table.html'
roberta_absolute_scores_avg_html_file_path = '../roberta_absolute_scores_avg_table.html'
pretrain_scores_html_file_path = '../pretrain_scores_table.md'
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


def print_table_to_html(df, html_file_path):
    pd.options.display.float_format = '{:,.2f}'.format
    with open(html_file_path, 'w') as f:
        f.write(df.to_markdown(floatfmt='.2f'))


def add_avg_and_sort_columns(df):
    df['avg'] = df.apply(lambda row:
                         np.average([row[column] for column in columns_to_avg
                                     if not math.isnan(row[column])]),
                         axis=1)
    df = df[sorted(df.columns, key=lambda st: (st != 'model_name', st != 'avg', st != 'mnli_lp', st))]
    return df


if __name__ == '__main__':
    pretrain_df = pd.read_csv(roberta_pretrain_scores_csv_path, sep='\t')
    pretrain_df['score'] = pretrain_df.apply(
        lambda row: row['accuracy'] if not math.isnan(row['accuracy']) else row['spearmanr'], axis=1)
    pretrain_df['score'] = pretrain_df['score'].apply(lambda val: 100 * val)

    avg_pretrain_df = pretrain_df.groupby('stage').agg(np.mean)
    std_df = pretrain_df.groupby('dataset name').agg(np.std)
    mean_df = pretrain_df.groupby('dataset name').agg(np.mean)
    pretrain_df = pd.concat([pd.pivot_table(mean_df, values=["score"], columns=['dataset name']),
                             pd.pivot_table(std_df, values=["score"], columns=['dataset name'])])
    pretrain_df.index = ['mean', 'std']
    pretrain_df = add_avg_and_sort_columns(pretrain_df)
    pretrain_df.at['std', 'avg'] = avg_pretrain_df.std(axis=0)['score']
    print_table_to_html(pretrain_df, pretrain_scores_html_file_path)

    models_df = pd.read_csv(roberta_absolute_scores_models_csv_path)
    cols = models_df.select_dtypes(np.number).columns
    models_df[cols] = models_df[cols].mul(100)
    models_df = models_df.drop(columns=dropped_columns)

    models_df = add_avg_and_sort_columns(models_df)
    models_df = models_df.sort_values(by=['avg', 'mnli_lp'], ascending=[False, False])
    models_df = models_df.reset_index(drop=True)

    models_df = pd.concat([models_df, pretrain_df.loc['mean'].to_frame().T], ignore_index=True)
    models_df = pd.concat([models_df.iloc[-1:], models_df.iloc[:-1]], ignore_index=True)
    models_df.at[0, 'model_name'] = 'Pretrained Model'

    print_table_to_html(models_df, roberta_absolute_scores_html_file_path)

    models_df = models_df[['model_name', 'avg', 'mnli_lp']]
    print_table_to_html(models_df, roberta_absolute_scores_avg_html_file_path)
