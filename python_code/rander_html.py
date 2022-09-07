import os

import numpy as np
import pandas as pd
import math
import re
# import tabulate

template_file_extension = 'tmp'
md_file_extension = 'md'
slot_symbol = '$$'
templates_dir_path = os.path.join(os.path.dirname(__file__), '..', './templates')
root_dir_path = os.path.join(os.path.dirname(__file__), '..')

roberta_base_table_csv = '../results/roberta_base_table.csv'
roberta_absolute_scores_models_csv_path = '../results/models_results_roberta_base.csv'
roberta_pretrain_scores_csv_path = '../results/models_results_roberta_pretrain.csv'

dropped_columns = ['base_model', 'size', 'tokenizer', 'model_type', 'classification', 'layers',
                   'from_flax', 'from_tf', 'last_modified']
columns_to_avg = ['cola', 'mrpc', 'qqp', 'stsb', 'boolq', 'cb', 'copa', 'multirc', 'wic', 'wsc', 'ag_news', 'isear',
                  'yahoo_answers', 'dbpedia', '20_newsgroup', 'trec_fine', 'trec_coarse', 'poem_sentiment', 'imdb',
                  'rotten_tomatoes', 'sst_5bins', 'sst2', 'amazon_reviews_multi', 'financial_phrasebank',
                  'tweet_ev_emoji', 'tweet_ev_emotion', 'tweet_ev_hate', 'tweet_ev_irony', 'tweet_ev_offensive',
                  'tweet_ev_sentiment', 'mnli', 'qnli', 'rte', 'wnli', 'esnli', 'anli']


def fill_templates(templates_dict):
    pattern = re.compile(re.escape(slot_symbol) +'(\w+)'+re.escape(slot_symbol))
    for root, dirs, filenames in os.walk(templates_dir_path):
        for filename in filenames:
            with open(os.path.join(root, filename)) as template_file:
                relative_path = root.replace(templates_dir_path, '').lstrip(os.sep)
                os.makedirs(os.path.join(root_dir_path, relative_path), exist_ok=True)
                with open(os.path.join(root_dir_path, relative_path,
                                       filename.replace('.'+template_file_extension, '.'+md_file_extension)), 'w') \
                        as md_file:
                    for line in template_file:
                        m = re.match(pattern, line)
                        if m:
                            md_file.write(line.replace(slot_symbol+m.group(1)+slot_symbol,
                                                       templates_dict[m.group(1)]))
                        else:
                            md_file.write(line)


def add_avg_and_sort_columns(df):
    df['avg'] = df.apply(lambda row:
                         np.average([row[column] for column in columns_to_avg
                                     if not math.isnan(row[column])]),
                         axis=1)
    df = df[sorted(df.columns, key=lambda st: (st != 'model_name', st != 'avg', st != 'mnli_lp', st))]
    return df


def df_to_md(df, path_to_csv_file=None):
    if path_to_csv_file:
        df.to_csv(path_to_csv_file)
    return df.to_markdown(floatfmt='.2f')


def calculate_template_dict():
    templates_dict = {}
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
    templates_dict['ROBERTA_BASE_PRETRAIN_TABLE'] = df_to_md(pretrain_df)

    models_df = pd.read_csv(roberta_absolute_scores_models_csv_path)
    cols = models_df.select_dtypes(np.number).columns
    models_df[cols] = models_df[cols].mul(100)
    models_df = models_df.drop(columns=dropped_columns)

    models_df = add_avg_and_sort_columns(models_df)
    models_df = models_df.sort_values(by=['avg', 'mnli_lp'], ascending=[False, False])
    models_df = models_df.reset_index(drop=True)

    models_df = pd.concat([models_df, pretrain_df.loc['mean'].to_frame().T], ignore_index=True)
    models_df = pd.concat([models_df.iloc[-1:], models_df.iloc[:-1]], ignore_index=True)
    models_df.at[0, 'model_name'] = 'roberta-base'
    templates_dict['ROBERTA_BASE_TABLE'] = df_to_md(models_df, roberta_base_table_csv)
    models_base_table_df = bold_non_baseline_rows(models_df.copy()[:11])
    templates_dict['ROBERTA_BASE_TABLE'] = df_to_md(models_base_table_df)


    models_df = models_df[['model_name', 'avg', 'mnli_lp']].iloc[0:6]
    models_df = bold_non_baseline_rows(models_df)
    templates_dict['ROBERTA_MODELS_SHORT'] = df_to_md(models_df)

    #models_df = models_df[['model_name', 'avg', 'mnli_lp']]
    #print_table_to_html(models_df, roberta_absolute_scores_avg_html_file_path)
    return templates_dict


def bold_non_baseline_rows(models_df):
    models_df[1:] = models_df[1:].applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else f'{x}')
    models_df[:1] = models_df[:1].applymap(lambda x: f'*{x:.2f}*' if isinstance(x, float) else f'*{x}*')
    models_df.index = ['*baseline*'] + ['' + str(i) + '' for i in range(1, len(models_df))]
    return models_df


def main():
    templates_dict = calculate_template_dict()
    fill_templates(templates_dict)


if __name__ == '__main__':
    main()