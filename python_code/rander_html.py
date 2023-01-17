import os
import urllib
import numpy as np
import pandas as pd
import math
import re

from hf_page_evaluation import create_hf_model_page_evaluation_content_for_model

template_file_extension = 'tmp'
md_file_extension = 'md'
slot_symbol = '$$'
templates_dir_path = os.path.join(os.path.dirname(__file__), '..', './templates')
root_dir_path = os.path.join(os.path.dirname(__file__), '..')

dropped_columns = ['base_model', 'size', 'tokenizer', 'model_type', 'classification', 'layers',
                   'from_flax', 'from_tf', 'last_modified']
columns_to_avg = ['cola', 'mrpc', 'qqp', 'stsb', 'boolq', 'cb', 'copa', 'multirc', 'wic', 'wsc', 'ag_news', 'isear',
                  'yahoo_answers', 'dbpedia', '20_newsgroup', 'trec_fine', 'trec_coarse', 'poem_sentiment', 'imdb',
                  'rotten_tomatoes', 'sst_5bins', 'sst2', 'amazon_reviews_multi', 'financial_phrasebank',
                  'tweet_ev_emoji', 'tweet_ev_emotion', 'tweet_ev_hate', 'tweet_ev_irony', 'tweet_ev_offensive',
                  'tweet_ev_sentiment', 'mnli', 'qnli', 'rte', 'wnli', 'esnli', 'anli']

REPLACE_TEMPLATE_PATTERN = re.compile(f'(?={re.escape(slot_symbol)}([^$\s]+){re.escape(slot_symbol)})',
                                      flags=re.IGNORECASE)


def regularize_model_name(model_name):
    return model_name.replace('-', '_')


def get_base_table_path(model_name):
    return os.path.join(os.path.dirname(__file__), '..', 'results',
                        f'{escape_files_name(model_name)}_table.csv')


def get_absolute_scores_models_csv_path():
    return os.path.join(os.path.dirname(__file__), '..', 'results', "models_tested.csv")


def get_pretrain_scores_csv_path(model_name):
    return os.path.join(os.path.dirname(__file__), '..', 'results',
                        f'models_results_{regularize_model_name(model_name)}_pretrain.csv')


def fill_templates(templates_dict):
    for root, dirs, filenames in os.walk(templates_dir_path):
        for filename in filenames:
            m = re.search(REPLACE_TEMPLATE_PATTERN, filename)
            if m:  # the filename contains a template.
                resolving_list = templates_dict[m.group(1)]
                for resolve in resolving_list:
                    resolve = escape_files_name(resolve)
                    solved_filename = filename.replace(slot_symbol + m.group(1) + slot_symbol, resolve)
                    actual_templates_dict = templates_dict.copy()
                    actual_templates_dict[m.group(1)] = resolve
                    fill_file_content_by_template(root, filename, solved_filename, actual_templates_dict)
            else:
                fill_file_content_by_template(root, filename, filename, templates_dict)


def fill_file_content_by_template(root, filename, solved_filename, templates_dict):
    with open(os.path.join(root, filename)) as template_file:
        relative_path = root.replace(templates_dir_path, '').lstrip(os.sep)
        os.makedirs(os.path.join(root_dir_path, relative_path), exist_ok=True)
        with open(os.path.join(root_dir_path, relative_path,
                               solved_filename.replace('.' + template_file_extension, '.' + md_file_extension)), 'w') \
                as md_file:
            for line in template_file:
                while True:
                    ms = re.findall(REPLACE_TEMPLATE_PATTERN, line)
                    if not ms:
                        break
                    for m in ms:
                        if to_template_name(m) in templates_dict:
                            line = line.replace(slot_symbol + m + slot_symbol, str(templates_dict[to_template_name(m)]))
                            break
                    else:
                        raise RuntimeError(
                            f'Failed to find resolvable pattern in line\n Line: {line}\n. Available keys: {list(templates_dict.keys())}.\n\n line: {line}.')
                md_file.write(line)


def add_avg_and_sort_columns(df):
    df['avg'] = df.apply(lambda row:
                         np.average([row[column] for column in columns_to_avg
                                     if not math.isnan(row[column])]),
                         axis=1)
    df = df[sorted(df.columns, key=lambda st: (st != 'model_name', st != 'avg', st != 'mnli_lp', st))]
    return df


def df_to_md(df):
    return df.to_markdown(floatfmt='.2f')


def escape_files_name(st):
    return st.replace('/', '_')


def to_template_name(st):
    return regularize_model_name(escape_files_name(st).upper())


def calculate_model_template(model_name):
    reg_model_name = regularize_model_name(model_name)
    templates_dict = {}
    pretrain_df = pd.read_csv(get_pretrain_scores_csv_path(escape_files_name(reg_model_name)), sep='\t', index_col=0)
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
    templates_dict[f'{to_template_name(model_name)}_PRETRAIN_TABLE'] = df_to_md(pretrain_df)

    models_df = pd.read_csv(get_absolute_scores_models_csv_path())

    #models_df =
    models_df = models_df[models_df["base_model"] == model_name]
    rows_with_nans = models_df[models_df.apply(
        lambda row: not (not row.loc['mnli_lp':].iloc[1:].hasnans or row.loc['mnli_lp':].iloc[1:].isna().values.all()),
        axis=1)]
    if not rows_with_nans.empty:
        print(f'Warning: df contains models with nan values: {rows_with_nans}')
    models_df = models_df[models_df.apply(
        lambda row: not row.loc['mnli_lp':].iloc[1:].hasnans or row.loc['mnli_lp':].iloc[1:].isna().values.all(),
        axis=1)]
    templates_dict[f'{to_template_name(reg_model_name)}_SUCCESSFULLY_TESTED'] = len(models_df)
    templates_dict[f'{to_template_name(reg_model_name)}_SUCCESSFULLY_FULLY_TESTED'] = len(models_df.dropna())
    cols = models_df.select_dtypes(np.number).columns
    models_df[cols] = models_df[cols].mul(100)
    models_df = models_df.drop(columns=dropped_columns)

    models_df = add_avg_and_sort_columns(models_df)
    models_df = models_df.sort_values(by=['avg', 'mnli_lp'], ascending=[False, False])
    models_df = models_df.reset_index(drop=True)

    # TODO If a model has nans, only report its LP score
    # TODO. if lp score is 0.0 remove from results.
    models_df = pd.concat([models_df, pretrain_df.loc['mean'].to_frame().T], ignore_index=True)
    models_df = pd.concat([models_df.iloc[-1:], models_df.iloc[:-1]], ignore_index=True)
    models_df.at[0, 'model_name'] = model_name
    models_df.to_csv(get_base_table_path(model_name))

    models_df = models_df[models_df['model_name'].apply(lambda st: 'ibm/ColD-Fusion-itr' not in st)]
    models_df = models_df.reset_index(drop=True)

    def create_dict(row):
        result = (row.iloc[1:] - models_df.iloc[0, 1:]).to_dict()
        result = {k: f'{float(v):.2f}' for k, v in result.items()}
        result['model_name'] = models_df['model_name'][row.name]
        result['base_name'] = model_name
        return result

    gains_dict = models_df.apply(create_dict, axis=1)
    models_df_raw = models_df.copy()
    models_df['gain_chart_url'] = models_df.apply(
        lambda row: f'model_gain_chart?{urllib.parse.urlencode(gains_dict[row.name])}', axis=1)
    for i in range(1, 4):
        create_hf_model_page_evaluation_content_for_model(model_df=models_df_raw.iloc[i:i + 1],
                                                          gain_chart_url=models_df['gain_chart_url'].iloc[i:i + 1].item(),
                                                          model_name=model_name, i=i,
                                                          pretrain_avg=pretrain_df['avg']['mean'])
    models_df['model_name'] = models_df.apply(
        lambda row: f"[{row['model_name']}]({row['gain_chart_url']})", axis=1)
    models_df = models_df.drop('gain_chart_url', axis=1)
    models_df.at[0, 'model_name'] = f'[{model_name}]({model_name}_pretrain_scores_table)'
    models_base_table_df = bold_non_baseline_rows(models_df.copy()[:11])
    templates_dict[f'{to_template_name(reg_model_name)}_TABLE'] = df_to_md(models_base_table_df)

    templates_dict[f'{to_template_name(reg_model_name)}_BEST'] = models_df.iloc[:2]
    # TODO add number of models tested templates_dict[f'{model_name.upper()}_NUM_TESTED'] =
    return templates_dict


def calculate_template_dict():
    templates_dict = {}
    best_per_model = []
    best_cols = ("Pretrained", "Best model", "Avg.", "Pretrained Avg.", "Ranking")
    best = []
    minimum_tested = 5
    scores_df = pd.read_csv(get_absolute_scores_models_csv_path())
    templates_dict['BASE_NAME'] = []
    templates_dict['SUCCESSFULLY_TESTED'] = 0
    for model_name in scores_df['base_model'].unique().tolist():
        if len(scores_df[scores_df['base_model'] == model_name].dropna()) < minimum_tested:
            continue
        templates_dict['BASE_NAME'].append(model_name)
        templates_dict[f'{to_template_name(model_name)}_BASE_NAME'] = escape_files_name(model_name.replace('-', '_'))
        templates_dict.update(calculate_model_template(model_name))
        reg_model_name = regularize_model_name(model_name)
        pt = templates_dict[f'{to_template_name(reg_model_name)}_BEST'].iloc[0]
        best_model = templates_dict[f'{to_template_name(reg_model_name)}_BEST'].iloc[1]
        pt_values = pt["model_name"].split('(')

        best.append(('('.join([pt_values[0], escape_files_name(pt_values[1])]), best_model["model_name"],
                     best_model["avg"], pt["avg"], f'[link]({escape_files_name(model_name)}_table)'))
        templates_dict['SUCCESSFULLY_TESTED'] += int(
            templates_dict[f'{to_template_name(reg_model_name)}_SUCCESSFULLY_TESTED'])
        templates_dict[f'{to_template_name(model_name)}_COMMENTS'] = ''
    templates_dict['BEST_PER_MODEL'] = \
        pd.DataFrame(best, columns=best_cols).to_markdown(floatfmt='.2f', index=False)

    templates_dict[f'{to_template_name("roberta-base")}_COMMENTS'] = '1. ' \
                                                                     '[ColD Fusion](https://arxiv.org/abs/2212.01378) ' \
                                                                     'variations were removed to' \
                                                                     ' avoid cluttering the table'

    # models_df = models_df[['model_name', 'avg', 'mnli_lp']]
    # print_table_to_html(models_df, roberta_absolute_scores_avg_html_file_path)
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
