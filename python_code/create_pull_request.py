import os
import pickle
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, create_commit, CommitOperationAdd

from hf_page_evaluation import records_file

PR_DF = 'pr.df'

token = os.environ.get('HF_TOKEN')
readme_file = 'README.md'


def get_orig_file_path(model_name):
    Path("md_files/orig_md").mkdir(parents=True, exist_ok=True)
    return f'md_files/orig_md/{model_name.replace("/", "_")}.md'


def get_modified_file_path(model_name):
    Path("md_files/with_fuse").mkdir(parents=True, exist_ok=True)
    return f'md_files/with_fuse/{model_name.replace("/", "_")}.md'


def ordinal(n):
    if n > 3:
        return f'{n}th'
    if n == 3:
        return '3rd'
    if n == 2:
        return '2nd'
    if n == 1:
        return '1st'
    raise RuntimeError(f'Failed to find ordinal for {n}')


def find_evaluation_borders(lines):
    k = None
    for i, _ in enumerate(lines):
        if lines[i].rstrip() == '## Evaluation results':
            k = i + 1
        if '##' in lines[i] and k:
            break
    else:
        return i+1, i+1
    return i + 1, k


def replace_content(lines, repo_id, avg, pretrain_avg, arch, rank, table, replace_from, replace_to, gain_chart_url):
    replace_from, replace_to = find_evaluation_borders(lines)
    del lines[replace_from:replace_to]
    content = ['\n## Model Recycling', '',
               f'[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/{gain_chart_url}) '
               f'using {repo_id} as a base model '
               f'yields average score of {avg:.2f} in comparison to {pretrain_avg:.2f} by {arch}.',
               '',
               f'The model is ranked {ordinal(rank)} among all tested models for the {arch} '
               f'architecture as of {date.today().strftime("%d/%m/%Y")}'
               '',
               'Results:',
               ''
               ]
    content.extend(table)
    content.extend(['',
                    '',
                    'For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)'])

    for j, line in enumerate(content):
        lines.insert(j + replace_from, line + '\n')


def read_orig_md_file(repo_id):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=readme_file, force_download=True)
        with open(path) as file:
            lines = [line for line in file]
        with open(get_orig_file_path(model_name=repo_id), 'w') as file:
            file.writelines(lines)
    except Exception as e:
        print(f'failed to download md file for {repo_id}')
        print(e)


def update_md_file_with_fuse_results(repo_id, table, rank, arch, avg, pretrain_avg,
                                     force_rewrite=False,
                                     replace_from=None, replace_to=None, gain_chart_url=None
                                     ):
    try:
        with open(get_orig_file_path(model_name=repo_id)) as file:
            lines = [line for line in file]
    except:
        lines = [f'# {repo_id} model\n'
                 f'This model is based on {arch} pretrained model.\n\n']
    lines_with_link_to_webpage = [line for line in lines
                                  if '[Model Recycling](https://ibm.github.io/model-recycling/)' in line]
    already_have_model_recycling_section = len(lines_with_link_to_webpage) > 0
    if already_have_model_recycling_section and not force_rewrite:
        print(f'already_have_model_recycling_section: {already_have_model_recycling_section},  '
              f'force_rewrite: {force_rewrite}. Done')
        return

    replace_content(lines=lines, repo_id=repo_id, avg=avg, pretrain_avg=pretrain_avg,
                    arch=arch, rank=rank, table=table, replace_from=replace_from,
                    replace_to=replace_to, gain_chart_url=gain_chart_url)
    with open(get_modified_file_path(model_name=repo_id), 'w') as file:
        file.writelines(lines)


def create_pr(arch, rank, repo_id):
    operations = [CommitOperationAdd(path_in_repo=readme_file,
                                     path_or_fileobj=get_modified_file_path(model_name=repo_id))]
    commit_message = f'Evaluation results for {repo_id} model as a base model for other tasks '
    commit_description = 'As part of a research effort to identify high quality models in Huggingface ' \
                         'that can serve as base models for further finetuning, ' \
                         'we evaluated this by finetuning on 36 datasets. ' \
                         f'The model ranks {ordinal(rank)} among all tested models for the {arch} ' \
                         f'architecture as of {date.today().strftime("%d/%m/%Y")}.\n\n\n' \
                         f'To share this information with others in your model card, ' \
                         f'please add the following evaluation results to your README.md page.\n\n' \
                         f'For more information please see https://ibm.github.io/model-recycling/ ' \
                         f'or contact me.\n\n' \
                         f'Best regards,\n' \
                         f'Elad Venezian\n' \
                         f'eladv@il.ibm.com\n' \
                         f'IBM Research AI'
    create_commit(repo_id=repo_id, operations=operations,
                  commit_message=commit_message, commit_description=commit_description,
                  token=token, create_pr=True)


if __name__ == '__main__':
    with open(records_file, 'rb') as handle:
        records = pickle.load(handle)
    if os.path.exists(PR_DF):
        pd_df = pd.read_csv(PR_DF)
    else:
        data = {'model': [record['model'] for record in records],
                'rank': [record['i'] for record in records],
                'arch': [record['arch'] for record in records],
                'pr': [False for _ in records]}

        pd_df = pd.DataFrame(data)
        pd_df.to_csv(PR_DF)

    for i, record in enumerate(records):
        print(record)
        read_orig_md_file(repo_id=record['model'])
        # update_md_file_with_fuse_results(repo_id=record['model'], table=record['lines'],
        #                                  rank=record['i'], arch=record['arch'], avg=record['avg'],
        #                                  pretrain_avg=record['pretrain_avg'],
        #                                  gain_chart_url=record['gain_chart_url'],
        #                                  force_rewrite=False,
        #                                  replace_from=53,
        #                                  replace_to=68
        #                                  )
        if len(pd_df.loc[pd_df['model'] == record['model']]) != 1:
            raise RuntimeError(f"Find {(pd_df.loc[pd_df['model'] == record['model']])} rows at pd_df "
                               f"(expecting 1)")
        if pd_df.loc[pd_df['model'] == record['model']]['pr'].item() == False:
            try:
                create_pr(arch=record['arch'], rank=record['i'], repo_id=record['model'])
                ind = pd_df.loc[pd_df['model'] == record['model']].index.item()
                print(f"pd_df['pr', {ind}]")
                pd_df.at[ind, 'pr'] = True
                pd_df.to_csv(PR_DF)
            except Exception as e:
                print(f'Failed to PR {record["model"]}')
                print(e)
        else:
            print(f'Already PR {record["model"]}. Skipping')
    print()
