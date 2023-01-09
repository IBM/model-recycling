import os
import pickle

import pandas as pd

from python_code.calc_pr_files import PR_DF, slash_mark
from python_code.hf_page_evaluation import records_file

md_files_path = 'md_files/with_fuse'


if __name__ == '__main__':
    with open(records_file, 'rb') as handle:
        records = pickle.load(handle)
    pd_df = pd.read_csv(PR_DF)
    for filename in os.listdir(md_files_path):
        f = os.path.join(md_files_path, filename)
        try:
            model = f.split('/')[-1][:-3].replace(slash_mark, '/')
            ds_ = pd_df.loc[pd_df['model'] == model]
            arch = ds_['arch'].item()
            rank = ds_['rank'].item()
            print(f' model: {model}, arh: {arch}, rank: {rank}')
            #         create_pr(arch=record['arch'], rank=record['i'], repo_id=record['model'])
            ind = pd_df.loc[pd_df['model'] == model].index.item()
            print(f"pd_df['pr', {ind}]")
            pd_df.at[ind, 'pr'] = True
            #         pd_df.to_csv(PR_DF)
        except Exception as e:
             print(f'Failed to PR {model}')
             print(e)
             raise e