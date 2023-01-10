import os
import pickle

import pandas as pd

from python_code.calc_pr_files import PR_DF, slash_mark, create_pr

md_files_path = 'md_files/with_fuse'


if __name__ == '__main__':
    pd_df = pd.read_csv(PR_DF)
    for filename in os.listdir(md_files_path):
        f = os.path.join(md_files_path, filename)
        try:
            model = f.split('/')[-1][:-3].replace(slash_mark, '/')
            ds_ = pd_df.loc[pd_df['model'] == model]
            arch = ds_['arch'].item()
            rank = ds_['rank'].item()
            if pd_df.loc[pd_df['model'] == model]['pr'].item() == 'False':
                print(f' model: {model}, arh: {arch}, rank: {rank}')
                create_pr(arch=arch, rank=rank, repo_id=model)
                ind = pd_df.loc[pd_df['model'] == model].index.item()
                print(f"pd_df['pr', {ind}]")
                pd_df.at[ind, 'pr'] = True
                pd_df.to_csv(PR_DF)
        except Exception as e:
            ind = pd_df.loc[pd_df['model'] == model].index.item()
            print(f"pd_df['pr', {ind}]")
            pd_df.at[ind, 'pr'] = 'Failed'
            pd_df.to_csv(PR_DF)
            print(f'Failed to PR {model}')
            print(e)