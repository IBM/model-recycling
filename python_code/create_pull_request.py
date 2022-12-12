import os
import pickle
from datetime import date

from huggingface_hub import hf_hub_download, create_commit, CommitOperationAdd

token = os.environ.get('HF_TOKEN')
readme_file = 'README.md'
temp_file_path = 'temp.md'


def open_pull_request_for_model_card(repo_id, content, rank, arch, do_create_commit=False):
    path = hf_hub_download(repo_id=repo_id, filename=readme_file, force_download=True)
    with open(path) as file:
        lines = [line for line in file]

    for i, _ in enumerate(lines):
        if lines[i].rstrip() == '## Evaluation results':
            k = i + 1
            while lines[k].rstrip() != '':
                k += 1
            del lines[i + 1:k]
            for j, line in enumerate(content):
                lines.insert(j + i + 1, line+'\n')

    with open(temp_file_path, 'w') as file:
        file.writelines(lines)

    if do_create_commit:
        operations = [CommitOperationAdd(path_in_repo=readme_file,
                                         path_or_fileobj=temp_file_path)]

        commit_message = 'We find your model to be a great base-model'
        commit_description = f'We find your model to be the {rank}th best base model over {arch} architecture.\n ' \
                             f'(Means that using your model as a starting point for finetuning is great)\n ' \
                             f'We suggest to add the following Evaluation to your README.md page.\n ' \
                             f'For any question please contact eladv@il.ibm.com'


        create_commit(repo_id=repo_id, operations=operations,
                      commit_message=commit_message, commit_description=commit_description,
                      token=token, create_pr=True)


records = []
records_file = 'records.pkl'


def create_hf_model_page_evaluation_content_for_model(model_df, model_name, i):
    lines = []
    lines.append(f'Evaluation on 36 dataset using {model_df["model_name"].item()} as a base model, yield average score '
                 f'of {model_df["avg"].item():.2f}.')
    lines.append(f'According to [website](https://ibm.github.io/model-recycling/), this is the {i}th best model for '
                 f'{model_name} models (updated to {date.today().strftime("%d/%m/%Y")})')
    lines.append('')
    lines.append('Results:')
    lines.append('')
    lines.extend(model_df.drop(['avg', 'model_name', 'mnli_lp'], axis=1).to_markdown(index=False).split('\n'))

    records.append({'arch': model_name, 'i':i, 'model': model_df["model_name"].item(), 'lines':lines})
    with open(records_file, 'wb') as handle:
        pickle.dump(records, handle)


if __name__ == '__main__':
    #open_pull_request_for_model_card()
    with open(records_file, 'rb') as handle:
        records = pickle.load(handle)
    for i, record in enumerate(records):
        print(record)
        open_pull_request_for_model_card(repo_id=record['model'], content=record['lines'],
                                         rank=record['i'], arch=record['arch'],
                                         do_create_commit=True
                                         )
        if i == 1:
            break
