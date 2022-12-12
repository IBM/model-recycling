import os
import pickle

from huggingface_hub import hf_hub_download, create_commit, CommitOperationAdd
from hf_page_evaluation import records_file

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
