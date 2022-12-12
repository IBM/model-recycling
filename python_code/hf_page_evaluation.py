import pickle
from datetime import date

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