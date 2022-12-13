import pickle

records = []
records_file = 'records.pkl'


def create_hf_model_page_evaluation_content_for_model(model_df, model_name, i, pretrain_avg, gain_chart_url):
    lines = model_df.drop(['avg', 'model_name', 'mnli_lp'], axis=1).to_markdown(index=False).split('\n')
    records.append({'arch': model_name, 'i': i, 'avg': model_df["avg"].item(), 'model': model_df["model_name"].item(),
                    'lines': lines, 'pretrain_avg': pretrain_avg, 'gain_chart_url': gain_chart_url})
    with open(records_file, 'wb') as handle:
        pickle.dump(records, handle)
