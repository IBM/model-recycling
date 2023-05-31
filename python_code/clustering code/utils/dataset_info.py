from enum import IntEnum
import datasets as ds

from utils.class_mapper import ClassMapper

STANDARD_SPLITS = ["train", "validation", "test"]

def get_split_names_mapping(train=None, validation=None, test=None):
    result = {split: split for split in STANDARD_SPLITS}
    if train is not None:
        result['train'] = train
    if validation is not None:
        result['validation'] = validation
    if test is not None:
        result['test'] = test
    return result


class Task(IntEnum):
    CLASSIFICATION = 0
    GENERATION = 1
    REGRESSION = 2
    # MULTILABEL_CLASSIFICATION = 3


class DatasetInfo:
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        self.name = name
        self.hf_name = hf_name
        if self.hf_name is None:
            self.hf_name = self.name
        if self.hf_name == '':
            self.hf_name = None
        self.path = path
        self.split_names_mapping = split_names_mapping
        if not self.split_names_mapping:
            self.split_names_mapping = get_split_names_mapping()
        self.sentence_names = sentence_names
        self.label_name = label_name
        self.task = task
        self.mapper = ClassMapper()

    def extract_label(self, data):
        return [str(element) for element in data[self.label_name]]

    def load_dataset(self, split):
        return ds.load_dataset(self.path, self.hf_name,
                                  split=self.split_names_mapping[split])

    def process_labels(self, labels):
        self.mapper.add_examples(labels)


class RegressionBy5BinsDatasetInfo(DatasetInfo):
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping,
                         sentence_names=sentence_names, label_name=label_name, task=task)

    def extract_label(self, data):
        return [str(int(element * 5)) for element in data[self.label_name]]


class TweetEvalDatasetInfo(DatasetInfo):
    def __init__(self, name, split_names_mapping=None, sentence_names=None,
                 label_name='label'):
        super().__init__(name=name, hf_name=name[len("tweet_ev_"):], path='tweet_eval', split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name)


class NliDatasetInfo(DatasetInfo):
    def __init__(self, name, path=None, split_names_mapping=None, label_name='label'):
        super().__init__(name=name, hf_name=None, path=path, split_names_mapping=split_names_mapping,
                         sentence_names=['premise', 'hypothesis'], label_name=label_name)
        self.hf_name = None


class SecretTestDatasetInfo(DatasetInfo):
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        if split_names_mapping is None:
            split_names_mapping = get_split_names_mapping(train='train[5%:95%]',
                                                          validation='train[:5%]+train[95%:]',
                                                          test='validation')
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)



class LargeSecretTestDatasetInfo(SecretTestDatasetInfo):  # large means 10% is larger than 1000 so datasets with 10K examples or more
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        if split_names_mapping is None:
            split_names_mapping = get_split_names_mapping(train='train[500:-500]',
                                                               validation='train[:500]+train[-500:]',
                                                               test='validation')
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)


class NoDevDatasetInfo(SecretTestDatasetInfo):
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        if split_names_mapping is None:
            split_names_mapping = get_split_names_mapping(train='train[5%:95%]',
                                                          validation='train[:5%]+train[95%:]',
                                                          test='test')
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)


class MultiTaskDatasetInfo(DatasetInfo):
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)
        self.tasks_map = None


class LargeNoDevDatasetInfo(LargeSecretTestDatasetInfo):  # large means 10% is larger than 1000 so datasets with 10K examples or more
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        if split_names_mapping is None:
            split_names_mapping = get_split_names_mapping(train='train[500:-500]',
                                                               validation='train[:500]+train[-500:]',
                                                               test='test')
        super().__init__(name=name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)


class CSVDatasetInfo(DatasetInfo):
    def __init__(self, name, data_files, hf_name=None, split_names_mapping=None, sentence_names=None,
                 label_name='label', task=Task.CLASSIFICATION):
        super().__init__(name=name, hf_name=hf_name, path='csv', split_names_mapping=split_names_mapping, sentence_names=sentence_names,
                         label_name=label_name, task=task)
        self.data_files = data_files

    def load_dataset(self, split):
        return ds.load_dataset(self.path, self.hf_name, data_files=self.data_files,
                                  split=self.split_names_mapping[split])


class SimpleGenerationDatasetInfo(DatasetInfo):
    def __init__(self, name, hf_name=None, path=None, split_names_mapping=None, sentence_names=None,
                 label_name='label'):
        super().__init__(name, hf_name=hf_name, path=path, split_names_mapping=split_names_mapping,
                       sentence_names=sentence_names, label_name=label_name, task=Task.GENERATION)

    def extract_label(self, data):
        return [l[0] for l in data[self.label_name]]

class UnitextGenerationDatasetInfo(DatasetInfo):
    def __init__(self, name, hf_name=None, split_names_mapping=None, sentence_names=None,
                 label_name='label'):
        super().__init__(name, hf_name=hf_name,
                         path='/u/eladv/unitext/src/unitext/resources.py', #resources.__file__,
                         split_names_mapping=split_names_mapping,
                         sentence_names=sentence_names, label_name=label_name, task=Task.GENERATION)
