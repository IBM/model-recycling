import functools
import logging

import datasets as ds
from typing import Optional, Callable, Mapping, Sequence, Union, List
from utils.class_mapper import ClassMapper


class DatasetEncoder:
    def __init__(self, tokenizer, encoder_max_len=250, decoder_max_len=None,
                 sentence_names=None, extract_labels=None,
                 post_input=None, post_output=None, sentence_separator=' <SEP> ', prefix=""):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.post_input = post_input # important! if prefix != "" then we don't use the post_input
        self.post_output = post_output
        if post_output is None:
            if tokenizer.eos_token:
                self.post_output = tokenizer.eos_token
            else:
                self.post_output = ""
        if self.post_output and not self.post_output.startswith(' '):
            self.post_output = ' ' + self.post_output
        self.sentence_names = sentence_names
        self.extract_labels = extract_labels
        self.sentence_separator = sentence_separator
        self.prefix = prefix

    def example2text_batch(self, examples):
        sentences = []
        for sentence_name in self.sentence_names:
            sentences.append([])
            for example in examples[sentence_name]:
                sentences[-1].append(str(example))
        if self.prefix:
            return [self.prefix + self.sentence_separator.join(row) for row in zip(*sentences)]
        return [self.sentence_separator.join(row) + self.post_input for row in zip(*sentences)]

    def extract_formatted_labels_batch(self, examples):
        return self.format_label_batch(self.extract_labels(examples))

    def format_label_batch(self, labels):
        return [str(label) + self.post_output for label in labels]

    def deformat_label_batch(self, formatted_labels):
        return [label[:-len(self.post_output)] if label.endswith(self.post_output) and self.post_output else label for
                label in formatted_labels]

    def encode_batch(self, examples, padding=False):
        encoder_inputs = self.tokenizer(self.example2text_batch(examples), truncation=True,
                                        max_length=self.encoder_max_len,
                                        padding=padding)

        decoder_inputs = self.encode_labels(examples)['input_ids']

        encoder_inputs['labels'] = decoder_inputs
        return encoder_inputs

    def encode_labels(self, examples, padding=False):

        if self.decoder_max_len is None and padding == "max_length":
            raise RuntimeError(
                "Encoder decoding length was not set. Either add it to initialization, "
                "set it with set_decoder_max_len "
                "or calculate maximum label length from a dataset calculate_decoder_max_len ")

        with self.tokenizer.as_target_tokenizer():
            formatted_labels = self.extract_formatted_labels_batch(examples)
            decoder_inputs = self.tokenizer(formatted_labels, truncation=True,
                                            max_length=self.decoder_max_len,
                                            padding=padding)
        return decoder_inputs  # ['input_ids']

    def set_decoder_max_len(self, length: int):
        self.decoder_max_len = length
        self.logger.info(f"Setting decoding max length to: {self.decoder_max_len}")

    def calculate_decoder_max_len(self, sample_ds):
        self.decoder_max_len = None
        encoded_ds = ds.Dataset.map(sample_ds, functools.partial(self.encode_labels, padding=False), batched=True,
                                    desc="Running tokenizer on sample of dataset ",
                                    remove_columns=sample_ds.column_names)
        # check max label length
        decoder_max_len = 3
        for example in encoded_ds:
            decoder_max_len = max(decoder_max_len, len(example['input_ids']))
        self.set_decoder_max_len(decoder_max_len)


class DatasetEncoderForRegresor(DatasetEncoder):
    def format_label_batch(self, labels):
        return [float(label) for label in labels]

    def deformat_label_batch(self, formatted_labels):
        return [label for label in formatted_labels]

    def encode_labels(self, examples, padding=False):
        decoder_inputs = self.extract_formatted_labels_batch(examples)
        return {'input_ids': decoder_inputs}


class DatasetEncoderForClassifier(DatasetEncoder):
    def __init__(self, tokenizer, encoder_max_len=250, decoder_max_len=None,
                 sentence_names=None, extract_labels=None,
                 post_input=None, post_output=None, sentence_separator=' <SEP> ', labels_set=None, prefix=""):
        super().__init__(tokenizer, encoder_max_len, decoder_max_len,
                         sentence_names, extract_labels, post_input, post_output, sentence_separator, prefix=prefix)
        self.mapper = ClassMapper()
        self.mapper.add_examples(self.format_label_batch(labels_set))

    def encode_labels(self, examples, padding=False):
        decoder_inputs = self.mapper.convert(self.extract_formatted_labels_batch(examples))
        return {'input_ids': decoder_inputs}

    def deconvert(self, batch, mapped_classes: List[int]) -> List[str]:
        return self.mapper.deconvert(mapped_classes)
