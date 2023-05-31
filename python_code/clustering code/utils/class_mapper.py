from typing import List


class ClassMapper:
    """
    A class that gets predictions and map them to numeral classes (so e.g. negative->0 positive ->1 neutral->3
    """

    def __init__(self):
        self.examples_set = set()
        self.class2int = {}
        self.reversed_map = {}
        self.inference_mode = False

    def add_examples(self, examples):
        if self.inference_mode:
            raise RuntimeError("Can't add example after convert.")
        self.examples_set.update(examples)

    def convert(self, predictions: List) -> List[int]:
        if not self.inference_mode:
            examples = sorted(str(example) for example in self.examples_set)
            self.class2int = {example: i for i, example in enumerate(examples)}
            self.reversed_map = {v: k for k, v in self.class2int.items()}
        self.inference_mode = True
        # return result
        unknown_label = max(self.class2int.values()) + 1
        res = [self.class2int.get(str(prediction).strip(), unknown_label) for prediction in predictions]
        return res

    def deconvert(self, mapped_classes: List[int]) -> List[str]:
        if not self.inference_mode:
            raise RuntimeError("Can't deconvert before converting")
        return [self.reversed_map[m] for m in mapped_classes]


class MultiUseClassMapper(ClassMapper):
    def add_examples(self, examples):
        self.examples_set.update(examples)
