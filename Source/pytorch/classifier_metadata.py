import torch
from Source.io_util import read_pickle
from Source.pytorch.component import Component
from enum import Enum


class Split(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


class ClassifierMetadata(Component):

    def __init__(self, path_to_pickle: str, split=Split.VAL):
        self.path = path_to_pickle
        self.split = split
        # This way predictions can be manipulated on GPU
        # self.register_buffer("predictions", None)
        # Call component's class constructor
        metadata = read_pickle(self.path)
        parameters = metadata['metrics']['params']
        super().__init__(p=parameters)

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        Receives the indices of a set of samples, and returns the precomputed predictions of those samples
        :param ids: Index of samples in the dataset
        :return: Predictions of the classifier for those indices
        """
        metadata = read_pickle(self.path)
        time_batch_128 = metadata['metrics']['time']
        if self.split == Split.TRAIN:
            precomputed_pred = metadata['train']['logits']
        elif self.split == Split.TEST:
            precomputed_pred = metadata['test']['logits']
        else:
            precomputed_pred = metadata['val']['logits']
        precomputed_pred = torch.from_numpy(precomputed_pred)
        self.update_processing_time(torch.numel(ids) * time_batch_128/128.0)
        return precomputed_pred[ids]

    def set_evaluation_split(self, split: Split):
        self.split = split

    def update_model(self, path: str):
        self.path = path

    def get_model(self):
        return self.path
