import torch
from Source.io_util import read_pickle
from Source.pytorch.component import Component
from Data.datasets import Split


class ClassifierMetadata(Component):

    def __init__(self, path_to_pickle: str, split=Split.VAL):
        # Call component's class constructor
        self.path = path_to_pickle
        self.split = split
        metadata = read_pickle(self.path)
        parameters = metadata['metrics']['params']
        super().__init__(p=parameters)
        # This way predictions can be manipulated on GPU
        self.register_buffer("predictions", None)

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
        self.update_processing_time(ids.numel() * time_batch_128/128.0)
        self.predictions = precomputed_pred[ids].to(ids.device)
        return self.predictions

    def set_evaluation_split(self, split: Split):
        self.split = split

    def update_model(self, path: str):
        self.path = path

    def get_model(self):
        return self.path

