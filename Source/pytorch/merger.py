import torch
from enum import Enum
from typing import List
from Source.pytorch.component import Component
from warnings import warn


def protocol_average(x, merged_modules: torch.nn.ModuleList) -> torch.Tensor:
    Y = torch.tensor(None)
    for m in merged_modules:
        if Y is None:
            Y = m(x)
        else:
            Y.add_(m(x))  # In-place operations
    Y /= len(merged_modules)
    return Y


def protocol_voting(x, merged_modules: torch.nn.ModuleList) -> torch.Tensor:
    warn("Some tensor operations in the voting merging do not support CUDA")

    n_classes = len(merged_modules[0](x[0]))
    tensors = tuple(m(x).argmax(dim=1)[:, None] for m in merged_modules)  # Assuming m(x) predictions 2-D tensor
    class_predictions = torch.cat(tensors, dim=1)  # row tensor predictions concatenated along axis 1
    voting = class_predictions.mode().values
    Y = torch.nn.functional.one_hot(voting, num_classes=n_classes)  # Returns a probability distribution
    return Y


def protocol_max(x, merged_modules: torch.nn.ModuleList) -> torch.Tensor:
    warn("Max protocol is uneficient, and does not support CUDA")

    tuple_pred = tuple(m(x) for m in merged_modules)
    stack_red = torch.stack(tuple_pred, dim=2)
    max_prob_pred = torch.max(stack_red, dim=1).values
    valid_pd = max_prob_pred.argmax(dim=1)
    Y = torch.empty_like(tuple_pred[0])
    for i in range(x.shape[0]):
        Y[i, :] = tuple_pred[valid_pd[i]][i]

    return Y


class MergeProtocol(Enum):
    AVERAGE = 0
    VOTING = 1
    MAX = 2  # Also known as winner-takes-all
    WEIGHTED_AVERAGE = 3
    WEIGHTED_VOTING = 4
    WEIGHTED_MAX = 5


class Merger(Component):

    def __init__(self, merged_modules: List[Component], protocol=MergeProtocol.AVERAGE):
        if len(merged_modules) == 0:
            raise ValueError("Number of merged components has to be > 0")

        super().__init__(p=1, t=0)
        self.protocol = protocol
        self.merged_modules = torch.nn.ModuleList(merged_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = None
        if self.protocol == MergeProtocol.AVERAGE:
            pred = protocol_average(x, self.merged_modules)
        elif self.protocol == MergeProtocol.VOTING:
            pred = protocol_voting(x, self.merged_modules)
        elif self.protocol == MergeProtocol.MAX:
            pred = protocol_max(x, self.merged_modules)
        return pred

    def get_merge_protocol(self):
        return self.protocol

    def update_merge_protocol(self, p: MergeProtocol):
        self.protocol = p

    def add_classifier(self, c: Component):
        self.merged_modules.extend([c])
