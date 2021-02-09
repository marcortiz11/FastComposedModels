import torch
from torch.nn import ModuleList
from torch.nn.functional import one_hot
from enum import Enum
from typing import List
from Source.pytorch.component import Component
from warnings import warn


class MergeProtocol(Enum):
    AVERAGE = 0
    VOTING = 1
    MAX = 2  # Also known as winner-takes-all
    WEIGHTED_AVERAGE = 3
    WEIGHTED_VOTING = 4
    WEIGHTED_MAX = 5


class Merger(Component):

    def protocol_average(self, x) -> torch.Tensor:
        for m in self.merged_modules:
            if self.output is None:
                self.output = m(x)
            else:
                self.output.add_(m(x))
        self.output.div_(len(self.merged_modules))
        return self.output

    def protocol_voting(self, x) -> torch.Tensor:
        warn("Some tensor operations in the voting merging do not support CUDA")

        n_classes = len(self.merged_modules[0](x[0]))
        tensors = tuple(m(x).argmax(dim=1)[:, None] for m in self.merged_modules)  # Assuming m(x) predictions 2-D tensor
        class_predictions = torch.cat(tensors, dim=1)
        voting = class_predictions.mode().values
        self.output = one_hot(voting, num_classes=n_classes).float()  # Returns a probability distribution
        return self.output

    def protocol_max(self, x) -> torch.Tensor:
        warn("Max protocol is uneficient, and does not support CUDA")

        tuple_pred = tuple(m(x) for m in self.merged_modules)
        stack_red = torch.stack(tuple_pred, dim=2)
        max_prob_pred = torch.max(stack_red, dim=1).values
        valid_pd = max_prob_pred.argmax(dim=1)
        # Fill the output with the valid distribution
        self.output = torch.empty_like(tuple_pred[0])
        for i in torch.unique(valid_pd):
            idx = torch.where(valid_pd == i)[0]
            self.output[idx, :] = tuple_pred[i][idx, :]
        return self.output

    def __init__(self, merged_modules: List[Component], protocol=MergeProtocol.AVERAGE):
        if len(merged_modules) == 0:
            raise ValueError("Number of merged components has to be > 0")

        super().__init__(p=1, t=0)
        self.protocol = protocol
        self.merged_modules = ModuleList(merged_modules)
        self.register_buffer("output", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.protocol == MergeProtocol.AVERAGE:
            return self.protocol_average(x)
        elif self.protocol == MergeProtocol.VOTING:
            return self.protocol_voting(x)
        elif self.protocol == MergeProtocol.MAX:
            return self.protocol_max(x)
        else:
            raise ValueError("Merging protocol not supported")

    def get_merge_protocol(self):
        return self.protocol

    def update_merge_protocol(self, p: MergeProtocol):
        self.protocol = p

    def add_classifier(self, c: Component):
        self.merged_modules.extend([c])
