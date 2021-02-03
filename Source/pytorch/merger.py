import torch
from enum import Enum
from typing import List
from Source.pytorch.component import Component


def protocol_average(x, merged_modules: torch.nn.ModuleList) -> torch.Tensor:
    Y = None
    for m in merged_modules:
        if Y is None:
            Y = m(x)
        else:
            Y += m(x)
    Y /= len(merged_modules)
    return Y


class MergeProtocol(Enum):
    AVERAGE = 0
    VOTING = 1
    MAX = 2
    WEIGHTED_AVERAGE = 3
    WEIGHTED_VOTING = 4
    WEIGHTED_MAX = 5


class Merger(Component):

    def __init__(self, merged_modules: List[torch.nn.Module], protocol=MergeProtocol.AVERAGE):
        assert protocol == MergeProtocol.AVERAGE, "Other merging protocols not currently supported"
        super().__init__(p=1, t=0)
        self.protocol = protocol
        self.merged_modules = torch.nn.ModuleList(merged_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = None
        if self.protocol == MergeProtocol.AVERAGE:
            pred = protocol_average(x, self.merged_modules)
        return pred
