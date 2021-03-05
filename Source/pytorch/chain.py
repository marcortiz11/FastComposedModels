import torch
from typing import List
from Source.pytorch.classifier_metadata import ClassifierMetadata
from Source.pytorch.trigger import Trigger
from Source.pytorch.classifier import Classifier
from Source.pytorch.component import Component
import torch.autograd.profiler as profiler


class Chain(Component):

    def __init__(self, chained_modules: List[Component]):
        if len(chained_modules) == 0:
            raise ValueError("Chain should have length > 0")

        super().__init__()
        self.chained_modules = torch.nn.ModuleList(chained_modules)
        self.register_buffer("classify_mask", None)
        self.register_buffer("output", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not 0 < len(x.shape) < 5:
            raise ValueError("Input tensor x, should have 0 < dim < 4")
        if not x.shape[0] > 0:
            raise ValueError("Number of samples (dim=0) should be greater than 0")

        # Assuming x = [N, 1, h, w]
        self.classify_mask = torch.ones_like(x).bool()

        for c in self.chained_modules:
            if self.classify_mask.any():
                if isinstance(c, ClassifierMetadata) or isinstance(c, Classifier):
                    if self.output is None:
                        self.output = c(x)
                    else:
                        self.output[self.classify_mask] = c(x[self.classify_mask])
                elif isinstance(c, Trigger):
                        self.classify_mask[self.classify_mask] = c(self.output[self.classify_mask])
                else:
                    raise ValueError("Components forming the chain module should be of Classifier or Trigger type")

        return self.output

    def extend_chain(self, component_list: List[Component]):
        self.chained_modules.extend(component_list)

