import torch
from typing import List
from Source.pytorch.classifier_metadata import ClassifierMetadata
from Source.pytorch.trigger import Trigger
from Source.pytorch.component import Component


class Chain(Component):

    def __init__(self, chained_modules: List[Component]):
        if len(chained_modules) == 0:
            raise ValueError("Chain should have length > 0")

        super().__init__()
        self.chained_modules = torch.nn.ModuleList(chained_modules)
        self.register_buffer("x_local_ids", None)
        self.register_buffer("output", None)

    def forward(self, x):
        if not 0 < len(x.shape) < 5:
            raise ValueError("Input tensor x, should have 0 < dim < 4")
        if not x.shape[0] > 0:
            raise ValueError("Number of samples (dim=0) should be greater than 0")

        # Assuming x = [N, 1, h, w]
        self.x_local_ids = torch.arange(x.shape[0]-1, dtype=torch.long)

        for c in self.chained_modules:

            # Classifier component
            if isinstance(c, ClassifierMetadata):
                if self.output is None:
                    self.output = c(x)
                else:
                    self.output[self.x_local_ids] = c(x[self.x_local_ids])

            # Trigger component
            elif isinstance(c, Trigger):
                if self.output is None:
                    raise ValueError("Trigger should not be the first component of the chain")
                weakly_classified_mask = c(self.output[self.x_local_ids])
                x_sub_ids = weakly_classified_mask.nonzero(as_tuple=True)[0].long()
                # Early-exit condition
                if x_sub_ids.numel() > 0:
                    self.x_local_ids = self.x_local_ids[x_sub_ids]
                else:
                    return self.output
            else:
                raise ValueError("Components forming the chain module should be of Classifier or Trigger type")
        return self.output

    def extend_chain(self, component_list: List[Component]):
        self.chained_modules.extend(component_list)
