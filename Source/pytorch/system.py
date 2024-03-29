from torch.nn import Module
from torch import arange
from typing import List
from Source.pytorch.component import Component
from Source.pytorch.trigger import Trigger
from Source.pytorch.classifier_metadata import ClassifierMetadata, Split
from Source.pytorch.merger import Merger
from Source.pytorch.chain import Chain
from Source.io_util import read_pickle


class System(Module):

    def __init__(self, graph: Module, split=Split.TEST):
        super().__init__()
        self.graph = graph
        self.split = split
        self.set_evaluation_split(self.split)

    def forward(self, x=None):
        if len(self.get_classifiers()) == 0:
            raise ValueError("System empty of classifiers")

        if x is None:
            metadata = read_pickle(self.get_classifiers()[0].get_model())
            if self.split == Split.TRAIN:
                num_samples = len(metadata["train"]["id"])
                x = arange(num_samples).long()
            elif self.split == Split.TEST:
                num_samples = len(metadata["test"]["id"])
                x = arange(num_samples).long()
            elif self.split == Split.VAL:
                num_samples = len(metadata["val"]["id"])
                x = arange(num_samples).long()
            else:
                raise ValueError("Dataset split not recognized")

        return self.graph(x)

    def set_evaluation_split(self, split: Split):
        self.split = split
        for classifier in self.get_classifiers():
            classifier.set_evaluation_split(self.split)

    def set_start(self, start:Component):
        self.graph = start

    def get_evaluation_split(self):
        return self.split

    def get_components(self) -> List[Component]:
        component_list = []
        for module in self.graph.modules():
            if isinstance(module, Component):
                component_list.append(module)
        return component_list

    def get_classifiers(self) -> List[Component]:
        classifier_list = []
        for module in self.graph.modules():
            if isinstance(module, ClassifierMetadata):
                classifier_list.append(module)
        return classifier_list

    def get_mergers(self) -> List[Component]:
        merger_list = []
        for module in self.graph.modules():
            if isinstance(module, Merger):
                merger_list.append(module)
        return merger_list

    def get_triggers(self) -> List[Component]:
        trigger_list = []
        for module in self.graph.modules():
            if isinstance(module, Trigger):
                trigger_list.append(module)
        return trigger_list

    def get_chains(self) -> List[Component]:
        chain_list = []
        for module in self.graph.modules():
            if isinstance(module, Chain):
                chain_list.append(module)
        return chain_list

    def get_start(self):
        return self.graph

    def get_sysid(self):
        return hash(self)

    def get_num_parameters(self):
        numpa = 0
        for c in self.get_components():
            numpa += c.get_num_parameters()
        return numpa

    def get_processing_time(self):
        ptime = 0
        for c in self.get_components():
            ptime += c.get_processing_time()
        return ptime

    def __str__(self):
        return str(self.graph.modules())

    def __sizeof__(self):
        # Returns the number of Bytes of the ensemble object
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        return mem_params + mem_bufs

    def copy(self):
        raise ValueError("Not implemented")
