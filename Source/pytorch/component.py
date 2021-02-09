import torch


class Component(torch.nn.Module):

    def __init__(self, t=0, p=0):
        super().__init__()
        self.time = t
        self.parameters = p

    def update_processing_time(self, t):
        self.time = t

    def update_num_parameters(self, p):
        self.parameters = p

    def get_processing_time(self):
        return self.time

    def get_num_parameters(self):
        return self.parameters
