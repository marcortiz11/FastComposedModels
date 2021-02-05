import torch
from Source.pytorch.component import Component


class Trigger(Component):
    
    def __init__(self, th: float):
        super(Trigger, self).__init__()
        self.th = torch.nn.Parameter(torch.scalar_tensor(th))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probability = torch.nn.Softmax(dim=1)(logits)
        max_prob_class = torch.max(probability, dim=1)
        mask = torch.gt(max_prob_class.values, self.th)
        return mask

    def update_threshold(self, th: float):
        self.th = torch.nn.Parameter(torch.tensor(th))

    def get_threshold_value(self) -> float:
        return self.th.item()
