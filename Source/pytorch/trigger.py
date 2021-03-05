import torch
from Source.pytorch.component import Component


class Trigger(Component):
    
    def __init__(self, th: float):
        super(Trigger, self).__init__()
        self.th = torch.nn.Parameter(torch.scalar_tensor(th))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        max_prob_class = torch.nn.Softmax(dim=1)(logits).max(1)
        mask = torch.le(max_prob_class.values, self.th).bool()
        return mask  # mask=1 classify again; mask=0 classification okay

    def update_threshold(self, th: float):
        self.th = torch.nn.Parameter(torch.tensor(th))

    def get_threshold_value(self) -> float:
        return self.th.item()
