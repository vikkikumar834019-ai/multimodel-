import torch


class Text_Train_to_Deploy:

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
