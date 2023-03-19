import torch
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self):
        self.data = torch.rand((2, 3, 256, 256))
        self.mask = torch.ones((2, 1, 256, 256))

        self.input_tensor = torch.cat([self.data, self.mask], dim=1)
        self.inception_embeds = torch.zeros((2, 1000))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "input_tensor": self.input_tensor[item],
            "inception_embeds": self.inception_embeds[item]
        }
