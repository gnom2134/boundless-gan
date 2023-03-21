from torchvision.datasets.places365 import Places365
from pathlib import Path

import numpy as np
import torch


class Places365Embedding(Places365):
    def __init__(self, embeddings_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = torch.from_numpy(np.load(str(embeddings_path)))

    def __getitem__(self, item):
        img, _ = super(Places365Embedding, self).__getitem__(item)
        embedding = self.embeddings[item, :]
        return {
            "input_tensor": img,
            "inception_embeds": embedding
        }
