import torch
import scipy
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets.places365 import Places365
from torch.utils.data import DataLoader
from typing import Callable, NoReturn

from src.transforms import transform_pipeline
from src.model import Boundless_GAN


def calculate_fid(
    gen_model: torch.nn.Module,
    enc_model: torch.nn.Module,
    enc_transforms: Callable,
    fake_dataloader: DataLoader,
    true_dataloader: DataLoader,
    device: torch.device,
) -> float:
    repr1 = []
    repr2 = []
    with torch.no_grad():
        for batch in fake_dataloader:
            batch = batch.to(device)
            fakes = gen_model(batch)
            repr1.append(enc_model(enc_transforms(fakes)).detach().cpu().numpy())

        for batch in true_dataloader:
            batch = batch.to(device)
            repr1.append(enc_model(enc_transforms(batch)).detach().cpu().numpy())

    repr1 = np.vstack(repr1)
    repr2 = np.vstack(repr2)

    first = np.linalg.norm(np.mean(repr1, axis=0) - np.mean(repr2, axis=0), ord=2)
    cov1 = np.cov(repr1, rowvar=False)
    cov2 = np.cov(repr2, rowvar=False)
    second = np.trace(cov1 + cov2 - 2 * np.abs(scipy.linalg.sqrtm(cov1 @ cov2)))
    return first + second


def draw_images(
    gen_model: torch.nn.Module,
    fake_dataloader: DataLoader,
    device: torch.device,
    size: int = 4,
    path_to_save: str = "./fake_images.png",
) -> NoReturn:
    num = 0
    row = 0
    col = 0
    number = size**2

    f, axarr = plt.subplots(size, size)

    with torch.no_grad():
        for batch in fake_dataloader:
            batch = batch.to(device)
            if num + batch.shape[0] > number:
                batch = batch[: number - num, :, :, :]
                num = number
            else:
                num += number
            fakes = gen_model(batch).detach().cpu()
            for i in range(fakes.shape[0]):
                if col == size:
                    row += 1
                    col = 0
                axarr[row, col].imshow(fakes[i, :, :, :])

    f.savefig(path_to_save)


if __name__ == "__main__":
    fake_dataset = Places365("./places365", split="val", download=False, small=True, transform=transform_pipeline)
    true_dataset = Places365("./places365", split="val", download=False, small=True, transform=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inception_model = torchvision.models.inception.inception_v3(
        weights=torchvision.models.inception.Inception_V3_Weights.IMAGENET1K_V1
    ).to(device)
    inception_transforms = torchvision.models.inception.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    fake_loader = torch.utils.data.DataLoader(fake_dataset, shuffle=False, batch_size=8)
    true_loader = torch.utils.data.DataLoader(true_dataset, shuffle=False, batch_size=8)

    gen_model = Boundless_GAN.load_from_checkpoint("./checkpoint_path").to(device)

    print(
        "FID (based on inception_v3 encoder):",
        calculate_fid(gen_model, inception_model, inception_transforms, fake_loader, true_loader, device),
    )

    draw_images(gen_model, fake_loader, device)
