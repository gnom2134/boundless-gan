import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from model import Boundless_GAN
from dataset import Places365Embedding
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=1e-3)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.9)
    parser.add_argument("--lambda-adv", type=float, default=1e-2)
    parser.add_argument("--nthread", default=4, type=int)
    parser.add_argument("--log-every", default=1, type=int)
    parser.add_argument("--embeddings-path", default="./embeddings_inceptionv3_places365.npy", type=str)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    model = Boundless_GAN(args)

    wandb_logger = pl.loggers.WandbLogger(project="Boundless_HSE")
    trainer = pl.Trainer(max_epochs=args.epochs, logger=wandb_logger, accelerator="auto")

    dataset = Places365Embedding(
        Path(args.embeddings_path), "./places365", small=True, download=True
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size)

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
