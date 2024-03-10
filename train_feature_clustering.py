import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import joblib
import argparse
from models import Autoencoder, autoencoder_train_transform


def train_autoencoder(dataloader, num_epochs=32):
    autoencoder = Autoencoder()
    criterion_autoencoder = nn.MSELoss()
    optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)

    autoencoder_losses = []
    logger.info("Training autoencoder")
    for epoch in range(num_epochs):
        logger.info("Epoch {epoch}", epoch=epoch)
        epoch_loss = 0.0
        for data in tqdm(dataloader):
            img, _ = data
            recon = autoencoder(img)
            loss_autoencoder = criterion_autoencoder(recon, img)
            optimizer_autoencoder.zero_grad()
            loss_autoencoder.backward()
            optimizer_autoencoder.step()
            epoch_loss += loss_autoencoder.item()

        autoencoder_losses.append(epoch_loss / len(dataloader))

    autoencoder.eval()
    return autoencoder, autoencoder_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder to discriminate between player teams and referees.')
    parser.add_argument('--data_folder', type=str, required=False, default="data", help='Path to the folder containing players and referee images.')
    parser.add_argument('--autoencoder_path', type=str, required=False, default="autoencoder.pth", help='Path to save the trained autoencoder model.')
    parser.add_argument('--kmeans_path', type=str, required=False, default="kmeans.pkl", help='Path to save the trained KMeans model.')
    args = parser.parse_args()


    dataset = ImageFolder(root=args.data_folder, transform=autoencoder_train_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # autoencoder, losses = train_autoencoder(dataloader)
    encoder = torchvision.models.resnet50(pretrained=True)
    encoder.eval()  
    # encoder = torch.nn.Sequential(*list(encoder.children())[:-1])

    # Plot training loss progression
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Autoencoder Training Loss Progression')
    # plt.savefig("autoencoder_training.png")

    # Save trained model to disk
    # torch.save(autoencoder.state_dict(), args.autoencoder_path)
    torch.save(encoder.state_dict(), args.autoencoder_path)
    logger.info("Autoencoder saved in {autoencoder_path}", autoencoder_path=args.autoencoder_path)

    # Extract latent representations
    latent_representations = []
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            # latent = autoencoder.encoder(img)
            latent = encoder(img)
            latent_representations.append(latent)
    latent_representations = torch.cat(latent_representations, dim=0)
    latent_np = latent_representations.view(latent_representations.size(0), -1).numpy()

    # Train KMeans with latent representations and save to disk
    kmeans = KMeans(n_clusters=3, random_state=42)
    logger.info("Training KMeans with autoencoder features")
    kmeans.fit(latent_np)
    
    joblib.dump(kmeans, args.kmeans_path)
    logger.info("KMeans saved in {kmeans_path}", kmeans_path=args.kmeans_path)

