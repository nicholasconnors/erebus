import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class __ImageSeriesDataset(Dataset):
	def __init__(self, data):
		self.data = data.reshape(len(data), -1)
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]

class __Autoencoder(nn.Module):
	def __init__(self, input_dim, latent_dim):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.LeakyReLU(),
			nn.Linear(128, 64),
			nn.LeakyReLU(),
			nn.Linear(64, latent_dim)
		)
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 64),
			nn.LeakyReLU(),
			nn.Linear(64, 128),
			nn.LeakyReLU(),
			nn.Linear(128, input_dim)
		)
	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded, encoded

def get_latent_space(data, latent_dimensions : int):
    input_frames = data.astype(np.float32)
    _, height, width = input_frames.shape
    input_dim = height * width

    batch_size = 16
    dataloader = DataLoader(__ImageSeriesDataset(input_frames), batch_size, shuffle=False)

    autoencoder = __Autoencoder(input_dim, latent_dimensions)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_loss = 0
        total_samples = 0
        for batch in dataloader:
            decoded, encoded = autoencoder(batch)
            loss = functional.mse_loss(decoded, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)
        epoch_loss /= total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss}")

    latents_list = []
    with torch.no_grad():
        for batch in dataloader:
            _, latents = autoencoder(batch)
            latents_list.append(latents)

    latents = torch.cat(latents_list, dim=0).numpy()
    
    return latents.T
