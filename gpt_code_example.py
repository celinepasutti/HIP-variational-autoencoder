import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)  # remove the sequence dimension
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(latent_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, z, seq_len):
        z = z.unsqueeze(1).repeat(1, seq_len, 1)  # replicate z along sequence dimension
        x, _ = self.lstm(z)
        x = self.fc(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, x.size(1))  # x.size(1) is the sequence length
        return recon, mu, logvar

def train_vae(vae, train_loader, device, num_epochs=1000, learning_rate=1e-4):
    weight_decay = 1e-2
    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def loss_function(recon_x, x, mu, logvar):
        # Reconstruction loss (mean squared error)
        MSE = F.mse_loss(recon_x, x)
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + 0 * KLD

    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader.dataset):.4f}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tensor
tensor = torch.load('tensor.pt', map_location=torch.device(device))

print(tensor.shape)
print(tensor.dtype)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define input and latent dimensions
input_dim = tensor.size(-1)  # Number of features per time step, assuming the last dimension is the feature dimension
latent_dim = 32  # Chosen latent dimension

# Initialize the VAE and move it to the appropriate device
vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)

# Train the VAE
train_vae(vae, train_loader, device)