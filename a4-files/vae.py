from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt

# !mkdir results

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self, flatten_input_size = 784, latent_size=20, encoder_hidden_size=400):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder_hidden_size = encoder_hidden_size
        self.flatten_input_size = flatten_input_size
        # define the encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_input_size, self.encoder_hidden_size),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(self.encoder_hidden_size, self.latent_size)
        self.encoder_var = nn.Linear(self.encoder_hidden_size, self.latent_size)

        # define the decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden_size, self.flatten_input_size)
        )
    def encode(self, x):
        # The encoder will take an input of size 784, and will produce two vectors of size latent_size
        # (corresponding to the coordinatewise means and log_variances)
        # It should have a single hidden linear layer with 400 nodes using ReLU activations,
        # and have two linear output layers (no activations)
        # TODO
        #x = torch.flatten(x, start_dim=1)
        feature_map = self.encoder(x)
        means = self.encoder_mu(feature_map)
        log_variances = self.encoder_var(feature_map)
        return [means, log_variances]

    def reparameterize(self, means, log_variances):
        # The reparameterization module lies between the encoder and the decoder
        # It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size),
        # and returns a sample from a Gaussian with the corresponding parameters
        # TODO
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return eps * std + means

    def decode(self, z):
        # The decoder will take an input of size latent_size, and will produce an output of size 784
        # It should have a single hidden linear layer with 400 nodes using ReLU activations,
        # and use Sigmoid activation for its outputs
        # TODO
        reconstr_img = self.decoder(z)
        return reconstr_img

    def forward(self, x):
        # Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        # Returns an output image of size 784, as well as the means and log_variances,
        # each of size latent_size (they will be needed when computing the loss)
        # TODO
        means, log_variances = self.encode(x)
        z = self.reparameterize(means, log_variances)
        reconstr_img = self.decoder(z)
        return [reconstr_img, means, log_variances]

def vae_loss_function(reconstructed_x, x, means, log_variances):
    # Compute the VAE loss
    # The loss is a sum of two terms: reconstruction error and KL divergence
    # Use cross entropy loss between x and reconstructed_x for the reconstruction error
    # (as opposed to L2 loss as discussed in lecture
    # -- this is sometimes done for data in [0,1] for easier optimization)
    # The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    # Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    kl_divergence = torch.mean(-1/2 * sum(1 + log_variances - means**2 - torch.exp(log_variances)))
    reconstruction_loss = torch.norm(reconstructed_x - x)
    loss = reconstruction_loss + kl_divergence
    return loss, reconstruction_loss

def train(model, optimizer):
    # Trains the VAE for one epoch on the training dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence)
    # and reconstruction loss only (both scalars)
    model.train()
    train_loss = 0
    train_reconstruction_loss = 0
    for i, data in enumerate(train_loader):
        tdata = data[0].flatten(start_dim=1)
        optimizer.zero_grad()
        reconstr_img, means, log_variances = model(tdata)
        loss, reconstruction_loss = vae_loss_function(reconstr_img, tdata, means, log_variances)
        loss.backward()
        train_loss += loss.item()
        train_reconstruction_loss += reconstruction_loss.item()
        optimizer.step()
    avg_train_loss = train_loss/len(train_loader.dataset)
    avg_train_reconstruction_loss = train_reconstruction_loss/len(train_loader.dataset)
    return avg_train_loss, avg_train_reconstruction_loss


def test(model):
    # Runs the VAE on the test dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence)
    # and reconstruction loss only (both scalars)
    # TODO
    model.eval()
    test_loss = 0
    test_reconstruction_loss = 0
    for i, (data, _) in enumerate(test_loader):
        tdata = data[0].flatten(start_dim=1)
        reconstr_img, means, log_variances = model(tdata)
        loss, reconstruction_loss = vae_loss_function(reconstr_img, tdata, means, log_variances)
        test_loss += loss.item()
        test_reconstruction_loss += reconstruction_loss.item()
    avg_test_loss = test_loss/len(test_loader.dataset)
    avg_test_reconstruction_loss = test_reconstruction_loss/len(test_loader.dataset)
    return avg_test_loss, avg_test_reconstruction_loss


epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
