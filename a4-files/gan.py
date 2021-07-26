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
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
"""
class Generator(nn.Module):
    # The generator takes an input of size latent_size, and will produce an output of size 784.
    # It should have a single hidden linear layer with 400 nodes using ReLU activations,
    # and use Sigmoid activation for its outputs
    def __init__(self, latent_size=20, generator_hidden_size=400, output_size=784):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.generator_hidden_size = generator_hidden_size
        self.output_size = output_size
        self.generator = nn.Sequential(
            nn.Linear(self.latent_size, self.generator_hidden_size),
            nn.ReLU(),
            nn.Linear(self.generator_hidden_size, self.output_size),
            nn.Sigmoid()
        )
    def forward(self, z):
        generated_img = self.generator(z)
        return generated_img


class Discriminator(nn.Module):
    # The discriminator takes an input of size 784, and will produce an output of size 1.
    # It should have a single hidden linear layer with 400 nodes using ReLU activations,
    # and use Sigmoid activation for its output
    def __init__(self, input_size=784, output_size=1, discriminator_hidden_size=400):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.discriminator_hidden_size = discriminator_hidden_size
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_size, self.discriminator_hidden_size),
            nn.ReLU(),
            nn.Linear(self.discriminator_hidden_size, self.output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        label = self.discriminator(x)
        return label

criterion = nn.BCELoss(size_average=False)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, latent_size, device=device)

# Establish convention for real and fake labels during training

real_label = 1.
fake_label = 0.



def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    # Trains both the generator and discriminator for one epoch on the training dataset.
    # Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    avg_discriminator_loss = 0
    avg_generator_loss = 0
    for i, data in enumerate(train_loader, 0):
        tdata = data[0].flatten(start_dim=1)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        real_img = tdata.to(device)
        #batch_size = real_img.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through discriminator
        output = discriminator(real_img).view(-1)
        # Calculate loss on all-real batch
        discriminator_loss_t = criterion(output, label)
        # Calculate gradients for D in backward pass
        discriminator_loss_t.backward()
        #discriminator_optimizer.step()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, latent_size, device=device)
        # Generate fake image batch with generator
        fake_img = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with discriminator
        output = discriminator(fake_img.detach()).view(-1)
        discriminator_loss_f = criterion(output, label)
        discriminator_loss_f.backward()
        discriminator_loss = discriminator_loss_f + discriminator_loss_t
        discriminator_optimizer.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake_img).view(-1)
        generator_loss = criterion(output, label)
        generator_loss.backward()
        generator_optimizer.step()
        avg_generator_loss += generator_loss/batch_size
        avg_discriminator_loss += discriminator_loss/batch_size
    return avg_generator_loss/(i+1), avg_discriminator_loss/(i+1)

def test(generator, discriminator):
    # Runs both the generator and discriminator over the test dataset.
    # Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    # TODO
    for i, data in enumerate(test_loader, 0):
        tdata = data[0].flatten(start_dim=1)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        real_img = tdata.to(device)
        # batch_size = real_img.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through discriminator
        output = discriminator(real_img).view(-1)
        # Calculate loss on all-real batch
        discriminator_loss_t = criterion(output, label)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, latent_size, device=device)
        # Generate fake image batch with generator
        fake_img = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with discriminator
        output = discriminator(fake_img.detach()).view(-1)
        discriminator_loss_f = criterion(output, label)
        discriminator_loss = discriminator_loss_f + discriminator_loss_t
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake_img).view(-1)
        generator_loss = criterion(output, label)
        avg_generator_loss = generator_loss / batch_size
        avg_discriminator_loss = discriminator_loss / batch_size
    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()
