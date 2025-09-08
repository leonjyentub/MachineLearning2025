import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import seaborn as sns

# create a transofrm to apply to each datapoint
transform = transforms.Compose([transforms.ToTensor()])

# download the MNIST datasets
path = "drive/My Drive/mnist/MNIST_data/"
train_dataset = MNIST(path, transform=transform, download=True)
test_dataset = MNIST(path, transform=transform, download=True)

# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device = {device}")
# get 25 sample training images for visualization
dataiter = iter(train_loader)
image = next(dataiter)

num_samples = 25
sample_images = [image[0][i, 0] for i in range(num_samples)]

fig = plt.figure(figsize=(5, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

for ax, im in zip(grid, sample_images):
    ax.imshow(im, cmap="gray")
    ax.axis("off")

plt.show()


class Encoder(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=256):
        super(Encoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True

    def forward(self, x):
        x = self.LeakyReLU(self.linear1(x))
        x = self.LeakyReLU(self.linear2(x))

        mean = self.mean(x)
        log_var = self.var(x)
        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        super(Decoder, self).__init__()

        self.linear2 = nn.Linear(latent_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.linear2(x))
        x = self.LeakyReLU(self.linear1(x))

        x_hat = torch.sigmoid(self.output(x))
        return x_hat


class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        # Returns a tensor with the same size as input that is filled with random numbers
        # from a normal distribution with mean 0 and variance 1.
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var


model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


def train(model, optimizer, epochs, device, x_dim=784):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim).to(device)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(
            "\tEpoch",
            epoch + 1,
            "\tAverage Loss: ",
            overall_loss / (batch_idx * batch_size),
        )
    return overall_loss


overall_loss = train(model, optimizer, epochs=50, device=device)


def generate_digit(mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28)  # reshape vector to 2d array
    plt.title(f"[{mean},{var}]")
    plt.imshow(digit, cmap="gray")
    plt.axis("off")
    plt.show()


# img1: mean0, var1 / img2: mean1, var0
generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)


def plot_latent_space(model, scale=5.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent_space(model, scale=1.0)
# plot_latent_space(model, scale=5.0)

import os
import torch


def save_model(model, optimizer, epoch, loss, save_path="vae_model.pth"):
    """
    Save the entire model state, including model parameters, optimizer state,
    current epoch, and loss.

    Args:
    - model: The VAE model
    - optimizer: The optimizer
    - epoch: Current training epoch
    - loss: Last recorded loss
    - save_path: Path to save the model
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def load_model(model, optimizer=None, load_path="vae_model.pth"):
    """
    Load a saved model state.

    Args:
    - model: The VAE model instance to load state into
    - optimizer: Optional optimizer to load state (if used during training)
    - load_path: Path to the saved model file

    Returns:
    - Tuple of (model, optimizer, epoch, loss) if optimizer is provided
    - Otherwise returns (model, epoch, loss)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Optional loading of optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch"], checkpoint["loss"]

    return model, checkpoint["epoch"], checkpoint["loss"]


# Example usage:
# After training, save the model
save_model(model, optimizer, 50, overall_loss, "mnist_vae.pth")

# To load the model later
# Option 1: Load model without optimizer
loaded_model, loaded_epoch, loaded_loss = load_model(
    VAE().to(device), load_path="mnist_vae.pth"
)

# Option 2: Load model with optimizer
loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_model(
    VAE().to(device),
    optimizer=Adam(model.parameters(), lr=1e-3),
    load_path="mnist_vae.pth",
)

# If you want to continue training from a saved checkpoint
model = loaded_model
optimizer = loaded_optimizer
start_epoch = loaded_epoch


def plot_latent_space_with_classes(model, train_loader, device):
    # Collect latent representations and their corresponding labels
    latent_means = []
    labels = []

    model.eval()
    with torch.no_grad():
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            mean, _ = model.encode(x)
            latent_means.append(mean.cpu().numpy())
            labels.append(y.numpy())

    # Concatenate all collected data
    latent_means = np.concatenate(latent_means)
    labels = np.concatenate(labels)

    # Create a plot
    plt.figure(figsize=(12, 8))

    # Use a color palette with distinct colors for 10 classes
    palette = sns.color_palette("husl", 10)

    # Scatter plot with different colors for each digit class
    for digit in range(10):
        mask = labels == digit
        plt.scatter(
            latent_means[mask, 0],
            latent_means[mask, 1],
            c=[palette[digit]],
            label=f"Digit {digit}",
            alpha=0.7,
        )

    plt.title("VAE Latent Space Visualization by Digit Class")
    plt.xlabel("Latent Dimension 1 (Mean)")
    plt.ylabel("Latent Dimension 2 (Mean)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# Use the function after training the model
plot_latent_space_with_classes(model, train_loader, device)
