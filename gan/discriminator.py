# gan/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(Discriminator, self).__init__()

        # Define a neural network architecture for the discriminator
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            nn.Sigmoid()  # Output in the range [0, 1] for binary classification
        )

    def forward(self, input_tensor):
        # Forward pass through the discriminator
        return self.model(input_tensor)

# Example usage:
# Define input size based on the dimensionality of your data
input_size = 100  # Adjust based on the dimensionality of your data

# Create an instance of the discriminator
discriminator = Discriminator(input_size)

# Pass fake data (generated by the generator) through the discriminator
fake_data = torch.randn(1, input_size)
discriminator_output_fake = discriminator(fake_data)

# Print the discriminator output for the fake data
print("Discriminator Output for Fake Data:")
print(discriminator_output_fake)

