# gan/generator.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()

        # Define a neural network architecture for the generator
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),  # Introducing Leaky ReLU for stability
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_size),
            nn.Tanh()  # Assuming output is in the range [-1, 1]
        )

    def forward(self, input_tensor):
        # Forward pass through the generator
        return self.model(input_tensor)

# Example usage:
# Define input and output sizes based on your requirements
input_size = 100
output_size = 784  # Adjust based on the dimensionality of your generated data

# Create an instance of the generator
generator = Generator(input_size, output_size)

# Generate fake data by passing random input to the generator
random_input = torch.randn(1, input_size)
fake_data = generator(random_input)

# Print the generated fake data
print("Generated Fake Data:")
print(fake_data)

