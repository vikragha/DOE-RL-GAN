# gan/training_loop.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class TrainingLoop:
    def __init__(self, generator, discriminator, dataloader, device, generate_random_input):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.device = device
        self.generate_random_input = generate_random_input

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for real_data in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)

                # Train Discriminator
                self.discriminator_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                discriminator_output_on_real_data = self.discriminator(real_data)
                real_loss = self.adversarial_loss(discriminator_output_on_real_data, real_labels)

                generated_data, ansatz_output = self.generator(self.generate_random_input(batch_size))
                discriminator_output_on_generated_data = self.discriminator(generated_data.detach())

                fake_loss = self.adversarial_loss(discriminator_output_on_generated_data, fake_labels)
                discriminator_loss = (real_loss + fake_loss) / 2

                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # Train Generator
                self.generator_optimizer.zero_grad()
                discriminator_output_on_generated_data = self.discriminator(generated_data)

                generator_loss = self.adversarial_loss(discriminator_output_on_generated_data, real_labels)
                generator_loss += ansatz_output  # Incorporate the ansatz representation as part of the generator loss

                generator_loss.backward()
                self.generator_optimizer.step()
