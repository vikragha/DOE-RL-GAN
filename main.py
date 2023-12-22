# main.py
from gan.generator import Generator
from gan.discriminator import Discriminator
from gan.training_loop import train_gan
from utils.ansatz import AnsatzRepresentations
from utils.matrix_representations import MatrixRepresentations

# Set up GAN parameters
ansatz_dim = 64  # Adjust based on your ansatz representation
layer_complexity = 2  # Adjust based on your layer complexity
num_epochs = 1000  # Adjust based on your training preferences

# Initialize GAN components
generator = Generator(ansatz_dim, layer_complexity)
discriminator = Discriminator(ansatz_dim, layer_complexity)

# Connect ansatz and matrix representations
ansatz_representation = AnsatzRepresentations()
matrix_representation = MatrixRepresentations()

# Train the GAN
train_gan(generator, discriminator, ansatz_dim, layer_complexity, num_epochs, dataloader)
