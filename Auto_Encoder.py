"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a Auto_encoder model to generate image

"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime

start_time = datetime.now()  # Start timer


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_dim = 20
lr = 0.001
num_epochs = 20

# Data processing
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Auto-encoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x.view(-1, 784))
        reconstructed = self.decoder(latent)
        return reconstructed.view(-1, 1, 28, 28)

# Initialization
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Visualization
    if epoch % 3 == 0:
        with torch.no_grad():
            test_img = img[0:1]
            reconstructed = model(test_img)
            
            plt.figure(figsize=(9,3))
            plt.subplot(1,2,1)
            plt.imshow(test_img.cpu().squeeze(), cmap='gray')
            plt.title('Original')
            
            plt.subplot(1,2,2)
            plt.imshow(reconstructed.cpu().squeeze(), cmap='gray')
            plt.title('reconstructed')
            plt.savefig(f'Comparison_{epoch}.png')
            plt.close()




end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")















