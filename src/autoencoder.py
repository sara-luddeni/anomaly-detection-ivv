import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_data import train_loader

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 50
for epoch in range(n_epochs):
    epoch_loss = 0
    for batch in train_loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.6f}")

torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved as autoencoder.pth")
