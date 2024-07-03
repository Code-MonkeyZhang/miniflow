import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
import time

def load_data(filename):
    data = pd.read_csv(filename)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.float32) / 255.0
    return X, y

# 检查GPU可用性并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 47),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def main():
    train_file = '/home/yufeng/Workspace/miniflow/data/eminst/emnist-balanced-train.csv'
    test_file = '/home/yufeng/Workspace/miniflow/data/eminst/emnist-balanced-test.csv'
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)

    x_train, y_train = torch.tensor(x_train).to(device), torch.tensor(y_train, dtype=torch.long).to(device)
    x_test, y_test = torch.tensor(x_test).to(device), torch.tensor(y_test, dtype=torch.long).to(device)

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = NeuralNet().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(10):  # 适当增加训练轮次
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as tepoch:
            for X, y in tepoch:
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
