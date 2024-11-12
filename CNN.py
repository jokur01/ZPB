import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from Data import train_loader
from Data import test_loader
from Data import test_data100
from PIL import Image
from Data import TEST

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # (128/2) * (128/2) = 32
        self.fc2 = nn.Linear(128, 2)  # 2 klasy: zmęczony i nie zmęczony

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.10f}')


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)

            # Wydrukowanie wyników wyjścia dla debugowania
            print("Outputs:", outputs)  # Dodano dla obserwacji wyników
            predicted = torch.argmax(outputs, 1)

            # Sprawdzanie przewidywanych wartości dla debugowania
            print("Predicted:", predicted)
            print("Labels:", labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
# train(model, train_loader, criterion, optimizer, num_epochs=10)

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, 'model.pth')

checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# test(model, test_data100)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

image = Image.open("notdrowsy2.jpg").resize((128, 128))
plt.imshow(image)
plt.show()
image_tensor = transform(image).unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(image_tensor)
print(output.data)

