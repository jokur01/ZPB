import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

batch_size = 32
image_size = (128, 128)
train_ratio = 0.8

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = ImageFolder(root='dataset', transform=transform)

train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


test_data = ImageFolder(root="./test_dataset", transform=transform)
class_mapping = test_data.class_to_idx
print(class_mapping)

test_data100 = DataLoader(dataset=test_data, batch_size=batch_size)

batch = next(iter(test_data100))
TEST = batch[0][0]

