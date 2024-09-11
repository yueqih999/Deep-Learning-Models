import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from Lenet import LeNet
import matplotlib.pyplot as plt
from Dataset import MNISTDataset
import os

def train(train_loader, val_loader, technique='none', weight_decay=0):

    model = LeNet(
        use_dropout=(technique == 'dropout'),
        use_bn=(technique == 'bn'))
    
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay) 

    train_accuracies = []
    val_accuracies = []

    for epoch in range(15):
        correct = 0
        total = 0
        running_loss = 0.0
        model.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()  

            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_acc}%")

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(inputs)  
                loss = criterion(outputs, labels) 
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)  
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss / len(val_loader)}, Accuracy on the val set: {val_accuracy}%")

    print("Finished Training")

    os.makedirs('Lenet5/models', exist_ok=True)
    torch.save(model.state_dict(), f'Lenet5/models/lenet5_{technique}.pth')

    return train_accuracies, model


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)  
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {test_accuracy}%")

    return test_accuracy


def plot(train_acc, test_acc, title):
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'r', label='Train Accuracy')
    plt.scatter(len(epochs), test_acc, color='b', label='Test Accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'Lenet5/results/{title}.png')
    plt.show()


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # full_train_dataset = torchvision.datasets.MNIST(root='HW1/download', train=True, transform=transform, download=True)
    full_train_dataset = MNISTDataset('Lenet5/data/train-images-idx3-ubyte.gz', 'Lenet5/data/train-labels-idx1-ubyte.gz', transform=transform)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # test_dataset = torchvision.datasets.MNIST(root='HW1/download', train=False, transform=transform, download=True)
    test_dataset = MNISTDataset('Lenet5/data/test-images-idx3-ubyte.gz', 'Lenet5/data/test-labels-idx1-ubyte.gz', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_acc_no_reg, model_no_reg = train(train_loader, val_loader, technique='none')
    test_acc_no_reg = test(model_no_reg, test_loader)
    plot(train_acc_no_reg, test_acc_no_reg, 'No Regularization')

    train_acc_dropout, model_dropout = train(train_loader, val_loader, technique='dropout')
    test_acc_dropout = test(model_dropout, test_loader)
    plot(train_acc_dropout, test_acc_dropout, 'Dropout')

    train_acc_weight_decay, model_weight_decay = train(train_loader, val_loader, technique='l2', weight_decay=1e-4)
    test_acc_weight_decay = test(model_weight_decay, test_loader)
    plot(train_acc_weight_decay, test_acc_weight_decay, 'Weight Decay (L2 Regularization)')

    train_acc_batch_norm, model_batch_norm = train(train_loader, val_loader, technique='batch_norm')
    test_acc_batch_norm = test(model_batch_norm, test_loader)
    plot(train_acc_batch_norm, test_acc_batch_norm, 'Batch Normalization')