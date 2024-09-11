import torch
import sys
from Lenet import LeNet
from Dataset import MNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def test_saved_model(technique):
    model = LeNet(
        use_dropout=(technique == 'dropout'),
        use_bn=(technique == 'batch_norm')
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(f'models/lenet5_{technique}.pth'))

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = MNISTDataset('Lenet5/data/test-images-idx3-ubyte.gz', 'Lenet5/data/test-labels-idx1-ubyte.gz', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)  
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {test_accuracy:.2f}%")

if __name__ == '__main__':
    technique = sys.argv[1] if len(sys.argv) > 1 else 'none'
    test_saved_model(technique)
