import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VAE
import matplotlib.pyplot as plt
from dataset import train_data_loader, test_data_loader
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import svm
import pickle
import random
from torch.utils.data import DataLoader, Subset
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(filename='VAE/training_log.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def loss_function(recon_x, x, mean, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KL

def get_subset_data_loader(dataset, num_labels):
    indices = list(range(len(dataset)))
    random.seed(42)  
    random.shuffle(indices)
    selected_indices = indices[:num_labels]
    subset = Subset(dataset, selected_indices)
    return DataLoader(subset, batch_size=64, shuffle=True)


def train(model, optimizer, train_loader, epochs, num_labels):
    best_vae_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()   
            z, recon_x, mean, log_var = model(data)
            loss = loss_function(recon_x, data, mean, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        log_msg = f'Labels: {num_labels}, Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}'
        print(log_msg)
        logging.info(log_msg)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_data_loader:
                data = data.view(data.size(0), -1).to(device)
                z, recon_x, mean, log_var = model(data)
                loss = loss_function(recon_x, data, mean, log_var)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_data_loader.dataset)
        log_msg = f'Labels: {num_labels}, Epoch: {epoch+1}, Test Loss: {avg_test_loss:.4f}'
        print(log_msg)
        logging.info(log_msg)

        if avg_test_loss < best_vae_loss:
            best_vae_loss = avg_test_loss
            torch.save(model.state_dict(), f'VAE/VAE_models/vae_model_{num_labels}_labels.pth')
            log_msg = f'Best VAE model saved with Test Loss: {avg_test_loss:.4f}'
            print(log_msg)
            logging.info(log_msg)



def extract_latent_features(data_loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mean, _ = model.encode(data)
            features.append(mean.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    return np.concatenate(features), np.concatenate(labels)

def train_svm(train_features, train_labels, test_features, test_labels, num_labels):
    best_svm_model = None
    best_error_rate = float('inf')
    svm_classifier = svm.SVC(kernel='rbf')  

    svm_classifier.fit(train_features, train_labels)

    test_predictions = svm_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, test_predictions)
    error_rate = 1 - accuracy
    log_msg = f'Labels: {num_labels}, SVM Test Accuracy: {accuracy:.4f}, Test Error Rate: {error_rate:.4f}'
    print(log_msg)
    logging.info(log_msg)

    if error_rate < best_error_rate:
        best_error_rate = error_rate
        best_svm_model = svm_classifier
        with open(f'VAE/SVM_models/svm_model with {num_labels} labels.pkl', 'wb') as f:
            pickle.dump(best_svm_model, f)
        log_msg = f'Best SVM model saved with Test Error Rate: {error_rate:.4f}'
        print(log_msg)
        logging.info(log_msg)


if __name__ == "__main__":
    input_dim = 784
    latent_dim = [512, 256, 128, 64]

    model = VAE(input_dim, latent_dim).to(device)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 40

    label_counts = [100, 600, 1000, 3000]
    for num_labels in label_counts:
        print(f"\nTraining with {num_labels} labels:")
        logging.info(f"\nTraining with {num_labels} labels:")
        
        train_loader = get_subset_data_loader(train_data_loader.dataset, num_labels)

        train(model, optimizer, train_loader, epochs, num_labels)

        train_features, train_labels = extract_latent_features(train_loader, model)
        test_features, test_labels = extract_latent_features(test_data_loader, model)

        train_svm(train_features, train_labels, test_features, test_labels, num_labels)
