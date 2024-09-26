import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import GRU
from dataset import load_data
import matplotlib.pyplot as plt
import math
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(train_loader, valid_loader, vocab_size, num_layers, num_epochs, batch_size, model_save_name, 
          learning_rate, dropout_prob, print_iter=100):
    
    model = GRU(vocab_size=vocab_size, hidden_size=200, num_layers=num_layers, dropout=dropout_prob).to(device)
    # model = torch.load()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_tokens = 0
        start_time = time.time()

        hidden = model.init_hidden(batch_size)

        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            # data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            logits, hidden = model(data, hidden)
            hidden = hidden.detach()
            
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

            if (i + 1) % print_iter == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}, Elapsed time: {elapsed:.2f} seconds')
                start_time = time.time()
        
        avg_train_loss = total_loss / total_tokens  
        train_losses.append(avg_train_loss)

        train_ppl = math.exp(avg_train_loss)
        print(f'Epoch {epoch+1}, Step {i+1}, Train Perplexity: {train_ppl:.4f}')

        model.eval() 
        val_loss = 0
        with torch.no_grad():
            hidden = model.init_hidden(batch_size)
            for data, targets in valid_loader:
                data, targets = data.to(device), targets.to(device)
                # data = data.view(data.size(0), -1)
                logits, hidden = model(data, hidden)
                hidden = hidden.detach()

                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)

        val_ppl = math.exp(avg_val_loss)
        print(f'Epoch {epoch+1}, Validation Perplexity: {val_ppl:.4f}')

    torch.save(model.state_dict(), f'{model_save_name}-final.pt')
    train_ppl_all = [math.exp(loss) for loss in train_losses]
    val_ppl_all = [math.exp(loss) for loss in val_losses]

    return train_ppl_all, val_ppl_all, model


def test(model, test_loader, vocab_size, batch_size):

    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0

    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            # data = data.view(data.size(0), -1)
            logits, hidden = model(data, hidden)
            hidden = hidden.detach()

            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            test_loss += loss.item()

    avg_tes_loss = test_loss / len(test_loader)
    test_ppl = math.exp(avg_tes_loss)
    print(f'Test Perplexity: {test_ppl:.4f}')
    return test_ppl


def plot_ppl(learning_rate, dropout_prob, num_epochs, train_ppl, val_ppl, test_ppl):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_ppl, label='Train Perplexity')
    plt.plot(range(1, num_epochs + 1), val_ppl, label='Validation Perplexity')
    plt.axhline(y=test_ppl, label='Test Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title(f'Learning Rate: {learning_rate}, Dropout: {dropout_prob}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'RNN/GRU/results/ppl_{learning_rate}_{dropout_prob}.png')
    plt.show()

if __name__ == "__main__":
    data_path = 'RNN/ptb_data'  

    parser = argparse.ArgumentParser(description='Train GRU model with custom parameters')

    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers')

    args = parser.parse_args()

    train_loader, valid_loader, test_loader, vocab_size = load_data(data_path, args.batch_size)

    print(f"Running with Learning Rate: {args.learning_rate}, Dropout: {args.dropout}")
    
    train_ppl, val_ppl, model = train(train_loader, valid_loader, vocab_size, args.num_layers, args.num_epochs, 
                                        batch_size=args.batch_size, 
                                        model_save_name=f'RNN/GRU/models/train-checkpoint-lr{args.learning_rate}-dropout{args.dropout}', 
                                        learning_rate=args.learning_rate, dropout_prob=args.dropout)

    model_path = f'RNN/GRU/models/train-checkpoint-lr{args.learning_rate}-dropout{args.dropout}-final.pt'
    model.load_state_dict(torch.load(model_path))

    test_ppl = test(model, test_loader, vocab_size, args.batch_size)

    plot_ppl(args.learning_rate, args.dropout, args.num_epochs, train_ppl, val_ppl, test_ppl)