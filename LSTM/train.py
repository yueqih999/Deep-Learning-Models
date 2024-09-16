import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from model import LSTM
from dataset import load_data

def train(train_loader, valid_loader, vocab_size, num_layers, num_epochs, batch_size, model_save_name, 
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=1000):
    
    model = LSTM(vocab_size=vocab_size, hidden_size=200, num_layers=num_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max_lr_epoch, gamma=lr_decay)

    model.train()  

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        hidden = model.init_hidden(batch_size)

        for i, (data, targets) in enumerate(train_loader):
            data = data.view(batch_size, -1)
            optimizer.zero_grad()
            logits, hidden = model(data, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % print_iter == 0:
                elapsed = time.time() - start_time
                print(f'Epoch {epoch+1}, Step {i+1}, Loss: {loss.item():.4f}, Elapsed time: {elapsed:.2f} seconds')
                start_time = time.time()

        model.eval() 
        val_loss = 0
        with torch.no_grad():
            hidden = model.init_hidden(batch_size)
            for data, targets in valid_loader:
                data = data.view(batch_size, -1)
                logits, hidden = model(data, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())

                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

        # Learning rate scheduling
        lr_scheduler.step()

    torch.save(model.state_dict(), f'{model_save_name}-final.pt')



if __name__ == "__main__":
    data_path = 'LSTM/ptb_data'  
    batch_size = 32
    train_loader, valid_loader, test_loader, vocab_size = load_data(data_path, batch_size)

    train(train_loader, valid_loader, vocab_size, num_layers=2, num_epochs=70, batch_size=batch_size,
          model_save_name='LSTM/models/train-checkpoint')
