import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.5, init_scale=0.05, is_training=True):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.uniform_(self.embedding.weight, -init_scale, init_scale)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout if is_training else 0.0, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        nn.init.uniform_(self.linear.weight, -init_scale, init_scale)
        
    def forward(self, x, hidden):
        embeds = self.embedding(x)
        if self.is_training:
            embeds = F.dropout(embeds, p=self.dropout)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)

        logits = self.linear(lstm_out)
        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size))