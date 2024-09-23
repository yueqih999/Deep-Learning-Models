import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.5):
        super(GRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size) 

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        if self.training and self.dropout > 0: 
            embeds = F.dropout(embeds, p=self.dropout)

        gru_out, hidden = self.gru(embeds, hidden)
        gru_out = gru_out.contiguous().view(-1, self.hidden_size)

        logits = self.linear(gru_out)
        return logits, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)