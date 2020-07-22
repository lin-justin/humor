import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1234)

class RNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1,:,:], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

class BiLSTM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers,
                 bidirectional, 
                 dropout, 
                 pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers,
                           bidirectional = bidirectional,
                           dropout = dropout)

        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)

        output, (hidden, cell) = self.rnn(embedded)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        return self.fc(hidden.squeeze(0))
