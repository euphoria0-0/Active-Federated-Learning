import torch.nn as nn
import torch


class BLSTM(nn.Module):
    def __init__(self, embedding_dim=64, vocab_size=2, blstm_hidden_size=32, mlp_hidden_size=64, blstm_num_layers=2):
        # AFL: 64-dim embedding, 32-dim BLSTM, MLP with one layer(64-dim)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.blstm = nn.LSTM(input_size=embedding_dim, hidden_size=blstm_hidden_size, num_layers=blstm_num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(mlp_hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output