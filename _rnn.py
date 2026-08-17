import torch
import torch.nn as nn

NON_PLAYER_CATEGORICAL = ["typeDescKey", "zoneCode", "shotType", "teamStrength", "opponent"]
PLAYER_SLOTS = ["Player1", "Player2", "Player3", "Player4", "Player5", "Player6"]


class HagelEmbedding(nn.Module):
    def __init__(self, embed_dims, vocab_sizes):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_sizes[feature], embed_dims[feature])
            for feature in NON_PLAYER_CATEGORICAL
        })
        # one shared table for all 6 player slots -- same identity vocab
        self.player_embedding = nn.Embedding(vocab_sizes["playerName"], embed_dims["playerName"])

    def forward(self, batch):
        pieces = [self.embeddings[feature](batch[feature]) for feature in NON_PLAYER_CATEGORICAL]
        pieces += [self.player_embedding(batch[slot]) for slot in PLAYER_SLOTS]
        pieces.append(batch["continuous"])
        return torch.cat(pieces, dim=-1)  # (B, K, input_size)


class CohesioNN(nn.Module):
    def __init__(self, embed_dims, vocab_sizes, num_continuous, rnn_hidden_size=64, dropout=0.2):
        super().__init__()

        self.embedding = HagelEmbedding(embed_dims, vocab_sizes)

        # num_continuous must match the width of batch["continuous"]'s last
        # dim (len(CONTINUOUS_FEATURES) in _datasetPreparer.py) -- passed in
        # explicitly rather than hardcoded, so adding/removing a continuous
        # feature there can't silently desync the GRU's input_size here.
        input_size = sum(
            dim * (6 if name == "playerName" else 1) for name, dim in embed_dims.items()
        ) + num_continuous

        self.rnn = nn.GRU(input_size=input_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.actor_head = nn.Linear(rnn_hidden_size, vocab_sizes["playerName"])
        self.type_head = nn.Linear(rnn_hidden_size, vocab_sizes["typeDescKey"])

    def forward(self, batch):
        x = self.embedding(batch)               # (B, K, input_size)
        _, hidden = self.rnn(x)                  # hidden: (1, B, rnn_hidden_size)
        last_hidden = self.dropout(hidden[-1])   # (B, rnn_hidden_size)
        actor_logits = self.actor_head(last_hidden)
        type_logits = self.type_head(last_hidden)
        return actor_logits, type_logits
