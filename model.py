import torch
import torch.nn as nn

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def forward(self, x):
        seq_length = x.size(1)
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Position embeddings
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Apply transformer layers
        x = x.permute(1, 0, 2)  # Transformer expects seq_len first
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Change back to batch first
        
        # Output layer
        x = self.output(x)
        
        return x

# Example usage
vocab_size = 25000  # Should match your tokenizer's vocabulary size
d_model = 512  # Embedding dimension
nhead = 8  # Number of attention heads
num_layers = 6  # Number of transformer layers
max_seq_length = 1024  # Maximum sequence length

model = GPTLanguageModel(vocab_size, d_model, nhead, num_layers, max_seq_length)

# Print model summary
print(model)
