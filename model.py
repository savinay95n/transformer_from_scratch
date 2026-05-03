import torch
import torch.nn as nn
import math

'''
Steps to build transformer from scratch:
1. Input Embedding: takes 2 inputs:
    a. dmodel: dimension of the model, which is the size of the embedding vector for each token.                
    b. vocab_size: size of the vocabulary, which is the number of unique tokens in the dataset.
    
    nn.Embedding(vocab_size, dmodel) creates an embedding layer that maps each token in the vocabulary to a 
    dmodel-dimensional vector. The embedding layer learns to represent each token in a continuous vector space, which allows the model to capture semantic relationships between tokens.
    
    Note: nn.Embedding takes a number and maps it to a vector of the specified dimension. 
    The input to the embedding layer is typically a batch of token indices, and the output is a batch of corresponding embedding vectors. 
    The multiplication by math.sqrt(dmodel) is a scaling factor that helps to stabilize the training process by preventing the embeddings from becoming too large.
    
2. Positional Encoding: Since the transformer architecture does not have any inherent notion of the order of tokens in a sequence, 
   we need to add positional information to the input embeddings.
   
   This takes 3 inputs:
    a. dmodel: dimension of the model, which is the size of the embedding vector
    b. seq_len: maximum length of the input sequence, which determines the size of the positional encoding matrix.
    c. dropout: dropout rate, which is used to prevent overfitting by randomly dropping out some of the positional encoding values during training.
    
3. Layer Normalization: This is a technique used to normalize the activations of a layer across the features, which helps to stabilize and accelerate the training process.
   
    This takes 1 input:
    a. eps: a small constant added to the denominator for numerical stability, which prevents division by zero when normalizing the activations.
    
4. Feed Forward: This is a fully connected feedforward neural network that is applied to each position in the sequence independently.
    This takes 3 inputs:
    a. dmodel: dimension of the model, which is the size of the input and output vectors for the feedforward network.
    b. dff: dimension of the feedforward network, which is the size of the hidden layer in the feedforward network.
    c. dropout: dropout rate, which is used to prevent overfitting by randomly dropping out some of the activations in the feedforward network during training.
    
    Formula: FFN(x) = Linear2(ReLU(Linear1(x)))
    
5. Multi-Head Attention Block: This is the core component of the transformer architecture that allows the model to attend to different parts of the input sequence simultaneously.
    This takes 3 inputs:
    a. dmodel: dimension of the model, which is the size of the input and output vectors for the multi-head attention block.
    b. h: number of attention heads, which determines how many different parts of the input sequence the model can attend to simultaneously.
    c. dropout: dropout rate, which is used to prevent overfitting by randomly dropping out some of the attention weights during training.
    
6. Encoder Block: This is a single block of the encoder, which consists of a multi-head attention block followed by a feedforward block, with residual connections and layer normalization applied to each sublayer.
    This takes 3 inputs:
    a. self_attention_block: an instance of the MultiHeadAttentionBlock class, which is used to compute the self-attention for the input sequence.
    b. feed_forward_block: an instance of the FeedForwardBlock class, which is used to compute the feedforward transformation for the input sequence.
    c. dropout: dropout rate, which is used to prevent overfitting by randomly dropping out some of the activations in the encoder block during training.
    
7. Decoder Block: This is a single block of the decoder, which consists of a self-attention block, a cross-attention block, and a feedforward block, with residual connections and layer normalization applied to each sublayer.
    This takes 4 inputs:
    a. self_attention_block: an instance of the MultiHeadAttentionBlock class, which is used to compute the self-attention for the target sequence.
    b. cross_attention_block: an instance of the MultiHeadAttentionBlock class, which is used to compute the cross-attention between the target sequence and the encoder output.
    c. feed_forward_block: an instance of the FeedForwardBlock class, which is used to compute the feedforward transformation for the target sequence.
    d. dropout: dropout rate, which is used to prevent overfitting by randomly dropping out some of the activations in the decoder block during training.
    
8. Linear Layer: This is the final linear layer that maps the output of the decoder to the vocabulary size, which allows the model to generate predictions for the next token in the sequence.  
    Also called Projection Layer.
    
'''

class InputEnbedding(nn.Module):
    
    def __init__(self, dmodel: int, vocab_size: int):
        super().__init__()
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dmodel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel) 

class PositionalEncoding(nn.Module):
    
    def __init__(self, dmodel: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, dmodel) to hold the positional encodings
        pe = torch.zeros(seq_len, dmodel)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))  # Shape: (dmodel/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, seq_len, dmodel)
        self.register_buffer('pe', pe)  # Register pe as a buffer to ensure it is not updated during training   
        
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)  # Add positional encoding to the input embeddings
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # scale parameter, initialized to 1
        self.bias = nn.Parameter(torch.zeros(1)) # shift parameter, initialized to 0
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)  # Compute the mean across the last dimension
        std = x.std(dim = -1, keepdim=True)  # Compute the standard deviation across the last dimension
        normalized_x = (x - mean) / (std + self.eps)  # Normalize the input
        return self.alpha * normalized_x + self.bias  # Scale and shift the normalized input

class FeedForwardBlock(nn.Module):
    
    def __init__(self, dmodel: int, dff: int, dropout: float):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dmodel, dff)  # First linear layer to expand the dimensionality
        self.linear2 = nn.Linear(dff, dmodel)  # Second linear layer to project back to the original dimensionality
        
    def forward(self, x):
        x = self.linear1(x)  # Apply the first linear transformation
        x = torch.relu(x)  # Apply ReLU activation function
        x = self.dropout(x)  # Apply dropout for regularization
        x = self.linear2(x)  # Apply the second linear transformation
        return x    
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, dmodel: int, h: int, dropout: float):
        super().__init__()
        self.dmodel = dmodel
        self.h = h # Number of attention heads
        self.dropout = nn.Dropout(dropout)
        assert dmodel % h == 0, "dmodel must be divisible by h" 
        self.d_k = dmodel // h  # Dimension of each head
        self.w_q = nn.Linear(dmodel, dmodel)  # Linear layer for query # Wq
        self.w_k = nn.Linear(dmodel, dmodel)  # Linear layer for key # Wk
        self.w_v = nn.Linear(dmodel, dmodel)  # Linear layer for value # Wv
        self.w_o = nn.Linear(dmodel, dmodel)  # Linear layer for output projection # Wo
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # Compute the attention scores
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)  # Apply the mask to the attention scores
        attention_scores = attention_scores.softmax(dim=-1)  # Apply softmax to get the attention weights
        if dropout is not None:
            attention_scores = dropout(attention_scores)  # Apply dropout to the attention weights
            
        output = attention_scores @ value  # Compute the weighted sum of the values
        return output, attention_scores
        
        
    def forward(self, query, key, value, mask):
        query = self.w_q(query)  # Shape: (batch_size, seq_len, dmodel)
        key = self.w_k(key)  # Shape: (batch_size, seq_len, dmodel)
        value = self.w_v(value)  # Shape: (batch_size, seq_len, dmodel)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) ->[Transpose] -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # Shape: (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # Shape: (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)  # Shape: (batch_size, h, seq_len, d_k)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)  # Shape: (batch_size, h, seq_len, d_k)
        
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], x.shape[2], self.dmodel)  # Shape: (batch_size, seq_len, dmodel)
        x = self.w_o(x)  # Shape: (batch_size, seq_len, dmodel)
        return x


class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
    
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])  # Two residual connections: one for self-attention and one for feedforward
    
    def forward(self, x, src_mask): # mask is for padding tokens
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))  # Self-attention sublayer with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)  # Feedforward sublayer with residual connection
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # Final layer normalization after the last encoder block
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  # Pass the input through each encoder block
        return self.norm(x)  # Apply final layer normalization to the output of the last encoder block
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])  # Three residual connections: one for self-attention, one for cross-attention, and one for feedforward
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))  # Self-attention sublayer with residual connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))  # Cross-attention sublayer with residual connection
        x = self.residual_connections[2](x, self.feed_forward_block)  # Feedforward sublayer with residual connection
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()  # Final layer normalization after the last decoder block
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # Pass the input through each decoder block
        return self.norm(x)  # Apply final layer normalization to the output of the last decoder block
                             

class ProjectionLayer(nn.Module):
    
    def __init__(self, dmodel: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dmodel, vocab_size)  # Linear layer to project the decoder output to the vocabulary size
        
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)  # Apply the linear transformation to get the final output logits
    

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEnbedding, tgt_embed: InputEnbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the the embedding layers
    src_embed = InputEnbedding(d_model, src_vocab_size)
    tgt_embed = InputEnbedding(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder and decoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))    
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters of the model using Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def __main__():
    # Sample usage of the transformer model
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    src_seq_len = 50
    tgt_seq_len = 50
    transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)
    print(transformer)  # Print the number of parameters in the model

if __name__ == "__main__":
    __main__()

        