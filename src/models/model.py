import torch
from transformers import BertTokenizer, BertForSequenceClassification

class Network(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        self.l1 = torch.nn.Linear(in_features, 500)
        self.l2 = torch.nn.Linear(500, out_features)
        self.r = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))
    
    
# Define the model
class FakeRealClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-cased', num_labels=2):
        super(FakeRealClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.logits
    
    
    
    
    
class TextTransformer(nn.Module):
    def __init__(self, num_heads, num_blocks, embed_dims, vocab_size, max_seq_len, num_classes=2, dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(embedding_dim=embed_dims, num_embeddings=vocab_size)
        self.positional_encoding = PositionalEncoding(embed_dims, max_seq_len)

        encoder_blocks = [EncoderBlock(embed_dim=embed_dims, num_heads=num_heads) for _ in range(num_blocks)]
        self.text_transformer_blocks = nn.Sequential(*encoder_blocks)
        self.output_layer = nn.Linear(embed_dims, num_classes)  # Output layer for classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.embedding(x)
        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        x = self.text_transformer_blocks(x)

        # Pooling (Here, using max pooling over the sequence dimension)
        x = x.max(dim=1)[0]

        # Pass through the output layer
        return self.output_layer(x)
                        
class Attention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(Attention, self).__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim * (-0.5)

        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()

        keys = self.k_projection(x)
        queries = self.q_projection(x)
        values = self.v_projection(x)

        keys = rearrange(
            keys, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )
        values = rearrange(
            values, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )
        queries = rearrange(
            queries, "b seq (h d) -> (b h) seq d", h=self.num_heads, d=self.head_dim
        )

        attention_logits = torch.matmul(keys, values.transpose(1, 2))
        attention_logits *= self.scale 
        attention = torch.nn.functional.softmax(attention_logits, dim=-1)

        out = torch.matmul(attention, values)

        # Rearrange output from all heads
        out = rearrange(
            out, "(b h) seq d->b seq (h d)", h=self.num_heads, d=self.head_dim
        )

        # Remember to add aseertations

        return self.o_projection(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_hidden_dims=None, dropout=0.0):
        super(EncoderBlock, self).__init__()

        # The transformer's encoder block has residual connections
        # Define layers one by one

        self.attention = Attention(
            num_heads, embed_dim
        )  # Can use nn.MultiheadAttention inst of ours

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        # Arbitrary selection of linear hidden layer dims. Can be passed in constructor
        self.fc_hidden_dims = (
            4 * embed_dim if fc_hidden_dims is None else fc_hidden_dims
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.fc_hidden_dims),
            nn.GELU(),
            nn.LayerNorm(self.fc_hidden_dims),
            nn.Linear(self.fc_hidden_dims, embed_dim),
        )

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.layer_norm1(x + attention_output)
        x = self.dropout(x) 

        fc_out = self.fc(x)

        x = self.layer_norm2(fc_out + x)
        x = self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0.0, max_seq_len).unsqueeze(1)

        ####
        # From https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ####

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]
        # return self.dropout(x)