import torch
import torch.nn as nn

class MiniGPTClasificador(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, num_classes, num_layers=2):
        super().__init__()
        
        # Mapea los IDs de los tokens a vectores densos
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        
        self.pos_emb = nn.Embedding(max_length, emb_dim)
        
        # Usamos la implementación optimizada de PyTorch por ahora
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=4,
            dim_feedforward=emb_dim * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm_final = nn.LayerNorm(emb_dim)

        # Reduce la dimensión del embedding a la cantidad de clases
        self.cabeza_clasificacion = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length = x.size()
        
        # Generar las posiciones (0, 1, 2, ..., seq_length - 1)
        posiciones = torch.arange(seq_length, device=x.device)
        
        x_emb = self.token_emb(x) + self.pos_emb(posiciones)
        
        # Pasar por las capas de Atención (Transformer)
        x_transformer = self.transformer_blocks(x_emb)
        
        ultimo_token = x_transformer[:, -1, :] 
        
        ultimo_token = self.layer_norm_final(ultimo_token)
        logits = self.cabeza_clasificacion(ultimo_token)
        
        return logits