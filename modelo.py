import torch
import torch.nn as nn

class MiniGPTClasificador(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_length, num_classes, num_layers=2):
        super().__init__()
        
        # 1. Capas de Representación (Embeddings)
        # Mapea los IDs de los tokens a vectores densos
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        # Le da a cada token información sobre su posición en la oración
        self.pos_emb = nn.Embedding(max_length, emb_dim)
        
        # 2. Bloques Transformer (El núcleo del modelo)
        # Usamos la implementación optimizada de PyTorch por ahora
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=4, # Multi-Head Attention (4 "cabezas" pensando a la vez)
            dim_feedforward=emb_dim * 4, 
            dropout=0.1, 
            batch_first=True # Fundamental porque nuestros datos entran como [Batch, Seq_len]
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. El Truco: La Cabeza de Clasificación
        self.layer_norm_final = nn.LayerNorm(emb_dim)
        # Reduce la dimensión del embedding a la cantidad de clases (Ej. 2: Positivo/Negativo)
        self.cabeza_clasificacion = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length = x.size()
        
        # Generar las posiciones (0, 1, 2, ..., seq_length - 1)
        posiciones = torch.arange(seq_length, device=x.device)
        
        # Sumar Token Embeddings + Positional Embeddings
        x_emb = self.token_emb(x) + self.pos_emb(posiciones)
        
        # Pasar por las capas de Atención (Transformer)
        x_transformer = self.transformer_blocks(x_emb)
        
        # Extraer SOLO la representación del ÚLTIMO token de cada oración
        # x_transformer tiene la forma: [batch_size, seq_length, emb_dim]
        ultimo_token = x_transformer[:, -1, :] 
        
        # Normalizar y clasificar
        ultimo_token = self.layer_norm_final(ultimo_token)
        logits = self.cabeza_clasificacion(ultimo_token)
        
        return logits

# === PRUEBA RÁPIDA DEL MODELO ===
def probar_modelo():
    print("=== INICIANDO PRUEBA DEL MODELO ===")
    
    # Configuramos los hiperparámetros
    vocab_size = 50257 # Tamaño del vocabulario de GPT-2
    emb_dim = 128      # Dimensión de los vectores (reducido para pruebas rápidas)
    max_length = 10    # Debe coincidir con lo que pusimos en datos.py
    num_classes = 2    # 0 (Negativo) o 1 (Positivo)
    
    # Instanciamos el modelo
    modelo = MiniGPTClasificador(vocab_size, emb_dim, max_length, num_classes)
    
    # Simulamos un lote (batch) de entrada que nos daría el DataLoader (2 oraciones, 10 tokens)
    # Usamos números aleatorios para simular los IDs de los tokens
    batch_prueba = torch.randint(0, vocab_size, (2, 10))
    
    print(f"Forma de entrada (Batch_size, Max_length): {batch_prueba.shape}")
    
    # Pasamos los datos por el modelo (Forward Pass)
    salida_logits = modelo(batch_prueba)
    
    print(f"Forma de salida (Batch_size, Num_classes): {salida_logits.shape}")
    print(f"Logits (Puntuaciones crudas del modelo):\n{salida_logits}")

if __name__ == "__main__":
    probar_modelo()