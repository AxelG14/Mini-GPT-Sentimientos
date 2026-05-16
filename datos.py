import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import tiktoken
import kagglehub

class DatasetIMDb(Dataset):
    def __init__(self, df, max_length):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_length = max_length
        self.pad_token = 50256
        
        # Detectar dinámicamente las columnas del CSV
        # Usualmente son 'review_es' y 'sentimiento' en este dataset
        texto_col = 'review_es' if 'review_es' in df.columns else df.columns[0]
        label_col = 'sentimiento' if 'sentimiento' in df.columns else df.columns[1]

        self.textos = df[texto_col].astype(str).values
        
        # Extraer etiquetas y convertirlas a números (1: positivo, 0: negativo)
        # Convertimos todo a minúsculas y buscamos la raíz 'positiv' para evitar fallos de formato
        etiquetas_raw = df[label_col].astype(str).str.lower().values
        self.etiquetas = [1 if 'positiv' in label else 0 for label in etiquetas_raw]

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        encoded = self.tokenizer.encode(texto)
        
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [self.pad_token] * (self.max_length - len(encoded))
            
        return torch.tensor(encoded), torch.tensor(self.etiquetas[idx], dtype=torch.long)

def preparar_datos_imdb(n_ejemplos=None):
    print("Descargando/Buscando dataset de IMDb en español vía Kaggle...")
    
    path = kagglehub.dataset_download("luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish")
    print(f"Ruta base del dataset: {path}")
    
    # Buscar el archivo CSV dentro de la carpeta descargada
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en la ruta de Kaggle.")
    
    archivo_csv = csv_files[0]
    print(f"Cargando archivo: {archivo_csv}")
    
    df = pd.read_csv(archivo_csv)
    print(f"Columnas detectadas: {df.columns.tolist()}")
    
    if n_ejemplos:
        df = df.head(n_ejemplos)
        
    return df