import torch
import torch.nn.functional as F
import tiktoken
from modelo import MiniGPTClasificador

def clasificar_texto():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hiperparámetros (DEBEN ser exactamente los mismos del entrenamiento)
    vocab_size = 50257
    emb_dim = 256
    max_length = 128
    num_classes = 2
    
    # Instanciar el modelo y cargar los pesos guardados
    print("Cargando el cerebro del Mini-GPT...")
    modelo = MiniGPTClasificador(vocab_size, emb_dim, max_length, num_classes)
    
    try:
        # Usamos weights_only=True por seguridad al cargar archivos .pth
        modelo.load_state_dict(torch.load("mini_gpt_amazon.pth", map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'mini_gpt_amazon.pth'. Asegúrate de haber ejecutado el entrenamiento completo.")
        return

    modelo.to(device)
    
    # Esto desactiva el Dropout y otros comportamientos exclusivos del entrenamiento.
    modelo.eval() 
    
    # 4. Preparar el tokenizador
    tokenizer = tiktoken.get_encoding("gpt2")
    pad_token = 50256
    
    print("\n" + "="*40)
    print("MINI-GPT: ANALIZADOR DE SENTIMIENTO")
    print("Escribe 'salir' para terminar.")
    print("="*40 + "\n")
    
    # Bucle interactivo
    while True:
        texto = input("Escribe una reseña o comentario: ")
        
        if texto.strip().lower() == 'salir':
            print("Cerrando el sistema...")
            break
            
        if not texto.strip():
            continue
            
        encoded = tokenizer.encode(texto)
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded = encoded + [pad_token] * (max_length - len(encoded))
            
        # Añadir la dimensión del batch (1, max_length)
        input_tensor = torch.tensor([encoded]).to(device)
        
        # Inferencia sin calcular gradientes (ahorra mucha memoria y es más rápido)
        with torch.no_grad():
            logits = modelo(input_tensor)
            
            # Convertir las puntuaciones crudas (logits) a porcentajes de probabilidad
            probabilidades = F.softmax(logits, dim=1)[0]
            
            prob_neg = probabilidades[0].item()
            prob_pos = probabilidades[1].item()
            
            prediccion = "POSITIVO 🟢" if prob_pos > prob_neg else "NEGATIVO 🔴"
            
        print(f"--> Análisis: {prediccion}")

if __name__ == "__main__":
    clasificar_texto()