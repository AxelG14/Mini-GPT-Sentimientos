import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datos import preparar_datos_imdb, DatasetIMDb # <-- NUEVOS IMPORTS
from modelo import MiniGPTClasificador

def entrenar_con_datos_reales():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando entrenamiento en: {device} ---")

    vocab_size = 50257
    emb_dim = 256      
    max_length = 128  
    num_classes = 2
    batch_size = 32    
    epochs = 5        
    learning_rate = 1e-4

    df = preparar_datos_imdb(n_ejemplos=20000) 
    full_dataset = DatasetIMDb(df, max_length=max_length)
    
    # Dividir: 80% para entrenar, 20% para validar
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Inicializar modelo
    modelo = MiniGPTClasificador(vocab_size, emb_dim, max_length, num_classes).to(device)
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.AdamW(modelo.parameters(), lr=learning_rate)

    # 5. Bucle de entrenamiento
    for epoch in range(epochs):
        modelo.train()
        total_loss = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizador.zero_grad()
            outputs = modelo(inputs)
            loss = criterio(outputs, labels)
            loss.backward()
            optimizador.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Época {epoch+1} | Lote {batch_idx}/{len(train_loader)} | Pérdida: {loss.item():.4f}")

        # Validación al final de cada época
        evaluar_modelo(modelo, val_loader, device)

    # Guardar el progreso
    torch.save(modelo.state_dict(), "mini_gpt_amazon.pth")
    print("\nModelo guardado como 'mini_gpt_amazon.pth'")

def evaluar_modelo(modelo, loader, device):
    modelo.eval()
    correctos = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modelo(inputs)
            predicciones = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correctos += (predicciones == labels).sum().item()
    
    print(f"--> Precisión en Validación (Accuracy): {100 * correctos / total:.2f}%")

if __name__ == "__main__":
    entrenar_con_datos_reales()