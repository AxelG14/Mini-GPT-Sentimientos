# 🧠 Mini-GPT: Análisis de Sentimiento en Español

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completado-success.svg)]()

Este proyecto implementa una arquitectura de red neuronal Transformer tipo GPT (Decoder-only) construida y entrenada absolutamente desde cero. A diferencia de los modelos generativos tradicionales, la arquitectura de este Mini-GPT ha sido adaptada mediante el uso del "Bottleneck del último token" para realizar una tarea de clasificación discreta: el **Análisis de Sentimiento de reseñas cinematográficas en español**.

## 🚀 Características Principales

* **Arquitectura Transformer Personalizada:** Implementación matemática desde cero de las capas de *Embeddings* (Semánticos y Posicionales), mecanismos de *Atención Causal (Masked Multi-Head Attention)* y redes *Feed-Forward*.
* **Tokenización BPE:** Integración de *Byte Pair Encoding* utilizando la librería `tiktoken` (estándar de OpenAI) para un manejo eficiente del vocabulario y palabras desconocidas.
* **Inferencia en Tiempo Real:** Interfaz de consola integrada para evaluar nuevas frases generadas por el usuario al instante.
* **Pipeline de Datos Escalable:** Extracción, limpieza y carga dinámica del dataset de Kaggle (50,000 reseñas) optimizada para procesamiento en GPU mediante `DataLoader`.

## 📁 Estructura del Proyecto

```text
MiniGPT_Sentimiento/
│
├── datos.py             # Lógica de extracción, limpieza y tokenización del dataset de Kaggle.
├── modelo.py            # Definición matemática de los bloques Transformer y la cabeza de clasificación.
├── entrenamiento.py     # Bucle de entrenamiento, Backpropagation y cálculo de métricas.
├── inferencia.py        # Script interactivo para cargar los pesos guardados y clasificar texto nuevo.
└── README.md            # Documentación del proyecto.

⚙️ Arquitectura del Modelo
El sistema fue escalado para un equilibrio óptimo entre precisión y tiempos de entrenamiento en hardware de consumo

Tamaño del vocabulario: 50,257
tokensDimensión de Embedding (d_model): 512
Longitud máxima de contexto: 256
tokensCapas Transformer (Blocks): 4
Cabezas de Atención: 4
Función de Pérdida: Cross Entropy LossOptimizador
AdamW📊 Dataset Utilizado

El modelo fue entrenado con el IMDb Dataset of 50K Movie Reviews - Spanish extraído mediante kagglehub.
El corpus consta de 50.000 reseñas balanceadas (25k positivas / 25k negativas) traducidas al español.
