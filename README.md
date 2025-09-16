# Vision-Transformers-and-Fruits-Dataset

Este proyecto implementa un modelo de **Vision Transformer (ViT)** para la **clasificación de frutas**. Se evalúa cómo los *Transformers*, originalmente diseñados para NLP, pueden superar a modelos CNN tradicionales en visión por computadora, utilizando un dataset de frutas con más de 200 clases.

---

## Objetivos
- Aplicar la arquitectura ViT en un dataset desafiante de frutas.  
- Mejorar la precisión empleando el optimizador AdamW 
- Comparar el rendimiento con arquitecturas clásicas de visión.  

---

## Dataset
- Clases: 208  
- Entrenamiento: 84,176 imágenes  
- Validación: 21,045 imágenes  
- Prueba: 35,119 imágenes  

*(El dataset incluye frutas como manzanas, uvas, peras, cítricos, bayas, frutos secos, etc.)*

---

## Tecnologías
- Python 3.10
- Conda (entorno virtual recomendado)  
- PyTorch  
- NumPy, Pandas, Matplotlib  
- scikit-learn  
- kornia

---

## Arquitectura
1. Patch Embedding: división de imágenes en parches.  
2. Positional Encoding: preserva información espacial.  
3. Encoder Transformer: capas de *Multi-Head Self-Attention* y *Feed Forward Layers*.  
4. Clasificador: capa fully connected para salida final de clases.  

---

## Entrenamiento
- Optimizador: AdamW  
- Scheduler: Cosine Decay (factor 0.1)  
- Épocas: 20  
- Augmentacion con GPU: Kornia-Augmentation


### Métricas de Entrenamiento
| Época | Accuracy Entrenamiento | Accuracy Validación |
|-------|------------------------|----------------------|
| 1     | 93.26%                | 98.32%              |
| 2     | 98.80%                | 99.33%              |
| 6     | 99.43%                | 99.60%              |
| 11    | 99.62%                | 99.88%              |
| 16    | 99.72%                | 99.97%              |
| 19    | 99.67%                | 99.99%          |
| 20    | 99.64%                | 99.64%              |

### Resultados Finales (Test Set)
- Pérdida: 0.0099  
- **Precisión:** **99.73%**  

---

## Uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Angello5/Vision-Transfomers-and-Fruits-Dataset.git
   cd Vision-Transfomers-and-Fruits-Dataset
2. Crear entorno Conda
    ```bash 
    conda create -n vit_fruits python=3.10
    conda activate vit_fruits

3. Ejecutar el notebook
    ```bash 
    jupyter notebook ViT.ipynb

4. Para cargar el modelo entrenado 
    ```bash
    import torch
    from model import VisionTransformer

    model = VisionTransformer(...)
    model.load_state_dict(torch.load("best_vit_model.pth"))
    model.eval()

## Autor 
Proyecto desarrollado por Angello5
