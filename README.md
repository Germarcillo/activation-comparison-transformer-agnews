# Comparación empírica de funciones de activación en un Transformer pequeño y un baseline MLP
Clasificación de noticias AG News en 4 categorías con un Transformer Encoder pequeño, comparando cuatro funciones de activación (ReLU, GELU, SiLU, Tanh) en el bloque feed-forward bajo condiciones experimentales controladas y un baseline MLP que clasifica textos usando el promedio de los embeddings de las palabras.

Proyecto final — Introducción a Deep Learning, Doctorado en Ingeniería Estadística, Universidad Nacional de Ingeniería - Perú.

## Pregunta de investigación

¿Produce la función de activación del bloque feed-forward un efecto medible en un Transformer de baja profundidad aplicado a una tarea de clasificación temática?

**Respuesta corta: no.** Las cuatro activaciones resultan indistinguibles (amplitud de 0.39 puntos porcentuales en exactitud de prueba), y el orden relativo entre ellas se invierte entre validación y prueba. En contraste, la elección arquitectónica sí produce un efecto sistemático.

## Resultados

| Configuración | Mejor val loss | Mejor val acc | Test loss | Test acc |
|---|---|---|---|---|
| ReLU | 0.2476 | 0.9153 | 0.4482 | 0.9093 |
| GELU | 0.2502 | 0.9150 | 0.4428 | 0.9087 |
| SiLU | 0.2503 | 0.9150 | 0.4354 | 0.9072 |
| Tanh | 0.2548 | 0.9146 | 0.4201 | **0.9111** |
| Baseline (MLP) | **0.2420** | **0.9195** | 0.6449 | 0.8995 |

Dos observaciones relevantes:

- **Las activaciones no se diferencian.** Tanh es última en validación y primera en prueba; ReLU hace el recorrido inverso. Esa inconsistencia es la evidencia de que se está midiendo ruido.
- **El baseline gana en validación y pierde en prueba.** Alcanza el mejor pico de validación en la época 1, pero sobreajusta más rápido y su exactitud de prueba es la más baja. La caída validación→prueba es de ~2.0 puntos frente a 0.4–0.8 del Transformer.

Todas las configuraciones sobreajustan desde la primera época: la pérdida de validación crece monótonamente a partir de la época 1.

## Diseño experimental

La función de activación es la única variable independiente. Todo lo demás se mantiene fijo:

- Misma arquitectura, misma semilla (42), misma partición train/val
- Mismo tamaño de lote, misma tasa de aprendizaje, mismo número de épocas
- La semilla se reinicializa antes de instanciar cada modelo, garantizando pesos iniciales idénticos y el mismo orden de barajado

## Datos

[AG News](https://huggingface.co/datasets/ag_news) — noticias en inglés en 4 categorías: World, Sports, Business, Sci/Tech.

| | Ejemplos |
|---|---|
| Entrenamiento | 108,000 |
| Validación | 12,000 |
| Prueba | 7,600 |

Partición 90/10 sobre el conjunto de entrenamiento oficial mediante `random_split` con semilla fija. El conjunto de prueba oficial se reserva y se usa una única vez.

**Preprocesamiento:** tokenización simple (minúsculas + regex `[a-z0-9]+`), vocabulario de 20,000 entradas construido solo sobre entrenamiento, frecuencia mínima 2. Tres tokens especiales: `<pad>`, `<unk>`, `<cls>`. Secuencias truncadas a 64 tokens con padding dinámico por lote y máscara de atención.

## Modelos

**Principal — Transformer Encoder** (2,825,476 parámetros)

Embedding → codificación posicional sinusoidal → 2 capas de encoder → agregación por token `<cls>` → capa lineal a 4 logits. La activación bajo estudio ocupa la posición intermedia del bloque feed-forward.

**Baseline — MLP sobre embeddings promediados** (2,594,052 parámetros)

Promedio enmascarado de embeddings → Linear → ReLU → Linear. Sin atención, sin codificación posicional: invariante al orden de las palabras. Comparte vocabulario, partición, cargadores y funciones de entrenamiento con el modelo principal, de modo que la única diferencia es la arquitectura.

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| `d_model` | 128 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_feedforward` | 256 |
| `dropout` | 0.1 |
| `batch_size` | 64 |
| `learning_rate` | 1e-3 |
| `epochs` | 10 |
| Optimizador | AdamW |
| Pérdida | CrossEntropyLoss |
| Semilla | 42 |
| `MAX_VOCAB` | 20,000 |
| `MAX_LEN` | 64 |

## Ejecución

Abrir el notebook en Google Colab y seleccionar entorno con GPU (**Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**).

```bash
pip install datasets
```

Ejecutar las celdas en orden. El notebook está organizado en secuencia:

1. Setup, semilla y carga de datos
2. Tokenización y vocabulario
3. Dataset, `collate_fn` y DataLoaders
4. Definición del modelo
5. Funciones de entrenamiento y validación
6. Entrenamiento de las cuatro activaciones
7. Baseline
8. Gráficas y tabla comparativa
9. Evaluación en test e inferencia

Tiempo aproximado en T4: ~5 min por configuración, ~25 min en total.

## Entorno

| | |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA Tesla T4 (Google Colab) |
| datasets | 2.14.6 |

Las semillas se fijan sobre `random`, `numpy`, `torch` y `torch.cuda`. La reproducibilidad bit a bit no está garantizada en GPU por el no determinismo de ciertas operaciones de cuDNN, aunque las variaciones esperadas son despreciables frente a los efectos analizados.

## Limitaciones

- **Semilla única.** No es posible cuantificar varianza entre corridas ni realizar contraste estadístico. La conclusión defendible se limita a la ausencia de efecto detectable, no a un orden entre activaciones.
- **Sobreajuste desde la época 1.** Nueve de las diez épocas aportan información sobre memorización, no sobre generalización.
- **Régimen específico.** Los hallazgos aplican a modelos pequeños y tareas con señal léxica dominante; no se extienden a modelos profundos ni a tareas que requieran modelado sintáctico fino.
- **Sin detención temprana.** La evaluación en prueba usa el modelo de la época 10, lo que penaliza especialmente al baseline. Evaluar el modelo del mínimo de validación permitiría discriminar entre las explicaciones propuestas para la brecha validación→prueba.

## Contenido del repositorio

```
├── notebook.ipynb          # Implementación completa
│   ├── train_loss.png
│   ├── validation_loss.png
│   ├── validation_acc.png
│   └── norm_grad.png
└── README.md
```

## Referencias

- Vaswani et al. (2017). *Attention is all you need.* NeurIPS.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Zhang, Zhao & LeCun (2015). *Character-level convolutional networks for text classification.* NeurIPS.
- Joulin et al. (2017). *Bag of tricks for efficient text classification.* EACL.

## Autor: 

Germán Arturo Marcillo Hernández — gerhardez@gmail.com

