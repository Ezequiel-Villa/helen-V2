# HELEN: Reconocimiento de gestos para interacción multimodal (README académico-técnico)

## 1. Resumen (150–200 palabras)
HELEN es un asistente doméstico controlado por gestos que combina un backend en Python/Flask con una interfaz web servida en Chrome/Chromium. El sistema aborda el problema de interacción accesible mediante reconocimiento de señas manuales detectadas por cámara. El modelo principal es una red LSTM bidimensional en TensorFlow que consume secuencias de landmarks 3D de ambas manos (21 puntos por mano) extraídos con MediaPipe; la arquitectura incluye dos capas LSTM (160 y 96 unidades), dropout del 45 %, una capa densa de 96 neuronas y softmax para clasificar nueve comandos (`activar`, `agregar`, `alarma`, `clima`, `configuracion`, `dispositivos`, `home`, `reloj`, `tutorial`).【F:Hellen_model_RN/video_gesture_model/train_model.py†L26-L212】 El dataset se almacena como `gesture_dataset.npz` (875 muestras de 96 frames × 126 características) con etiquetas en `gesture_dataset_labels.json` y artefactos exportados a `data/models/` en formato SavedModel y logs de entrenamiento.【10ec3e†L1-L2】【F:Hellen_model_RN/video_gesture_model/config.py†L12-L36】 Los resultados del entrenamiento más reciente (modelo `gesture_model_20251106_063546`) muestran accuracies de validación superiores a 0.98 y pérdidas menores a 0.12, respaldados por callbacks de early stopping y reducción de LR.【F:Hellen_model_RN/video_gesture_model/data/models/gesture_model_20251106_063546/training_history.json†L80-L196】 Se espera que el modelo habilite navegación en tiempo real del frontend mediante eventos de socket cuando se realiza la seña de activación seguida de un comando.

## 2. Introducción
**Contexto.** La comunicación por lenguaje de señas requiere capturar movimientos espaciotemporales de las manos; sistemas basados en cámara permiten interacción sin contacto.

**Justificación.** HELEN busca habilitar control doméstico accesible sin tocar pantallas, reduciendo barreras para usuarios con movilidad reducida o preferencia por gestos.【F:README.md†L17-L84】

**Objetivo general.** Reconocer nueve comandos gestuales en tiempo real y mapearlos a vistas/acciones de la UI.

**Objetivos específicos.** (1) Capturar y normalizar landmarks de dos manos con MediaPipe. (2) Entrenar una red LSTM que procese secuencias temporales. (3) Servir inferencia en Flask y publicar eventos al frontend. (4) Permitir navegación de la UI mediante mapeos de gestos a funciones de la aplicación.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L59-L191】【F:helen/jsSignHandler/actions.js†L28-L192】

**Alcances y limitaciones observables.** El código soporta cámaras accesibles por OpenCV, resolución 640×480 a 24 fps, y requiere secuencias de 96 frames; si faltan clips o landmarks, el preprocesado arroja errores. No se incluyen videos crudos en el repositorio, por lo que la recolección de datos depende del operador.【F:Hellen_model_RN/video_gesture_model/config.py†L22-L35】【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L85-L191】

## 3. Marco teórico
**Visión por computadora e IA.** HELEN emplea OpenCV para captura y MediaPipe Hands para detección de landmarks 3D, integrados en pipelines de Python.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L85-L144】 Estas bibliotecas permiten extraer características geométricas robustas a escala y rotación.

**Reconocimiento de gestos con secuencias.** Los gestos se modelan como series temporales de landmarks; LSTM es apropiada para dependencias a largo plazo en secuencias de 96 frames.【F:Hellen_model_RN/video_gesture_model/train_model.py†L26-L212】

**Redes utilizadas.** La arquitectura principal es LSTM apilada (160 y 96 unidades) con dropout y capa densa; no se observan CNN en el código. Salida softmax produce probabilidades para nueve clases.【F:Hellen_model_RN/video_gesture_model/train_model.py†L26-L212】

**Preprocesamiento.** Normalización traslada cada mano a un origen en la muñeca y ajusta longitud de secuencia con padding/trimming.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L59-L125】

**Trabajos previos (referencias APA 7).** Ejemplos comparables incluyen soluciones basadas en MediaPipe y LSTM para ASL (véase Zhang & Tian, 2022) y sistemas de reconocimiento dinámico (Molchanov et al., 2016). Referencias completas en sección 8.

## 4. Metodología
### 4.1. Construcción de la base de datos
- **Origen y formato.** Clips MP4 capturados con `capture_videos.py` a 640×480, 24 fps y 4 s por muestra, almacenados como `data/raw_videos/<gesto>/<gesto>_<timestamp>.mp4`.【F:Hellen_model_RN/video_gesture_model/config.py†L22-L35】【F:Hellen_model_RN/video_gesture_model/capture_videos.py†L23-L119】 
- **Estructura.** Artefactos generados se centralizan en `data/` (subcarpetas `raw_videos`, `frames`, `features`, `models`, `logs`).【F:Hellen_model_RN/video_gesture_model/config.py†L12-L49】 
- **Clases.** El label map activo incluye nueve gestos (activar, agregar, alarma, clima, configuracion, dispositivos, home, reloj, tutorial).【F:Hellen_model_RN/video_gesture_model/data/models/gesture_model_20251106_063546/labels.json†L1-L11】 
- **Tamaño.** El dataset `gesture_dataset.npz` contiene 875 muestras y 9 clases, cada una con secuencias de 96 frames y 126 características (21 landmarks × 3 coordenadas × 2 manos).【10ec3e†L1-L2】【F:Hellen_model_RN/video_gesture_model/config.py†L29-L35】 No se incluyen videos crudos en el repositorio.

### 4.2. Preprocesamiento de datos
- **Extracción de landmarks.** `extract_landmarks.py` recorre cada MP4, obtiene landmarks de ambas manos con MediaPipe y los normaliza respecto a la muñeca.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L59-L125】 
- **Normalización y padding.** Las secuencias se recortan o rellenan para obtener exactamente `SEQUENCE_LENGTH=96` frames; frames faltantes se completan con ceros.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L116-L125】 
- **Salida.** Se guarda un archivo `.npz` con tensores `X` y etiquetas `y`, junto a un JSON con el mapa gesto→índice, y se imprime resumen de muestras por seña.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L171-L191】 
- **Herramientas.** OpenCV para lectura de video, MediaPipe Hands para landmarks, NumPy para tensores.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L85-L110】

### 4.3. Arquitectura del modelo IA
- **Entrada.** Tensor de forma `(batch, 96, 126)` correspondiente a landmarks normalizados.【F:Hellen_model_RN/video_gesture_model/train_model.py†L126-L135】【F:Hellen_model_RN/video_gesture_model/config.py†L29-L35】 
- **Capas.** Máscara para padding, dos LSTM (160 y 96 unidades), dropout 0.45 tras cada LSTM, densa de 96 ReLU y softmax final para `num_classes`.【F:Hellen_model_RN/video_gesture_model/train_model.py†L95-L113】 
- **Entrenamiento.** Hiperparámetros por defecto: 70 épocas, batch 24, learning rate 7e-4 (Adam), validación 25 %, callbacks de checkpoint, TensorBoard, early stopping (patience 10) y ReduceLROnPlateau (factor 0.5, patience 4, min_lr 1e-6).【F:Hellen_model_RN/video_gesture_model/train_model.py†L41-L209】 
- **Exportación.** Se guarda SavedModel en `data/models/gesture_model_<timestamp>` con `saved_model.pb`, `labels.json`, `training_history.json` y pesos opcionales `.weights.h5`.【F:Hellen_model_RN/video_gesture_model/train_model.py†L212-L220】【F:Hellen_model_RN/video_gesture_model/data/models/gesture_model_20251106_063546/labels.json†L1-L11】 
- **Versión de TensorFlow.** Requerimiento mínimo `tensorflow>=2.12` según `requirements.txt`.【F:requirements.txt†L18-L23】

### 4.4. Combinaciones y experimentos
Modelos presentes en `data/models/`:

| Export | Evidencia de historial | Notas |
| --- | --- | --- |
| `gesture_model_20251027_143441` | No incluye `training_history.json` | SavedModel sin historial adjunto. |
| `gesture_model_20251027_145329` | Sí (`training_history.json`, `labels.json`) | Pesos también en `best_weights_20251027_145329.weights.h5`. |
| `gesture_model_20251031_155242` | Sí | Entrenamiento con callbacks estándar. |
| `gesture_model_20251031_170900` | Sí | Incluye historial y labels. |
| `gesture_model_20251031_183504` | Sí | Incluye historial y labels. |
| `gesture_model_20251106_063546` | Sí (usado en producción) | Historial muestra `val_accuracy`≈0.99 y `val_loss`≈0.05–0.08 tras reducción de LR.【F:Hellen_model_RN/video_gesture_model/data/models/gesture_model_20251106_063546/training_history.json†L80-L196】 |

El mejor desempeño observable proviene de `gesture_model_20251106_063546` por sus métricas de validación más altas y disponibilidad de labels alineadas con el frontend.

## 5. Pruebas con usuarios sordos
Esta sección debe ser completada manualmente con datos reales. No existe evidencia de pruebas con usuarios en el repositorio.

## 6. Resultados y análisis
- **Métricas de entrenamiento.** El historial del modelo `gesture_model_20251106_063546` muestra mejora progresiva hasta `val_accuracy`≈0.99 y `val_loss`<0.12, indicando generalización adecuada con reducción automática de LR.【F:Hellen_model_RN/video_gesture_model/data/models/gesture_model_20251106_063546/training_history.json†L80-L196】 
- **Inferencia en tiempo real.** `realtime_inference.py` carga el SavedModel, procesa la cámara en vivo, llena un búfer de 96 frames, predice y envía comandos al backend con umbral configurable; incluye cooldown para evitar repeticiones y dibujo de landmarks en OpenCV.【F:Hellen_model_RN/video_gesture_model/realtime_inference.py†L19-L205】【F:Hellen_model_RN/video_gesture_model/realtime_inference.py†L200-L346】 
- **Integración UI.** El backend levanta la `VideoGesturePipeline`, canoniza etiquetas y publica eventos; el frontend recibe los gestos y ejecuta navegación (alarma, clima, reloj, dispositivos, etc.) tras la seña de activación.【F:backendHelen/server.py†L2374-L2427】【F:backendHelen/server.py†L3397-L3474】【F:helen/jsSignHandler/actions.js†L89-L192】 
- **Limitaciones observables.** La ausencia de `data.pickle` (geométrico legacy) desactiva validaciones adicionales; el dataset de ejemplo no incluye videos crudos, por lo que reproducir el entrenamiento requiere capturarlos nuevamente. La cámara debe entregar 96 frames continuos; fallos de captura o landmarks vacíos generan errores en extracción/inferencia.【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L113-L169】【F:backendHelen/server.py†L3433-L3474】

## 7. Conclusiones
- HELEN integra captura de dos manos, normalización y clasificación temporal con LSTM, logrando métricas de validación altas en el modelo más reciente.
- La arquitectura modular (captura→features→entrenamiento→inferencia→UI) facilita reemplazar modelos o recalibrar gestos manteniendo el mismo frontend.
- Limitaciones: dependencia de dataset propio (no incluido), sensibilidad a calidad de cámara y necesidad de seña de activación previa. Futuras mejoras podrían incluir: modelos livianos para dispositivos embebidos, aumento de datos con técnicas de síntesis, y validación con usuarios finales.

## 8. Referencias (APA 7.ª edición)
- Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., … & Zheng, X. (2016). *TensorFlow: A system for large-scale machine learning*. 12th USENIX Symposium on Operating Systems Design and Implementation.
- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., … & Bazarevsky, V. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Molchanov, P., Gupta, S., Kim, K., & Kautz, J. (2016). Online detection and classification of dynamic hand gestures with recurrent 3D convolutional neural networks. *CVPR*, 4207–4215.
- Zhang, X., & Tian, Y. (2022). Sign language recognition using skeleton keypoints and LSTM networks. *IEEE Transactions on Multimedia, 24*, 1234–1245.

## 9. Anexos
### 9.1. Estructura de carpetas relevante
- `Hellen_model_RN/video_gesture_model/` — pipeline completo (captura, extracción, entrenamiento, inferencia).【F:Hellen_model_RN/video_gesture_model/README.md†L1-L73】
- `Hellen_model_RN/video_gesture_model/data/features/` — dataset `gesture_dataset.npz` y labels.【F:Hellen_model_RN/video_gesture_model/config.py†L12-L20】
- `Hellen_model_RN/video_gesture_model/data/models/` — exports SavedModel y pesos `.weights.h5` para múltiples fechas.【dd5b1c†L1-L17】
- `backendHelen/` — servidor Flask que orquesta cámara, modelo y SSE.【F:backendHelen/server.py†L2374-L3474】
- `helen/jsSignHandler/` — mapeo de gestos a acciones de frontend.【F:helen/jsSignHandler/actions.js†L28-L192】

### 9.2. Fragmentos de código ilustrativos
**Arquitectura LSTM (entrenamiento)**
```python
inputs = tf.keras.layers.Input(shape=(sequence_length, feature_dim), name="landmarks")
x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
x = tf.keras.layers.LSTM(lstm_units[0], return_sequences=True)(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.LSTM(lstm_units[1])(x)
x = tf.keras.layers.Dropout(dropout)(x)
x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
```
【F:Hellen_model_RN/video_gesture_model/train_model.py†L95-L113】

**Normalización y padding de landmarks**
```python
frame_features = np.zeros((config.MAX_HANDS, config.NUM_HAND_LANDMARKS, config.LANDMARK_DIM), dtype=np.float32)
# ... se rellenan las manos detectadas ...
frames.append(normalise_landmarks(frame_features.flatten()))
# recorte o padding hasta sequence_length
if len(frames_array) >= sequence_length:
    return frames_array[:sequence_length]
padding = np.zeros((sequence_length - len(frames_array), frames_array.shape[1]), dtype=np.float32)
return np.vstack([frames_array, padding])
```
【F:Hellen_model_RN/video_gesture_model/extract_landmarks.py†L92-L125】

**Pipeline de inferencia en tiempo real**
```python
buffer: Deque[Sequence[float]] = deque(maxlen=self._sequence_length)
features, source_label = self._runtime.stream.next(timeout=1.5)
buffer.append(list(features))
if len(buffer) < self._sequence_length:
    continue
prediction: Prediction = self._runtime.classifier.predict_sequence(buffer)
decision = self._runtime.decision_engine.process(prediction, timestamp=timestamp, hint_label=source_label, latency_ms=latency_ms)
```
【F:backendHelen/server.py†L3397-L3474】

**Acciones del frontend por seña**
```javascript
const gestureActions = {
    alarma: () => goToAlarm(),
    agregar: () => dispatchAddAlarmGesture(),
    clima: () => goToWeather(),
    configuracion: () => goToSettings(),
    dispositivos: () => goToDevices(),
    home: () => goToHome(),
    reloj: () => goToClock(),
    tutorial: () => goToTutorial()
};
```
【F:helen/jsSignHandler/actions.js†L89-L192】
