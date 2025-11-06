"""Real-time gesture recognition using a TensorFlow model trained on video clips.

Ejecuta la cámara en vivo, detecta ambas manos con MediaPipe y clasifica la
secuencia acumulada mediante el modelo entrenado (SavedModel o archivo Keras).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Imports robustos: paquete o script directo
try:
    from . import config
    from .cli_utils import list_saved_models, prompt_for_model_dir
    from .extract_landmarks import normalise_landmarks
except Exception:
    import config  # type: ignore
    from cli_utils import list_saved_models, prompt_for_model_dir  # type: ignore
    from extract_landmarks import normalise_landmarks  # type: ignore


def parse_args() -> argparse.Namespace:
    """Definir los parámetros de ejecución aceptados desde la terminal."""
    parser = argparse.ArgumentParser(description="Run real-time gesture detection")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,  # si falta, se pedirá por CLI
        help="Directory containing the SavedModel (export) OR a .keras/.h5 model file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional path to labels.json. If omitted, the script looks inside the model directory.",
    )
    parser.add_argument("--device", type=int, default=0, help="Índice de cámara para OpenCV.")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="Umbral de confianza para mostrar etiqueta.")
    parser.add_argument("--sequence-length", type=int, default=config.SEQUENCE_LENGTH, help="Longitud de la ventana temporal.")
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://127.0.0.1:5000/gestures/gesture-key",
        help=(
            "Endpoint del backend Helen para reenviar los gestos reconocidos. "
            "Usa 'none' o cadena vacía para deshabilitar el envío."
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=1.0,
        help=(
            "Tiempo mínimo en segundos entre envíos consecutivos del mismo gesto hacia el frontend. "
            "Permite evitar múltiples activaciones por la misma predicción."
        ),
    )
    return parser.parse_args()


def load_label_map(path: Path) -> Dict[int, str]:
    """Invertir el diccionario gesto->índice para mostrar etiquetas legibles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {idx: gesture for gesture, idx in data.items()}


def build_predict_fn(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """
    Devuelve una función predict(x: np.ndarray)->np.ndarray que entrega
    probabilidades (1, num_classes). Soporta:
      - SavedModel exportado en carpeta (contiene saved_model.pb)
      - Archivo Keras (.keras / .h5)
    """
    if model_path.is_dir() and (model_path / "saved_model.pb").exists():
        # SavedModel exportado con model.export(...)
        saved = tf.saved_model.load(str(model_path))
        # Nombre típico de la firma impreso por export: "serve"
        if "serve" in saved.signatures:
            infer = saved.signatures["serve"]
        else:
            # fallback común
            infer = saved.signatures.get("serving_default")
            if infer is None:
                raise RuntimeError("No se encontró la firma 'serve' ni 'serving_default' en el SavedModel.")
        # La firma espera argumento llamado como el InputSpec; en tu export se imprime 'landmarks'
        input_name = next(iter(infer.structured_input_signature[1].keys()))
        output_name = next(iter(infer.structured_outputs.keys()))

        def _predict(x: np.ndarray) -> np.ndarray:
            out = infer(**{input_name: tf.constant(x, dtype=tf.float32)})
            y = out[output_name].numpy()
            return y  # shape (1, num_classes)

        return _predict

    # Si es archivo .keras / .h5 cargamos con Keras
    if model_path.suffix.lower() in {".keras", ".h5"} or model_path.is_file():
        model = tf.keras.models.load_model(str(model_path))

        def _predict(x: np.ndarray) -> np.ndarray:
            return model.predict(x, verbose=0)

        return _predict

    raise ValueError(
        f"No se reconoce el formato del modelo en: {model_path}. "
        f"Usa una carpeta SavedModel (con saved_model.pb) o un archivo .keras / .h5."
    )


class FrontendBridge:
    """Envía los gestos reconocidos al backend vía HTTP."""

    _COOLDOWN_OVERRIDES = {
        "activar": 1.5,
        "agregar": 2.0,
    }

    def __init__(self, endpoint: Optional[str], *, cooldown: float = 1.0) -> None:
        endpoint = (endpoint or "").strip()
        if endpoint.lower() in {"", "none", "null", "off"}:
            endpoint = ""

        self.endpoint = endpoint
        self.cooldown = max(0.0, float(cooldown))
        self._last_sent: Dict[str, float] = {}
        self._sequence = 0
        self._last_error_at = 0.0

    def _cooldown_for(self, label: str) -> float:
        return float(self._COOLDOWN_OVERRIDES.get(label, self.cooldown))

    def _should_send(self, label: str) -> bool:
        if not self.endpoint or not label:
            return False

        now = time.monotonic()
        cooldown = self._cooldown_for(label)
        last = self._last_sent.get(label)
        if last is not None and cooldown > 0 and (now - last) < cooldown:
            return False

        self._last_sent[label] = now
        return True

    def send(self, label: str, score: float) -> None:
        """Publica el gesto en el backend Helen."""

        normalized = label.strip().lower()
        if not self._should_send(normalized):
            return

        self._sequence += 1
        payload = {
            "gesture": label,
            "character": label,
            "score": float(score),
            "sequence": self._sequence,
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "helen-gesture-bridge"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
                if response.status >= 400:
                    raise urllib.error.HTTPError(
                        self.endpoint,
                        response.status,
                        response.reason,
                        response.headers,
                        None,
                    )
        except urllib.error.URLError as error:
            now = time.monotonic()
            if now - self._last_error_at > 5.0:
                print(f"[WARN] No se pudo notificar el gesto '{label}' al backend: {error}")
                self._last_error_at = now
            return

        print(f"[INFO] Gesto enviado al backend: {label} ({score:.2f})")


def main() -> None:
    """Configurar MediaPipe, cargar el modelo y realizar inferencia cuadro a cuadro."""
    args = parse_args()

    model_dir_or_file: Optional[Path] = args.model_dir
    if model_dir_or_file is None:
        # Selección interactiva de un SavedModel reciente
        model_dir_or_file = prompt_for_model_dir(list_saved_models())

    # labels.json: si no se especifica, se busca dentro del directorio del modelo
    label_path = args.labels or (model_dir_or_file / "labels.json")
    if not label_path.exists():
        raise FileNotFoundError(
            f"No se encontró labels.json. Indica la ruta con --labels o colócalo en {model_dir_or_file}."
        )

    idx_to_label = load_label_map(label_path)
    print("Gestos reconocidos por este modelo:")
    for idx, label in sorted(idx_to_label.items()):
        print(f"   • {idx}: {label}")

    # Predictor unificado (SavedModel o .keras/.h5)
    predict = build_predict_fn(model_dir_or_file)
    print(f"Modelo cargado desde {model_dir_or_file}")

    bridge = FrontendBridge(args.backend_url, cooldown=args.cooldown_seconds)
    if bridge.endpoint:
        print(f"Los gestos se enviarán al backend en {bridge.endpoint}")
    else:
        print("Envío al frontend deshabilitado (sin backend-url)")

    # MediaPipe Hands detectará hasta dos manos y devolverá sus landmarks por frame.
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=config.MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    buffer: Deque[np.ndarray] = deque(maxlen=args.sequence_length)

    print(
        "Presiona 'q' en la ventana de video para salir. Umbral de confianza:",
        args.confidence_threshold,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_features = np.zeros(
                (config.MAX_HANDS, config.NUM_HAND_LANDMARKS, config.LANDMARK_DIM),
                dtype=np.float32,
            )
            if results.multi_hand_landmarks and results.multi_handedness:
                ordering = {"Left": 0, "Right": 1}
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label
                    idx = ordering.get(label, 0)
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                        dtype=np.float32,
                    )
                    frame_features[idx] = coords
                    drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

            buffer.append(normalise_landmarks(frame_features.flatten()))

            if len(buffer) == args.sequence_length:
                # Cuando el buffer está lleno se construye el tensor y se predice la etiqueta.
                input_tensor = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)
                probabilities = predict(input_tensor)[0]  # (num_classes,)
                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])

                if confidence >= args.confidence_threshold:
                    label = idx_to_label.get(pred_idx, "?")
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )
                    if label and label != "?":
                        bridge.send(str(label), confidence)

            cv2.imshow("Detección de gestos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
