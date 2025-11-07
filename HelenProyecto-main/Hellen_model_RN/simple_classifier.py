"""Lightweight pure-Python gesture classifier used as backend fallback."""

from __future__ import annotations

import importlib
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

helpers = importlib.import_module(__name__.rsplit(".", 1)[0] + ".helpers" if "." in __name__ else "helpers")


@dataclass(frozen=True)
class Prediction:
    """Simple value object holding the predicted label and confidence score."""

    label: str
    score: float


class SimpleGestureClassifier:
    """Compute nearest-centroid predictions from a stored dataset."""

    def __init__(self, dataset_path: str | bytes | Path) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"No se encontró el dataset en {self.dataset_path!s}")

        payload = self._load_dataset(self.dataset_path)
        features: Sequence[Sequence[float]] = payload["data"]
        labels: Sequence[str | int] = payload["labels"]
        if not features or not labels:
            raise ValueError("El dataset no contiene datos suficientes para clasificar gestos")
        if len(features) != len(labels):
            raise ValueError("El dataset tiene una cantidad desigual de muestras y etiquetas")

        canonical_labels = [self._coerce_label(raw_label) for raw_label in labels]
        self._dimension = len(features[0])
        self._centroids = self._compute_centroids(features, canonical_labels)
        self._max_distance = self._compute_max_distance(features, canonical_labels, self._centroids)

    @staticmethod
    def _load_dataset(path: Path) -> Dict[str, Sequence]:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("El dataset debe ser un diccionario con claves 'data' y 'labels'")
        if "data" not in payload or "labels" not in payload:
            raise ValueError("El dataset no contiene las claves requeridas 'data' y 'labels'")
        return payload

    @staticmethod
    def _coerce_label(raw_value: str | int) -> str:
        try:
            numeric = int(raw_value)
        except (TypeError, ValueError):
            return str(raw_value)
        return helpers.labels_dict.get(numeric, str(raw_value))

    def _compute_centroids(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> Dict[str, List[float]]:
        centroids: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}

        for sample, label in zip(features, labels):
            if label not in centroids:
                centroids[label] = [0.0] * len(sample)
                counts[label] = 0
            accum = centroids[label]
            for index, value in enumerate(sample):
                accum[index] += float(value)
            counts[label] += 1

        for label, accum in centroids.items():
            divisor = float(counts[label]) or 1.0
            centroids[label] = [value / divisor for value in accum]

        return centroids

    def _compute_max_distance(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[str],
        centroids: Dict[str, List[float]],
    ) -> float:
        max_distance = 0.0
        for sample, label in zip(features, labels):
            centroid = centroids[label]
            distance = self._euclidean_distance(sample, centroid)
            if distance > max_distance:
                max_distance = distance
        return max(max_distance, 1e-6)

    @staticmethod
    def _euclidean_distance(sample: Sequence[float], centroid: Sequence[float]) -> float:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(sample, centroid)))

    def predict(self, features: Iterable[float]) -> Prediction:
        vector = [float(value) for value in features]
        if len(vector) != self._dimension:
            raise ValueError(
                f"Se esperaban vectores de dimensión {self._dimension}, pero se recibió uno de tamaño {len(vector)}"
            )

        best_label = None
        best_distance = float("inf")
        for label, centroid in self._centroids.items():
            distance = self._euclidean_distance(vector, centroid)
            if distance < best_distance:
                best_distance = distance
                best_label = label

        if best_label is None:
            raise RuntimeError("No se pudo determinar el gesto más cercano")

        score = max(0.0, 1.0 - (best_distance / self._max_distance))
        return Prediction(label=best_label, score=score)


class SyntheticGestureStream:
    """Yield dataset samples in a loop adding small random jitter."""

    def __init__(self, dataset_path: str | bytes | Path, *, jitter: float = 0.02) -> None:
        payload = SimpleGestureClassifier._load_dataset(Path(dataset_path).expanduser().resolve())
        self._features: Sequence[Sequence[float]] = payload["data"]
        self._labels: Sequence[str | int] = payload["labels"]
        if not self._features or not self._labels:
            raise ValueError("El dataset está vacío")
        if len(self._features) != len(self._labels):
            raise ValueError("El dataset tiene longitudes inconsistentes")

        self._index = 0
        self._rng = random.Random()
        self._jitter = float(max(0.0, jitter))
        self._canonical_labels = [SimpleGestureClassifier._coerce_label(label) for label in self._labels]

    def __iter__(self) -> Iterator[Tuple[List[float], str]]:
        while True:
            yield self.next()

    def next(self) -> Tuple[List[float], str]:
        sample = list(self._features[self._index])
        label = self._canonical_labels[self._index]
        self._index = (self._index + 1) % len(self._features)

        if self._jitter:
            sample = [float(value) + self._rng.gauss(0.0, self._jitter) for value in sample]

        return sample, label

    def reset(self) -> None:
        self._index = 0


__all__ = ["Prediction", "SimpleGestureClassifier", "SyntheticGestureStream"]
