"""Helper utilities exposing the gesture label map used across the project."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent

LABELS_PATH_ENV = "HELEN_LABELS_JSON"
MODELS_DIR = PACKAGE_ROOT / "video_gesture_model" / "models"


def _normalise_label(value: str) -> str:
    """Return a human readable label (title-case without leading/trailing spaces)."""
    collapsed = value.strip()
    if not collapsed:
        return ""
    if collapsed.isupper() or collapsed.islower():
        return collapsed.capitalize()
    return collapsed


def _candidate_label_files() -> Iterator[Path]:
    """Yield all ``labels.json`` files bundled with the saved models."""
    if MODELS_DIR.exists():
        for entry in sorted(MODELS_DIR.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".json":
                if entry.name.lower().startswith("labels"):
                    yield entry
            elif entry.is_dir():
                candidate = entry / "labels.json"
                if candidate.exists():
                    yield candidate


def _resolve_labels_path(path: str | os.PathLike[str] | None) -> Path:
    """Resolve the labels file either from an explicit path or the latest model."""
    if path:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"No se encontró el archivo de etiquetas en {resolved!s}")
        return resolved

    env_value = os.getenv(LABELS_PATH_ENV)
    if env_value:
        return _resolve_labels_path(env_value)

    candidates = list(_candidate_label_files())
    if not candidates:
        raise FileNotFoundError(
            "No se encontraron archivos labels.json dentro de Hellen_model_RN/video_gesture_model/models"
        )

    # Prefer the newest file according to modification time.
    newest = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return newest


@lru_cache(maxsize=4)
def load_labels_dict(path: str | os.PathLike[str] | None = None) -> Dict[int, str]:
    """Load the gesture labels dictionary from ``labels.json``."""
    labels_path = _resolve_labels_path(path)
    data = json.loads(labels_path.read_text(encoding="utf-8"))

    mapping: Dict[int, str] = {}
    for gesture, index in data.items():
        try:
            numeric = int(index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"El archivo de etiquetas {labels_path!s} contiene un índice inválido para '{gesture}': {index!r}"
            ) from error
        mapping[numeric] = _normalise_label(gesture)

    return dict(sorted(mapping.items()))


def iter_labels() -> Iterable[Tuple[int, str]]:
    """Iterate over the known labels as ``(index, label)`` tuples."""
    return load_labels_dict().items()


labels_dict: Dict[int, str] = load_labels_dict()
"""Dictionary mapping numeric class ids to the display label used by the UI."""

__all__ = ["LABELS_PATH_ENV", "load_labels_dict", "labels_dict", "iter_labels"]
