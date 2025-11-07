"""Utility package bridging classic Helen gesture helpers with the new pipeline.

The original project exposed several loose modules (``helpers.py``,
``backendConexion.py``, ``simple_classifier.py``) directly inside the
``Hellen_model_RN`` folder.  The refreshed repository keeps the video pipeline
inside ``video_gesture_model/`` but the backend still imports those helper
modules.  This package re-introduces them so both the legacy tests and the new
backend can operate without modification.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
__all__ = ["PACKAGE_ROOT"]
