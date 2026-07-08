"""
Aurika Tracking v2 — Model Resolver
====================================
Single source of truth for model paths across execution environments.

Supported environments
-----------------------
LOCAL   : returns  Path("models/<name>.pt")   (file must exist under PROJECT_ROOT)
KAGGLE  : returns  "<name>.pt"                (Ultralytics auto-downloads from hub)

Environment detection
---------------------
Kaggle is detected by the presence of the ``/kaggle`` directory OR the
``KAGGLE_KERNEL_RUN_TYPE`` environment variable.

Usage
-----
    from scripts.model_resolver import ModelResolver

    resolver = ModelResolver()
    path_or_id = resolver.resolve("yolo11l")  # → Path (local) or str (Kaggle)

    registry = resolver.build_registry(MODELS_META)
"""

import logging
import os
from pathlib import Path
from typing import Dict, Union

log = logging.getLogger("Benchmark")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel."""
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or Path("/kaggle").exists()
    )


class ModelResolver:
    """
    Resolves a YOLO11 model name to the correct path or hub identifier
    for the current execution environment.

    Parameters
    ----------
    project_root : Path, optional
        Override the project root (useful for tests).
    kaggle : bool, optional
        Override environment detection (useful for tests).
    """

    def __init__(
        self,
        project_root: Path = _PROJECT_ROOT,
        kaggle: bool | None = None,
    ) -> None:
        self.project_root = project_root
        self.is_kaggle: bool = _is_kaggle() if kaggle is None else kaggle
        env_label = "Kaggle" if self.is_kaggle else "Local"
        log.debug(f"[ModelResolver] environment = {env_label}")

    def resolve(self, name: str) -> Union[Path, str]:
        """
        Return the model path or Ultralytics identifier for *name*.

        Parameters
        ----------
        name : str
            Model name, e.g. ``"yolo11l"`` or ``"yolo11l.pt"``.

        Returns
        -------
        Path
            Absolute path to the local weight file.
        str
            Ultralytics hub identifier (Kaggle environment).
        """
        fname = f"{name}.pt" if not name.endswith(".pt") else name

        if self.is_kaggle:
            log.debug(f"[ModelResolver] {name} → kaggle  '{fname}'")
            return fname

        local_path = self.project_root / "models" / fname
        if local_path.exists():
            log.debug(f"[ModelResolver] {name} → local   {local_path}")
            return local_path

        # Model not found locally — let Ultralytics attempt a download
        log.warning(
            f"[ModelResolver] '{name}' not found at {local_path}. "
            f"Falling back to Ultralytics auto-download ('{fname}')."
        )
        return fname

    def build_registry(self, models_meta: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Accept a metadata dict and return a complete registry with
        ``"path"`` filled in by the resolver for each entry.

        Expected schema per model
        --------------------------
        {
            "label":          str,         # human-readable name
            "person_classes": List[int],   # COCO class IDs to keep
        }

        Returns
        -------
        Dict[str, Dict]
            Same keys, each entry extended with ``"path": Path | str``.
        """
        registry: Dict[str, Dict] = {}
        for key, meta in models_meta.items():
            entry = dict(meta)
            entry["path"] = self.resolve(key)
            registry[key] = entry
        return registry
