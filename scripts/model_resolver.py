"""
Aurika Tracking v2 — Model Resolver
====================================
Single source of truth for model paths across environments.

Rules
-----
- Local   : return  Path("models/<name>.pt")  (file must exist under PROJECT_ROOT)
- Kaggle  : return  "<name>.pt"               (Ultralytics auto-downloads from hub)
- Custom  : an entry with an explicit ``custom_path`` is always used as-is, after
            resolving it relative to PROJECT_ROOT when it is a relative path.

Usage
-----
    from scripts.model_resolver import ModelResolver

    resolver = ModelResolver()          # auto-detects environment
    path_or_id = resolver.resolve("yolo11l")   # → Path or str

    # Full registry — same dict schema as benchmark.py MODELS, but with
    # "path" keys filled in by the resolver instead of hardcoded.
    registry = resolver.build_registry(MODELS_META)

Environment detection
---------------------
Kaggle is detected by the presence of the ``/kaggle`` directory OR the
``KAGGLE_KERNEL_RUN_TYPE`` environment variable, matching the same
heuristic used by the production run.py.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Union

log = logging.getLogger("Benchmark")

# Project root is the parent of the scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel."""
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or Path("/kaggle").exists()
    )


class ModelResolver:
    """
    Resolves a model name or custom path to what YOLO() should receive.

    Parameters
    ----------
    project_root : Path, optional
        Override the project root (useful for tests). Defaults to the
        parent of the scripts/ directory.
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

    def resolve(
        self,
        name: str,
        custom_path: str | None = None,
    ) -> Union[Path, str]:
        """
        Return the model path or Ultralytics identifier for *name*.

        Parameters
        ----------
        name : str
            Registry key, e.g. ``"yolo11l"`` or ``"yolo_staff_customer"``.
        custom_path : str | None
            When provided, always take precedence over auto-resolution.
            Relative paths are resolved from PROJECT_ROOT.

        Returns
        -------
        Path
            When a local file exists and should be loaded from disk.
        str
            When the model should be fetched from Ultralytics hub
            (Kaggle) or when custom_path is an absolute string.
        """
        # ── Explicit custom path always wins ────────────────────────────────
        if custom_path is not None:
            p = Path(custom_path)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.exists():
                raise FileNotFoundError(
                    f"Custom model path does not exist: {p}\n"
                    f"(resolved from custom_path={custom_path!r})"
                )
            log.debug(f"[ModelResolver] {name} → custom  {p}")
            return p

        # ── Kaggle: return bare model name; Ultralytics downloads it ────────
        if self.is_kaggle:
            identifier = f"{name}.pt" if not name.endswith(".pt") else name
            log.debug(f"[ModelResolver] {name} → kaggle  '{identifier}'")
            return identifier

        # ── Local: look under models/ ────────────────────────────────────────
        fname = f"{name}.pt" if not name.endswith(".pt") else name
        local_path = self.project_root / "models" / fname
        if local_path.exists():
            log.debug(f"[ModelResolver] {name} → local   {local_path}")
            return local_path

        # ── Local fallback: let Ultralytics try to download ─────────────────
        log.warning(
            f"[ModelResolver] '{name}' not found at {local_path}. "
            f"Falling back to Ultralytics auto-download ('{fname}')."
        )
        return fname

    def build_registry(self, models_meta: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Accept a metadata dict (without ``"path"`` keys) and return a
        complete registry with ``"path"`` filled in by the resolver.

        Expected input schema per model
        --------------------------------
        {
            "label":          str,             # human-readable name
            "person_classes": List[int],       # COCO/fine-tuned class IDs
            "custom_path":    str | None,      # optional override
        }

        Returns
        -------
        Dict[str, Dict]
            Same keys, each entry extended with ``"path": Path | str``.
        """
        registry: Dict[str, Dict] = {}
        for key, meta in models_meta.items():
            entry = dict(meta)  # shallow copy — never mutate caller's dict
            entry["path"] = self.resolve(
                name=key,
                custom_path=meta.get("custom_path"),
            )
            registry[key] = entry
        return registry
