"""Unified label schema and per-source remapping.

Implements decisions D1, D2 and D7: one 5 class schema, an explicit ignore
class for supervision a source cannot provide, and hard failure on any label
value that is not declared in the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

IGNORE_ID = 255

_CONFIG_ENV = "OILSPILL_CLASSES_CONFIG"
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "classes.yaml"


class SchemaError(ValueError):
    """Raised when a mask contains a label value the source has not declared."""


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str
    colour: tuple[int, int, int]
    description: str


@dataclass(frozen=True)
class SourceSpec:
    """How one dataset encodes labels, and what it is able to express.

    Attributes:
        name: Registry key, matching the key in ``configs/classes.yaml``.
        expresses: Unified class names this source provides supervision for.
        encoding: One of ``palette_png``, ``binary_raster``, ``index_raster``
            or ``vector_objects``.
        index_map: Native integer or palette index to unified class id.
        rgb_map: Native RGB triple to unified class id. Empty when unused.
        object_class_map: Native object label string to unified class id.
        background_policy: ``sea`` or ``ignore``. What unannotated pixels
            inside an annotated extent become.
        provisional: True when the mapping was transcribed from a publication
            rather than confirmed against the distributed files.
    """

    name: str
    expresses: frozenset[str]
    encoding: str
    index_map: Mapping[int, int]
    rgb_map: Mapping[tuple[int, int, int], int]
    object_class_map: Mapping[str, int]
    background_policy: str
    provisional: bool

    @property
    def background_id(self) -> int:
        return IGNORE_ID if self.background_policy == "ignore" else 0

    def expressible_ids(self) -> frozenset[int]:
        return frozenset(CLASS_ID_BY_NAME[n] for n in self.expresses)


def _load_yaml(path: Path | None = None) -> dict:
    import os

    if path is None:
        path = Path(os.environ.get(_CONFIG_ENV, _DEFAULT_CONFIG))
    if not path.is_file():
        raise FileNotFoundError(f"class schema config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=4)
def _config(path_str: str | None = None) -> dict:
    return _load_yaml(Path(path_str) if path_str else None)


def _build_classes(cfg: dict) -> tuple[ClassDef, ...]:
    return tuple(
        ClassDef(
            id=int(c["id"]),
            name=str(c["name"]),
            colour=tuple(int(v) for v in c["colour"]),  # type: ignore[arg-type]
            description=str(c.get("description", "")),
        )
        for c in cfg["schema"]["classes"]
    )


CLASSES: tuple[ClassDef, ...] = _build_classes(_config())
CLASS_ID_BY_NAME: dict[str, int] = {c.name: c.id for c in CLASSES}
CLASS_NAME_BY_ID: dict[int, str] = {c.id: c.name for c in CLASSES}
NUM_CLASSES: int = len(CLASSES)


def _resolve_class(token: str) -> int:
    if token == "ignore":
        return IGNORE_ID
    try:
        return CLASS_ID_BY_NAME[token]
    except KeyError as exc:  # pragma: no cover - configuration error path
        raise SchemaError(
            f"unknown class name {token!r} in configuration; "
            f"valid names are {sorted(CLASS_ID_BY_NAME)} or 'ignore'"
        ) from exc


def _parse_rgb_key(key: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(key).split(",")]
    if len(parts) != 3:
        raise SchemaError(f"rgb_map key {key!r} is not three comma separated integers")
    r, g, b = (int(p) for p in parts)
    return (r, g, b)


def load_source_spec(name: str, config_path: str | Path | None = None) -> SourceSpec:
    """Build the :class:`SourceSpec` for ``name`` from configuration."""
    cfg = _config(str(config_path) if config_path else None)
    try:
        raw = cfg["sources"][name]
    except KeyError as exc:
        raise SchemaError(
            f"source {name!r} is not declared in the class schema config; "
            f"declared sources are {sorted(cfg.get('sources', {}))}"
        ) from exc

    return SourceSpec(
        name=name,
        expresses=frozenset(raw.get("expresses", [])),
        encoding=str(raw["encoding"]),
        index_map={int(k): _resolve_class(v) for k, v in (raw.get("index_map") or {}).items()},
        rgb_map={
            _parse_rgb_key(k): _resolve_class(v) for k, v in (raw.get("rgb_map") or {}).items()
        },
        object_class_map={
            str(k): _resolve_class(v) for k, v in (raw.get("object_class_map") or {}).items()
        },
        background_policy=str(raw.get("background_policy", "sea")),
        provisional=bool(raw.get("provisional", False)),
    )


def list_sources(config_path: str | Path | None = None) -> list[str]:
    return sorted(_config(str(config_path) if config_path else None)["sources"])


# ---------------------------------------------------------------------------
# Remapping
# ---------------------------------------------------------------------------


def remap_index_mask(mask: np.ndarray, spec: SourceSpec) -> np.ndarray:
    """Map a single band integer mask into the unified schema.

    Args:
        mask: Integer array of native label values.
        spec: Source specification supplying ``index_map``.

    Returns:
        ``uint8`` array of unified class ids, with :data:`IGNORE_ID` where the
        source declared values it cannot express.

    Raises:
        SchemaError: If ``mask`` contains any value absent from ``index_map``.
            Never silently defaults, per decision D7.
    """
    if not spec.index_map:
        raise SchemaError(f"source {spec.name!r} has no index_map but an index mask was supplied")

    arr = np.asarray(mask)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)):
            raise SchemaError(f"source {spec.name!r}: mask contains non finite values")
        if not np.all(arr == np.round(arr)):
            raise SchemaError(f"source {spec.name!r}: float mask has non integer values")
        arr = arr.astype(np.int64)

    present = np.unique(arr)
    unknown = [int(v) for v in present if int(v) not in spec.index_map]
    if unknown:
        counts = {int(v): int((arr == v).sum()) for v in unknown}
        raise SchemaError(
            f"source {spec.name!r}: mask contains undeclared label values {counts}. "
            f"Declared values are {sorted(spec.index_map)}. "
            "Add them to configs/classes.yaml rather than letting them default."
        )

    lut = np.full(int(present.max()) + 1 if present.size else 1, IGNORE_ID, dtype=np.uint8)
    for native, unified in spec.index_map.items():
        if native >= lut.size:
            lut = np.pad(lut, (0, native - lut.size + 1), constant_values=IGNORE_ID)
        lut[native] = unified
    return lut[arr]


def remap_rgb_mask(mask: np.ndarray, spec: SourceSpec) -> np.ndarray:
    """Map an ``(H, W, 3)`` RGB colour coded mask into the unified schema.

    Raises:
        SchemaError: If any colour present is absent from ``rgb_map``.
    """
    if not spec.rgb_map:
        raise SchemaError(f"source {spec.name!r} has no rgb_map but an RGB mask was supplied")

    arr = np.asarray(mask)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise SchemaError(f"source {spec.name!r}: expected (H, W, 3) RGB mask, got {arr.shape}")
    rgb = arr[..., :3].astype(np.int64)

    # Pack to a single integer per pixel so uniqueness is one pass.
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    lookup = {
        (r << 16) | (g << 8) | b: unified for (r, g, b), unified in spec.rgb_map.items()
    }

    present = np.unique(packed)
    unknown = [int(v) for v in present if int(v) not in lookup]
    if unknown:
        detail = {
            f"{(v >> 16) & 255},{(v >> 8) & 255},{v & 255}": int((packed == v).sum())
            for v in unknown
        }
        raise SchemaError(
            f"source {spec.name!r}: mask contains undeclared colours {detail}. "
            "Add them to configs/classes.yaml rgb_map. "
            + (
                "This source is marked provisional, so a wrong palette is the likely cause."
                if spec.provisional
                else ""
            )
        )

    out = np.full(packed.shape, IGNORE_ID, dtype=np.uint8)
    for value, unified in lookup.items():
        out[packed == value] = unified
    return out


def apply_expressible_mask(mask: np.ndarray, spec: SourceSpec) -> np.ndarray:
    """Force classes the source cannot express to :data:`IGNORE_ID`.

    Decision D2. A source that only labels oil must not assert anything about
    look-alikes, ships or land, so any such pixel becomes ignore even if the
    remap produced a concrete class for it.
    """
    allowed = spec.expressible_ids()
    out = np.asarray(mask).copy()
    keep = np.isin(out, list(allowed)) | (out == IGNORE_ID)
    out[~keep] = IGNORE_ID
    return out


def class_pixel_counts(mask: np.ndarray) -> dict[str, int]:
    """Per class pixel counts for one mask, including ignore."""
    arr = np.asarray(mask)
    counts = {name: 0 for name in CLASS_ID_BY_NAME}
    counts["ignore"] = 0
    values, freq = np.unique(arr, return_counts=True)
    for value, n in zip(values.tolist(), freq.tolist()):
        if value == IGNORE_ID:
            counts["ignore"] += int(n)
        else:
            counts[CLASS_NAME_BY_ID[int(value)]] += int(n)
    return counts


def colourise(mask: np.ndarray, ignore_colour: Sequence[int] = (255, 0, 255)) -> np.ndarray:
    """Render a unified mask to RGB for inspection."""
    arr = np.asarray(mask)
    out = np.zeros(arr.shape + (3,), dtype=np.uint8)
    for c in CLASSES:
        out[arr == c.id] = c.colour
    out[arr == IGNORE_ID] = np.asarray(ignore_colour, dtype=np.uint8)
    return out


def scan_unmapped(masks: Iterable[np.ndarray], spec: SourceSpec) -> dict[str, int]:
    """Report undeclared label values across many masks without raising.

    Used by ``oilspill verify`` so an operator sees every problem at once
    instead of fixing them one exception at a time.
    """
    found: dict[str, int] = {}
    for mask in masks:
        arr = np.asarray(mask)
        if spec.encoding == "palette_png" and arr.ndim == 3:
            rgb = arr[..., :3].astype(np.int64)
            packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
            known = {
                (r << 16) | (g << 8) | b for (r, g, b) in spec.rgb_map
            }
            values, freq = np.unique(packed, return_counts=True)
            for v, n in zip(values.tolist(), freq.tolist()):
                if v not in known:
                    key = f"rgb:{(v >> 16) & 255},{(v >> 8) & 255},{v & 255}"
                    found[key] = found.get(key, 0) + int(n)
        else:
            values, freq = np.unique(arr, return_counts=True)
            for v, n in zip(values.tolist(), freq.tolist()):
                if int(v) not in spec.index_map:
                    key = f"index:{int(v)}"
                    found[key] = found.get(key, 0) + int(n)
    return found
