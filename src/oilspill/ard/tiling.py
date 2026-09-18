"""Deterministic, content aware tiling.

Decision D10. Tiles are produced in a fixed raster order with a fixed id, so the
manifest is meaningful and a run is reproducible. Pure-sea tiles are subsampled
under a recorded seed because oil occupies a tiny pixel fraction and an
unbalanced manifest makes every downstream statistic meaningless.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..schema import IGNORE_ID, CLASS_ID_BY_NAME, class_pixel_counts

OIL_ID = CLASS_ID_BY_NAME["oil"]
LOOK_ALIKE_ID = CLASS_ID_BY_NAME["look_alike"]


@dataclass(frozen=True)
class TileSpec:
    """Tiling configuration.

    Attributes:
        size: Tile edge in pixels.
        overlap: Overlap between neighbouring tiles in pixels. Overlap matters
            because a slick straddling a tile boundary is otherwise seen only as
            two truncated fragments, and elongated discharge slicks are the
            common case.
        negative_keep_ratio: Fraction of tiles containing neither oil nor
            look-alike to retain.
        min_valid_fraction: Reject a tile whose non-ignore pixel fraction falls
            below this, which drops scene edge padding.
        seed: Seed for negative subsampling. Recorded in the manifest.
        drop_all_ignore: Drop tiles that carry no supervision at all.
    """

    size: int = 256
    overlap: int = 32
    negative_keep_ratio: float = 0.15
    min_valid_fraction: float = 0.25
    seed: int = 20260910
    drop_all_ignore: bool = True

    def __post_init__(self) -> None:
        if self.overlap >= self.size:
            raise ValueError(f"overlap {self.overlap} must be smaller than size {self.size}")
        if not 0.0 <= self.negative_keep_ratio <= 1.0:
            raise ValueError("negative_keep_ratio must lie in [0, 1]")

    @property
    def stride(self) -> int:
        return self.size - self.overlap


@dataclass
class Tile:
    """One extracted tile and everything the manifest needs about it."""

    tile_id: str
    sample_id: str
    source: str
    row: int
    col: int
    bands: np.ndarray
    mask: np.ndarray
    band_names: list[str]
    counts: dict[str, int] = field(default_factory=dict)
    kept_as: str = ""


def _origins(extent: int, size: int, stride: int) -> list[int]:
    """Tile origins along one axis, with the last tile flush to the edge.

    Flushing the final tile rather than padding it means every tile carries real
    data, at the cost of a wider overlap in the last step.
    """
    if extent <= size:
        return [0]
    origins = list(range(0, extent - size + 1, stride))
    if origins[-1] != extent - size:
        origins.append(extent - size)
    return origins


def _stable_unit_interval(key: str) -> float:
    """Deterministic value in [0, 1) from a string, stable across processes.

    ``random.Random(hash(s))`` is not usable here: Python string hashing is
    salted per process, so the same tile would be kept in one run and dropped in
    the next. Blake2b gives the same answer everywhere, forever.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def classify_tile(counts: dict[str, int]) -> str:
    """Label a tile by what supervision it carries.

    Returns one of ``oil``, ``look_alike``, ``negative`` or ``empty``.
    """
    if counts.get("oil", 0) > 0:
        return "oil"
    if counts.get("look_alike", 0) > 0:
        return "look_alike"
    labelled = sum(v for k, v in counts.items() if k != "ignore")
    return "negative" if labelled > 0 else "empty"


def tile_sample(
    sample_id: str,
    source: str,
    bands: np.ndarray,
    mask: np.ndarray,
    band_names: list[str],
    spec: TileSpec | None = None,
) -> list[Tile]:
    """Cut one scene into tiles.

    Positive tiles, meaning those containing oil or look-alike, are always kept.
    Negatives are kept deterministically at ``negative_keep_ratio`` using a hash
    of the tile id, so the decision is reproducible without carrying a random
    number generator's state through the pipeline.

    Args:
        sample_id: Scene identifier.
        source: Source registry key.
        bands: ``(C, H, W)`` stack.
        mask: ``(H, W)`` unified label mask.
        band_names: Channel names matching ``bands``.
        spec: Tiling configuration.

    Returns:
        Tiles in deterministic raster order.
    """
    spec = spec or TileSpec()
    if bands.ndim != 3:
        raise ValueError(f"bands must be (C, H, W), got {bands.shape}")
    height, width = mask.shape
    tiles: list[Tile] = []

    for row in _origins(height, spec.size, spec.stride):
        for col in _origins(width, spec.size, spec.stride):
            sub_mask = mask[row : row + spec.size, col : col + spec.size]
            counts = class_pixel_counts(sub_mask)

            total = sub_mask.size
            valid = total - counts.get("ignore", 0)
            if spec.drop_all_ignore and valid == 0:
                continue
            if valid / total < spec.min_valid_fraction:
                continue

            kind = classify_tile(counts)
            tile_id = f"{source}:{sample_id}:{row:06d}:{col:06d}"

            if kind == "negative":
                threshold = _stable_unit_interval(f"{spec.seed}:{tile_id}")
                if threshold >= spec.negative_keep_ratio:
                    continue

            tiles.append(
                Tile(
                    tile_id=tile_id,
                    sample_id=sample_id,
                    source=source,
                    row=row,
                    col=col,
                    bands=bands[:, row : row + spec.size, col : col + spec.size].copy(),
                    mask=sub_mask.copy(),
                    band_names=list(band_names),
                    counts=counts,
                    kept_as=kind,
                )
            )
    return tiles


def oil_pixel_fraction(counts: dict[str, int]) -> float:
    """Oil pixels as a fraction of supervised pixels, 0 when nothing is supervised."""
    labelled = sum(v for k, v in counts.items() if k != "ignore")
    return (counts.get("oil", 0) / labelled) if labelled else 0.0
