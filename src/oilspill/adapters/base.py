"""Adapter contract: every dataset becomes a stream of :class:`Sample`.

Decision D6. Source specific knowledge lives in exactly one file per source and
nowhere else in the codebase.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..schema import SourceSpec, load_source_spec


@dataclass
class Sample:
    """One georeferenced scene or patch with a unified label mask.

    Attributes:
        sample_id: Stable identifier, unique within the source.
        source: Source registry key.
        bands: ``(C, H, W)`` float32 stack. Channel order is given by
            ``band_names``. Backscatter channels are in dB.
        band_names: Names of the channels in ``bands``, for example
            ``["vv_db", "vh_db"]``.
        mask: ``(H, W)`` uint8 unified class ids, with 255 for ignore.
        incidence: Per pixel incidence angle in degrees, or a scalar, or None
            when the source does not carry it.
        geotransform: GDAL style affine ``(x0, dx, rx, y0, ry, dy)`` in
            ``crs``, or None if the source is not georeferenced.
        crs: EPSG code as an integer, or None.
        acquired: ISO 8601 acquisition timestamp, or None.
        centroid_lonlat: ``(lon, lat)`` of the patch centre in degrees, or None.
        attrs: Free form source specific metadata carried through to the
            manifest without interpretation.
    """

    sample_id: str
    source: str
    bands: np.ndarray
    band_names: list[str]
    mask: np.ndarray
    incidence: np.ndarray | float | None = None
    geotransform: tuple[float, float, float, float, float, float] | None = None
    crs: int | None = None
    acquired: str | None = None
    centroid_lonlat: tuple[float, float] | None = None
    attrs: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Check internal consistency. Raises ``ValueError`` on any violation."""
        if self.bands.ndim != 3:
            raise ValueError(f"{self.sample_id}: bands must be (C, H, W), got {self.bands.shape}")
        if self.bands.shape[0] != len(self.band_names):
            raise ValueError(
                f"{self.sample_id}: {self.bands.shape[0]} channels but "
                f"{len(self.band_names)} band names"
            )
        if self.mask.ndim != 2:
            raise ValueError(f"{self.sample_id}: mask must be (H, W), got {self.mask.shape}")
        if self.mask.shape != self.bands.shape[1:]:
            raise ValueError(
                f"{self.sample_id}: mask {self.mask.shape} does not match "
                f"bands {self.bands.shape[1:]}"
            )
        if self.mask.dtype != np.uint8:
            raise ValueError(f"{self.sample_id}: mask must be uint8, got {self.mask.dtype}")
        if isinstance(self.incidence, np.ndarray) and self.incidence.shape not in (
            (),
            self.mask.shape,
        ):
            raise ValueError(
                f"{self.sample_id}: incidence {self.incidence.shape} does not match "
                f"mask {self.mask.shape}"
            )


class DatasetAdapter(abc.ABC):
    """Base class for source adapters.

    Subclasses implement :meth:`discover` and :meth:`load_one`. The base class
    supplies the source specification, the iteration protocol and the validation
    that every emitted sample is well formed.
    """

    #: Registry key, must match a key under ``sources:`` in classes.yaml.
    name: str = ""

    def __init__(self, root: str | Path, config_path: str | Path | None = None) -> None:
        self.root = Path(root)
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a class level `name`")
        self.spec: SourceSpec = load_source_spec(self.name, config_path)

    @abc.abstractmethod
    def discover(self) -> list[dict]:
        """Enumerate available items without reading pixel data.

        Returns:
            A list of opaque descriptors, each of which :meth:`load_one` accepts.
            Sorted deterministically so tile ids are stable across runs.
        """

    @abc.abstractmethod
    def load_one(self, item: dict) -> Sample:
        """Materialise one descriptor into a validated :class:`Sample`."""

    def __iter__(self) -> Iterator[Sample]:
        for item in self.discover():
            sample = self.load_one(item)
            sample.validate()
            yield sample

    def __len__(self) -> int:
        return len(self.discover())

    def describe(self) -> dict:
        """Summary for logs and for the manifest provenance columns."""
        return {
            "source": self.name,
            "root": str(self.root),
            "expresses": sorted(self.spec.expresses),
            "encoding": self.spec.encoding,
            "background_policy": self.spec.background_policy,
            "provisional_labels": self.spec.provisional,
            "n_items": len(self.discover()),
        }


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Class decorator adding an adapter to the registry."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> type[DatasetAdapter]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"no adapter registered for {name!r}; registered adapters are "
            f"{sorted(_REGISTRY)}"
        ) from exc


def registered() -> list[str]:
    return sorted(_REGISTRY)
