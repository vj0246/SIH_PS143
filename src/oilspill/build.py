"""End to end ARD build: adapter to feature stack to tiles to manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import get_adapter
from .adapters.base import Sample
from .ard.features import build_feature_stack, scene_wind_summary
from .ard.tiling import TileSpec, tile_sample
from .io import TileWriter
from .manifest import build_manifest, tile_record
from .schema import CLASS_ID_BY_NAME

LAND_ID = CLASS_ID_BY_NAME["land"]


@dataclass
class BuildConfig:
    """Parameters for one ARD build.

    Attributes:
        channels: Feature channels to compute. ``None`` means every channel
            whose inputs are available.
        tile: Tiling configuration.
        shard_size: Tiles per shard in the output store.
        phi: Assumed wind to look angle for the CMOD inversion.
        pixel_metres: Ground sample distance, used by the land distance channel.
        compute_features: When False, only the source's own bands are written.
            Useful for a fast structural check of a new source.
    """

    channels: list[str] | None = None
    tile: TileSpec = field(default_factory=TileSpec)
    shard_size: int = 512
    phi: float = 45.0
    pixel_metres: float = 10.0
    compute_features: bool = True


def _vv_vh(sample: Sample) -> tuple[np.ndarray | None, np.ndarray | None]:
    names = {n: i for i, n in enumerate(sample.band_names)}
    vv_key = next((k for k in ("vv_db", "vv_render_norm", "vv") if k in names), None)
    vh_key = next((k for k in ("vh_db", "vh") if k in names), None)
    vv = sample.bands[names[vv_key]] if vv_key else None
    vh = sample.bands[names[vh_key]] if vh_key else None
    return vv, vh


def prepare_sample(
    sample: Sample, config: BuildConfig
) -> tuple[np.ndarray, list[str], dict]:
    """Compute the feature stack and the scene level metadata for one sample.

    Returns:
        ``(bands, band_names, scene_meta)``.

    Notes:
        Feature computation is skipped for sources whose radiometry is not
        calibrated sigma0, because CMOD5.N and the damping ratio are both
        defined on sigma0 and applying them to an 8 bit rendering produces
        numbers that look plausible and mean nothing. MKLab is such a source.
    """
    uncalibrated = sample.attrs.get("radiometry") == "8bit_render_not_calibrated"
    vv, vh = _vv_vh(sample)

    if not config.compute_features or uncalibrated or vv is None:
        meta = dict(sample.attrs)
        meta["features_computed"] = False
        if uncalibrated:
            meta["features_skipped_reason"] = "radiometry is not calibrated sigma0"
        return sample.bands, list(sample.band_names), meta

    land_mask = sample.mask == LAND_ID
    bands, names = build_feature_stack(
        vv_db=vv,
        vh_db=vh,
        incidence=sample.incidence if isinstance(sample.incidence, np.ndarray) else None,
        land_mask=land_mask if land_mask.any() else None,
        channels=config.channels,
        phi=config.phi,
        block=min(64, max(8, min(vv.shape) // 4)),
        pixel_metres=config.pixel_metres,
    )

    meta = dict(sample.attrs)
    meta["features_computed"] = True
    meta.update(scene_wind_summary(vv, sample.incidence, phi=config.phi))
    if sample.centroid_lonlat:
        meta["lon"], meta["lat"] = sample.centroid_lonlat
    if sample.acquired:
        meta["acquired"] = sample.acquired
    return bands, names, meta


def build_source(
    source: str,
    raw_root: str | Path,
    out_root: str | Path,
    config: BuildConfig | None = None,
    config_path: str | Path | None = None,
    limit: int | None = None,
    progress=None,
) -> pd.DataFrame:
    """Run the full ARD build for one source.

    Args:
        source: Adapter registry key.
        raw_root: Directory holding that source's distributed files.
        out_root: Directory to write the tile store into.
        config: Build parameters.
        config_path: Alternate classes.yaml.
        limit: Process at most this many scenes. For smoke tests.
        progress: Optional callable taking ``(index, total, sample_id)``.

    Returns:
        The manifest fragment for this source, splits not yet assigned.
    """
    config = config or BuildConfig()
    adapter = get_adapter(source)(raw_root, config_path)
    items = adapter.discover()
    if limit is not None:
        items = items[:limit]

    records: list[dict] = []
    store = Path(out_root) / source

    with TileWriter(store, shard_size=config.shard_size) as writer:
        for i, item in enumerate(items):
            sample = adapter.load_one(item)
            sample.validate()

            bands, names, scene_meta = prepare_sample(sample, config)
            tiles = tile_sample(
                sample_id=sample.sample_id,
                source=sample.source,
                bands=bands,
                mask=sample.mask,
                band_names=names,
                spec=config.tile,
            )
            for tile in tiles:
                ref = writer.add(tile)
                records.append(tile_record(tile, ref, scene_meta))

            if progress is not None:
                progress(i + 1, len(items), sample.sample_id)

    return build_manifest(records)


def build_many(
    sources: dict[str, str | Path],
    out_root: str | Path,
    config: BuildConfig | None = None,
    config_path: str | Path | None = None,
    limit: int | None = None,
    progress=None,
) -> pd.DataFrame:
    """Build several sources and concatenate their manifests."""
    frames = [
        build_source(name, root, out_root, config, config_path, limit, progress)
        for name, root in sources.items()
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return build_manifest([])
    return pd.concat(frames, ignore_index=True).sort_values("tile_id").reset_index(drop=True)
