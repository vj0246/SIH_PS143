"""The manifest: one parquet row per tile, and the single source of truth.

Decision D8. Training reads this table, never the directory tree. Class balance,
split integrity, per-source metrics and stratified sampling then all become one
query, and they stay correct as sources are added.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .ard.tiling import Tile, oil_pixel_fraction
from .io import ShardRef
from .schema import CLASS_ID_BY_NAME

#: Columns every manifest carries. Order is stable so a diff between two
#: manifests is readable.
COLUMNS = [
    "tile_id",
    "sample_id",
    "source",
    "store",
    "shard",
    "shard_index",
    "row",
    "col",
    "height",
    "width",
    "kept_as",
    "label_completeness",
    "n_px_sea",
    "n_px_oil",
    "n_px_look_alike",
    "n_px_ship",
    "n_px_land",
    "n_px_ignore",
    "oil_fraction",
    "supervised_fraction",
    "wind_ms_mean",
    "wind_ms_scene",
    "wind_fraction_observable",
    "observability",
    "lon",
    "lat",
    "acquired",
    "split",
]


def tile_record(
    tile: Tile,
    ref: ShardRef,
    scene_meta: dict | None = None,
) -> dict:
    """Flatten one tile into a manifest row."""
    scene_meta = scene_meta or {}
    counts = tile.counts
    total = int(tile.mask.size)
    ignore = int(counts.get("ignore", 0))

    record = {
        "tile_id": tile.tile_id,
        "sample_id": tile.sample_id,
        "source": tile.source,
        "store": ref.store,
        "shard": ref.shard,
        "shard_index": int(ref.index),
        "row": int(tile.row),
        "col": int(tile.col),
        "height": int(tile.mask.shape[0]),
        "width": int(tile.mask.shape[1]),
        "kept_as": tile.kept_as,
        "label_completeness": scene_meta.get("label_completeness", "unknown"),
        "oil_fraction": float(oil_pixel_fraction(counts)),
        "supervised_fraction": float((total - ignore) / total) if total else 0.0,
        "lon": scene_meta.get("lon"),
        "lat": scene_meta.get("lat"),
        "acquired": scene_meta.get("acquired"),
        "split": "",
    }
    for name in CLASS_ID_BY_NAME:
        record[f"n_px_{name}"] = int(counts.get(name, 0))
    record["n_px_ignore"] = ignore

    for key in (
        "wind_ms_mean",
        "wind_ms_scene",
        "wind_fraction_observable",
        "observability",
    ):
        record[key] = scene_meta.get(key)

    return record


def build_manifest(records: Iterable[dict]) -> pd.DataFrame:
    """Assemble rows into a manifest with a stable column order and dtypes."""
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMNS]
    return frame.sort_values("tile_id").reset_index(drop=True)


def write_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Per source summary: tile counts, class pixel shares and observability.

    Reporting per source is not optional. Decision D2 means supervision differs
    between sources, so a single pooled number hides which sources are actually
    carrying the look-alike signal.
    """
    if frame.empty:
        return pd.DataFrame()

    pixel_cols = [f"n_px_{n}" for n in CLASS_ID_BY_NAME] + ["n_px_ignore"]
    grouped = frame.groupby("source", dropna=False)

    out = grouped.agg(
        n_tiles=("tile_id", "count"),
        n_scenes=("sample_id", "nunique"),
        oil_tiles=("kept_as", lambda s: int((s == "oil").sum())),
        look_alike_tiles=("kept_as", lambda s: int((s == "look_alike").sum())),
        negative_tiles=("kept_as", lambda s: int((s == "negative").sum())),
        mean_oil_fraction=("oil_fraction", "mean"),
        mean_supervised_fraction=("supervised_fraction", "mean"),
    )

    pixels = grouped[pixel_cols].sum()
    total_px = pixels.sum(axis=1).replace(0, pd.NA)
    for col in pixel_cols:
        out[col.replace("n_px_", "pct_")] = (pixels[col] / total_px * 100).round(3)

    if "observability" in frame.columns:
        obs = grouped["observability"].apply(
            lambda s: float((s == "observable").mean()) if s.notna().any() else float("nan")
        )
        out["frac_scenes_observable"] = obs.round(3)

    return out.reset_index()


def integrity_report(frame: pd.DataFrame) -> dict:
    """Checks that catch the failure modes this pipeline is designed against.

    Returns a dict of findings. An empty ``errors`` list means the manifest is
    internally consistent; ``warnings`` are things a human should look at.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if frame.empty:
        return {"errors": ["manifest is empty"], "warnings": []}

    dupes = frame["tile_id"].duplicated().sum()
    if dupes:
        errors.append(f"{dupes} duplicate tile_id values")

    address = ["store", "shard", "shard_index"]
    if frame[address].duplicated().any():
        n = int(frame[address].duplicated().sum())
        errors.append(
            f"{n} tiles share a (store, shard, shard_index) address; training would "
            "read the wrong pixels for those rows"
        )

    # A source declared as binary must never assert non-oil classes.
    binary = frame[frame["label_completeness"] == "binary_oil_only"]
    if not binary.empty:
        leaked = binary[["n_px_sea", "n_px_look_alike", "n_px_ship", "n_px_land"]].sum().sum()
        if leaked:
            errors.append(
                f"binary-labelled sources assert {int(leaked)} pixels of classes they "
                "cannot express; decision D2 has been violated"
            )

    if (frame["oil_fraction"] > 1.0).any() or (frame["oil_fraction"] < 0.0).any():
        errors.append("oil_fraction outside [0, 1]")

    oil_share = float(frame["n_px_oil"].sum()) / max(
        float(frame[[c for c in frame.columns if c.startswith("n_px_")]].sum().sum()), 1.0
    )
    if oil_share < 0.001:
        warnings.append(
            f"oil is {oil_share * 100:.3f} percent of all pixels; consider lowering "
            "negative_keep_ratio or the metrics will be dominated by empty water"
        )

    if frame["split"].eq("").all():
        warnings.append("no split assigned yet; run `oilspill split`")

    if frame["lon"].isna().all():
        warnings.append(
            "no tile carries lon/lat, so spatially blocked splits are impossible; "
            "scene-blocked splitting is the strongest available fallback"
        )

    return {"errors": errors, "warnings": warnings}
