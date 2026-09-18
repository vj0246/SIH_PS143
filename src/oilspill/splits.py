"""Blocked train / validation / test assignment.

Decision D3. Random per-tile splitting leaks: neighbouring tiles from one
Sentinel-1 scene share the same slick, sea state, wind field, incidence ramp and
speckle realisation. A model can score well on a random split without having
learned anything transferable, which is how a detector that reports 68 percent
mIoU at home lands at 52 percent in another basin.

Four strategies, in descending order of how honestly they measure generalisation:

``cross_source``
    Train on some sources, test on others. Strictly the hardest, and the closest
    proxy available for "does this work on data we did not build it from".
``spatial``
    Group tiles into geographic cells and assign whole cells. Needs lon/lat.
``scene``
    Group by scene id. Weakest of the blocked strategies but still removes the
    dominant leak, and it is the only one available for benchmark datasets that
    ship no geolocation.
``random``
    Provided only so the inflation can be measured and reported. Never use it
    for a headline number.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

VALID_STRATEGIES = ("cross_source", "spatial", "scene", "random")


@dataclass(frozen=True)
class SplitConfig:
    """Split parameters.

    Attributes:
        strategy: One of :data:`VALID_STRATEGIES`.
        train: Target train fraction, by group not by tile.
        val: Target validation fraction.
        cell_degrees: Cell edge for the ``spatial`` strategy.
        seed: Assignment seed. Recorded so a split is reproducible.
        test_sources: Sources held out entirely under ``cross_source``.
    """

    strategy: str = "scene"
    train: float = 0.7
    val: float = 0.15
    cell_degrees: float = 1.0
    seed: int = 20260910
    test_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"unknown strategy {self.strategy!r}; valid are {VALID_STRATEGIES}"
            )
        if self.train + self.val >= 1.0:
            raise ValueError(
                f"train {self.train} plus val {self.val} leaves nothing for test"
            )
        if self.strategy == "cross_source" and not self.test_sources:
            raise ValueError("cross_source requires test_sources")


def _group_key(frame: pd.DataFrame, config: SplitConfig) -> pd.Series:
    if config.strategy == "scene":
        return frame["source"].astype(str) + "|" + frame["sample_id"].astype(str)

    if config.strategy == "spatial":
        if frame["lon"].isna().all() or frame["lat"].isna().all():
            raise ValueError(
                "spatial splitting needs lon/lat on every tile, but the manifest has "
                "none. Use strategy='scene' instead, and record in the writeup that "
                "geographic blocking was not possible for these sources."
            )
        missing = int(frame["lon"].isna().sum())
        if missing:
            raise ValueError(
                f"{missing} tiles have no lon/lat; fix the manifest or filter them "
                "out explicitly rather than letting them fall into one bucket"
            )
        cell = config.cell_degrees
        lon_cell = (frame["lon"].astype(float) / cell).apply(lambda v: int(v // 1))
        lat_cell = (frame["lat"].astype(float) / cell).apply(lambda v: int(v // 1))
        return lon_cell.astype(str) + "|" + lat_cell.astype(str)

    if config.strategy == "random":
        return frame["tile_id"].astype(str)

    return frame["source"].astype(str)


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_splits(frame: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    """Return a copy of ``frame`` with the ``split`` column filled in.

    Assignment is by hash of the group key, not by shuffling, so it is
    reproducible across machines and stable when new tiles are appended: adding
    data never reshuffles what was already assigned, which matters because a
    reshuffle silently invalidates every previously reported number.
    """
    config = config or SplitConfig()
    out = frame.copy()

    if config.strategy == "cross_source":
        held = set(config.test_sources)
        unknown = held - set(out["source"].unique())
        if unknown:
            raise ValueError(f"test_sources not present in the manifest: {sorted(unknown)}")
        is_test = out["source"].isin(held)
        keys = out.loc[~is_test, "source"].astype(str) + "|" + out.loc[
            ~is_test, "sample_id"
        ].astype(str)
        fractions = keys.map(lambda k: _stable_fraction(k, config.seed))
        train_share = config.train / (config.train + config.val)
        out.loc[~is_test, "split"] = fractions.map(
            lambda f: "train" if f < train_share else "val"
        )
        out.loc[is_test, "split"] = "test"
        return out

    keys = _group_key(out, config)
    fractions = keys.map(lambda k: _stable_fraction(k, config.seed))
    out["split"] = pd.cut(
        fractions,
        bins=[-0.001, config.train, config.train + config.val, 1.001],
        labels=["train", "val", "test"],
    ).astype(str)
    return out


def verify_disjoint(frame: pd.DataFrame, config: SplitConfig) -> dict:
    """Confirm no group straddles two splits, and report the split sizes.

    Returns:
        A report dict. ``leaked_groups`` must be empty; anything else means the
        split is not doing its job and every metric computed on it is inflated.
    """
    keys = _group_key(frame, config)
    grouped = pd.DataFrame({"key": keys, "split": frame["split"]})
    per_group = grouped.groupby("key")["split"].nunique()
    leaked = sorted(per_group[per_group > 1].index.tolist())

    sizes = frame["split"].value_counts().to_dict()
    scenes = frame.groupby("split")["sample_id"].nunique().to_dict()

    report = {
        "strategy": config.strategy,
        "n_groups": int(per_group.size),
        "leaked_groups": leaked[:20],
        "n_leaked_groups": len(leaked),
        "tiles_per_split": {k: int(v) for k, v in sizes.items()},
        "scenes_per_split": {k: int(v) for k, v in scenes.items()},
    }

    oil_by_split = frame.groupby("split")["n_px_oil"].sum().to_dict()
    report["oil_pixels_per_split"] = {k: int(v) for k, v in oil_by_split.items()}
    for split in ("train", "val", "test"):
        if oil_by_split.get(split, 0) == 0:
            report.setdefault("warnings", []).append(
                f"split {split!r} contains no oil pixels at all"
            )
    return report
