"""Tiling, store, manifest, splits and the end to end build."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from oilspill.ard.tiling import TileSpec, classify_tile, tile_sample
from oilspill.build import BuildConfig, build_many, build_source
from oilspill.cdse import CdseClient
from oilspill.io import TileReader, TileWriter
from oilspill.manifest import (
    build_manifest,
    integrity_report,
    read_manifest,
    summarise,
    tile_record,
    write_manifest,
)
from oilspill.schema import IGNORE_ID, CLASS_ID_BY_NAME
from oilspill.splits import SplitConfig, assign_splits, verify_disjoint
from tests import fixtures

OIL = CLASS_ID_BY_NAME["oil"]
SEA = CLASS_ID_BY_NAME["sea"]
LOOK = CLASS_ID_BY_NAME["look_alike"]


def _scene(height=512, width=512, oil_box=(100, 160, 100, 200)):
    bands = np.random.default_rng(0).normal(-15, 2, size=(2, height, width)).astype(np.float32)
    mask = np.full((height, width), SEA, dtype=np.uint8)
    r0, r1, c0, c1 = oil_box
    mask[r0:r1, c0:c1] = OIL
    return bands, mask


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


class TestTiling:
    def test_tile_ids_are_deterministic_across_runs(self):
        bands, mask = _scene()
        a = [t.tile_id for t in tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"])]
        b = [t.tile_id for t in tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"])]
        assert a == b

    def test_negative_subsampling_is_stable_across_processes(self):
        """Python string hashing is salted per process. If the tiler used it, the
        same tile would be kept in one run and dropped in the next, and no
        experiment would ever be reproducible."""
        code = (
            "import numpy as np, sys;"
            "sys.path[:0]=['src','.'];"
            "from oilspill.ard.tiling import tile_sample;"
            "from oilspill.schema import CLASS_ID_BY_NAME;"
            "b=np.zeros((1,512,512),dtype=np.float32);"
            "m=np.full((512,512),CLASS_ID_BY_NAME['sea'],dtype=np.uint8);"
            "m[100:160,100:200]=CLASS_ID_BY_NAME['oil'];"
            "print(','.join(t.tile_id for t in tile_sample('s','src',b,m,['vv_db'])))"
        )
        runs = [
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout.strip()
            for _ in range(2)
        ]
        assert runs[0] == runs[1]
        assert runs[0]  # non empty

    def test_every_oil_tile_is_kept(self):
        bands, mask = _scene()
        spec = TileSpec(size=128, overlap=32, negative_keep_ratio=0.0)
        tiles = tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"], spec)
        assert tiles, "oil tiles must survive a zero negative ratio"
        assert all(t.kept_as == "oil" for t in tiles)
        assert all(t.counts["oil"] > 0 for t in tiles)

    def test_negative_ratio_controls_how_many_negatives_survive(self):
        bands, mask = _scene(oil_box=(0, 1, 0, 1))
        spec_none = TileSpec(size=64, overlap=0, negative_keep_ratio=0.0)
        spec_all = TileSpec(size=64, overlap=0, negative_keep_ratio=1.0)
        n_none = len(tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"], spec_none))
        n_all = len(tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"], spec_all))
        assert n_none < n_all
        # 64 x 64 tiles over 512 x 512 with no overlap
        assert n_all == 64

    def test_last_tile_is_flush_with_the_edge(self):
        bands, mask = _scene(height=300, width=300)
        spec = TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)
        tiles = tile_sample("s", "src", bands, mask, ["vv_db", "vh_db"], spec)
        rows = {t.row for t in tiles}
        assert max(rows) == 300 - 128, "final tile must be flush, never padded"
        assert all(t.bands.shape[1:] == (128, 128) for t in tiles)

    def test_overlap_must_be_smaller_than_tile_size(self):
        with pytest.raises(ValueError, match="must be smaller"):
            TileSpec(size=64, overlap=64)

    def test_all_ignore_tiles_are_dropped(self):
        bands = np.zeros((1, 256, 256), dtype=np.float32)
        mask = np.full((256, 256), IGNORE_ID, dtype=np.uint8)
        tiles = tile_sample("s", "src", bands, mask, ["vv_db"], TileSpec(size=128, overlap=0))
        assert tiles == []

    def test_classify_tile(self):
        assert classify_tile({"oil": 5, "sea": 10}) == "oil"
        assert classify_tile({"look_alike": 5, "sea": 10}) == "look_alike"
        assert classify_tile({"sea": 10}) == "negative"
        assert classify_tile({"ignore": 10}) == "empty"
        # Oil wins over look-alike: a tile with both is a positive.
        assert classify_tile({"oil": 1, "look_alike": 99}) == "oil"


# ---------------------------------------------------------------------------
# Tile store
# ---------------------------------------------------------------------------


class TestTileStore:
    def test_round_trip_preserves_pixels_exactly(self, tmp_path):
        bands, mask = _scene(256, 256)
        tiles = tile_sample(
            "s", "src", bands, mask, ["vv_db", "vh_db"], TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)
        )
        refs = {}
        with TileWriter(tmp_path / "store", shard_size=2) as writer:
            for tile in tiles:
                refs[tile.tile_id] = writer.add(tile)

        reader = TileReader(tmp_path / "store")
        assert len(reader) == len(tiles)
        assert reader.band_names == ["vv_db", "vh_db"]

        by_id = {t.tile_id: t for t in tiles}
        for tile_id, ref in refs.items():
            got_bands, got_mask, got_id = reader.read(ref.shard, ref.index)
            assert got_id == tile_id
            assert np.array_equal(got_bands, by_id[tile_id].bands)
            assert np.array_equal(got_mask, by_id[tile_id].mask)

    def test_shards_are_split_at_the_configured_size(self, tmp_path):
        bands, mask = _scene(256, 256)
        tiles = tile_sample(
            "s", "src", bands, mask, ["vv_db", "vh_db"], TileSpec(size=64, overlap=0, negative_keep_ratio=1.0)
        )
        with TileWriter(tmp_path / "store", shard_size=4) as writer:
            writer.extend(tiles)
        reader = TileReader(tmp_path / "store")
        assert len(reader.shards()) == int(np.ceil(len(tiles) / 4))

    def test_mixed_band_layouts_are_refused(self, tmp_path):
        bands, mask = _scene(128, 128)
        tiles = tile_sample(
            "s", "src", bands, mask, ["vv_db", "vh_db"], TileSpec(size=64, overlap=0, negative_keep_ratio=1.0)
        )
        with TileWriter(tmp_path / "store") as writer:
            writer.add(tiles[0])
            tiles[1].band_names = ["vv_db", "something_else"]
            with pytest.raises(ValueError, match="opened with"):
                writer.add(tiles[1])

    def test_reader_without_store_json_fails_clearly(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="was TileWriter closed"):
            TileReader(tmp_path / "empty")

    def test_store_is_part_of_the_address(self, tmp_path):
        """Shard filenames restart per store, so (shard, index) alone collides.

        Caught by the multi-source end to end test: two sources both produced
        tiles_00000.npz index 0, the manifest looked healthy, and half the rows
        would have resolved to the wrong pixels.
        """
        bands, mask = _scene(128, 128)
        spec = TileSpec(size=64, overlap=0, negative_keep_ratio=1.0)

        refs = {}
        for name in ("source_a", "source_b"):
            tiles = tile_sample("s", name, bands, mask, ["vv_db", "vh_db"], spec)
            with TileWriter(tmp_path / "ard" / name) as writer:
                refs[name] = [writer.add(t) for t in tiles]

        a, b = refs["source_a"][0], refs["source_b"][0]
        assert (a.shard, a.index) == (b.shard, b.index), "the collision this guards against"
        assert a.store != b.store

        from oilspill.io import TileStoreSet

        stores = TileStoreSet(tmp_path / "ard")
        assert stores.stores() == ["source_a", "source_b"]
        _, _, tile_id_a = stores.read(a.store, a.shard, a.index)
        _, _, tile_id_b = stores.read(b.store, b.shard, b.index)
        assert tile_id_a != tile_id_b
        assert tile_id_a.startswith("source_a:")
        assert tile_id_b.startswith("source_b:")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def _frame(self):
        bands, mask = _scene(256, 256)
        tiles = tile_sample(
            "s", "src", bands, mask, ["vv_db", "vh_db"], TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)
        )
        from oilspill.io import ShardRef

        records = [
            tile_record(
                t,
                ShardRef("src", "tiles_00000.npz", i),
                {"label_completeness": "full_5class"},
            )
            for i, t in enumerate(tiles)
        ]
        return build_manifest(records)

    def test_columns_and_ordering_are_stable(self):
        from oilspill.manifest import COLUMNS

        frame = self._frame()
        assert list(frame.columns) == COLUMNS
        assert frame["tile_id"].is_monotonic_increasing

    def test_round_trip_through_parquet(self, tmp_path):
        frame = self._frame()
        path = write_manifest(frame, tmp_path / "m.parquet")
        again = read_manifest(path)
        pd.testing.assert_frame_equal(frame, again)

    def test_integrity_catches_a_d2_violation(self):
        """A binary source asserting look-alike pixels is the exact bug
        decision D2 exists to prevent, so the manifest must catch it."""
        frame = self._frame()
        frame["label_completeness"] = "binary_oil_only"
        frame.loc[0, "n_px_look_alike"] = 500

        report = integrity_report(frame)
        assert any("D2" in e for e in report["errors"])

    def test_integrity_catches_duplicate_addresses(self):
        frame = self._frame()
        frame.loc[1, "shard_index"] = frame.loc[0, "shard_index"]
        frame.loc[1, "shard"] = frame.loc[0, "shard"]
        frame.loc[1, "store"] = frame.loc[0, "store"]
        report = integrity_report(frame)
        assert any("shard_index" in e for e in report["errors"])

    def test_integrity_warns_about_unassigned_splits(self):
        report = integrity_report(self._frame())
        assert any("oilspill split" in w for w in report["warnings"])

    def test_summarise_reports_per_source(self):
        frame = self._frame()
        summary = summarise(frame)
        assert list(summary["source"]) == ["src"]
        assert summary.loc[0, "n_tiles"] == len(frame)
        assert "pct_oil" in summary.columns


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _split_frame(n_scenes=20, tiles_per_scene=6, sources=("a", "b")):
    rows = []
    for source in sources:
        for scene in range(n_scenes):
            for tile in range(tiles_per_scene):
                rows.append(
                    {
                        "tile_id": f"{source}:{scene}:{tile}",
                        "sample_id": f"scene_{scene}",
                        "source": source,
                        "lon": -10.0 + scene * 0.7,
                        "lat": 35.0 + (scene % 5) * 0.6,
                        "n_px_oil": 100 if tile % 2 else 0,
                        "split": "",
                    }
                )
    return pd.DataFrame(rows)


class TestSplits:
    def test_scene_blocking_keeps_scenes_whole(self):
        frame = _split_frame()
        config = SplitConfig(strategy="scene")
        out = assign_splits(frame, config)
        report = verify_disjoint(out, config)
        assert report["n_leaked_groups"] == 0
        assert set(out["split"]) <= {"train", "val", "test"}

    def test_random_splitting_leaks_scenes_across_splits(self):
        """The comparison decision D3 rests on. Random splitting is provided
        only so that the inflation it causes can be measured."""
        frame = _split_frame()
        random_config = SplitConfig(strategy="random")
        out = assign_splits(frame, random_config)
        leaked = verify_disjoint(out, SplitConfig(strategy="scene"))
        assert leaked["n_leaked_groups"] > 0, (
            "random splitting should scatter one scene's tiles across splits"
        )

    def test_spatial_blocking_keeps_cells_whole(self):
        frame = _split_frame()
        config = SplitConfig(strategy="spatial", cell_degrees=1.0)
        out = assign_splits(frame, config)
        assert verify_disjoint(out, config)["n_leaked_groups"] == 0

    def test_spatial_blocking_refuses_when_geolocation_is_missing(self):
        frame = _split_frame()
        frame["lon"] = np.nan
        frame["lat"] = np.nan
        with pytest.raises(ValueError, match="needs lon/lat"):
            assign_splits(frame, SplitConfig(strategy="spatial"))

    def test_spatial_blocking_refuses_partial_geolocation(self):
        frame = _split_frame()
        frame.loc[0, "lon"] = np.nan
        with pytest.raises(ValueError, match="no lon/lat"):
            assign_splits(frame, SplitConfig(strategy="spatial"))

    def test_cross_source_holds_out_whole_sources(self):
        frame = _split_frame()
        config = SplitConfig(strategy="cross_source", test_sources=("b",))
        out = assign_splits(frame, config)
        assert set(out.loc[out["source"] == "b", "split"]) == {"test"}
        assert "test" not in set(out.loc[out["source"] == "a", "split"])

    def test_cross_source_rejects_an_unknown_source(self):
        with pytest.raises(ValueError, match="not present in the manifest"):
            assign_splits(
                _split_frame(), SplitConfig(strategy="cross_source", test_sources=("z",))
            )

    def test_assignment_is_stable_when_new_data_is_appended(self):
        """Appending tiles must never reshuffle existing assignments, because a
        reshuffle silently invalidates every number reported before it."""
        frame = _split_frame(n_scenes=10)
        config = SplitConfig(strategy="scene")
        first = assign_splits(frame, config).set_index("tile_id")["split"]

        extra = _split_frame(n_scenes=20)
        extra = extra[extra["sample_id"].str.replace("scene_", "").astype(int) >= 10]
        grown = pd.concat([frame, extra], ignore_index=True)
        assert not grown["tile_id"].duplicated().any()
        second = assign_splits(grown, config).set_index("tile_id")["split"]

        common = first.index.intersection(second.index)
        assert (first.loc[common] == second.loc[common]).all()

    def test_train_val_must_leave_room_for_test(self):
        with pytest.raises(ValueError, match="leaves nothing for test"):
            SplitConfig(train=0.9, val=0.15)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            SplitConfig(strategy="kfold")

    def test_report_warns_when_a_split_has_no_oil(self):
        frame = _split_frame()
        frame["n_px_oil"] = 0
        config = SplitConfig(strategy="scene")
        out = assign_splits(frame, config)
        report = verify_disjoint(out, config)
        assert any("no oil pixels" in w for w in report.get("warnings", []))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_build_one_source_produces_a_consistent_manifest(self, tmp_path):
        raw = fixtures.make_zenodo_s1(tmp_path / "raw", n=2, size=256)
        frame = build_source(
            "zenodo_s1",
            raw,
            tmp_path / "ard",
            BuildConfig(tile=TileSpec(size=128, overlap=32, negative_keep_ratio=1.0)),
        )
        assert not frame.empty
        assert set(frame["source"]) == {"zenodo_s1"}
        assert (frame["label_completeness"] == "binary_oil_only").all()
        assert frame["n_px_sea"].sum() == 0, "D2: binary source must not assert sea"
        assert frame["n_px_oil"].sum() > 0

        report = integrity_report(frame)
        assert report["errors"] == []

    def test_build_writes_a_readable_store_matching_the_manifest(self, tmp_path):
        raw = fixtures.make_zenodo_s1(tmp_path / "raw", n=1, size=256)
        frame = build_source(
            "zenodo_s1",
            raw,
            tmp_path / "ard",
            BuildConfig(tile=TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)),
        )
        reader = TileReader(tmp_path / "ard" / "zenodo_s1")
        assert len(reader) == len(frame)

        from oilspill.io import TileStoreSet

        stores = TileStoreSet(tmp_path / "ard")
        row = frame.iloc[0]
        bands, mask, tile_id = stores.read_row(row)
        assert tile_id == row["tile_id"]
        assert bands.shape[1:] == (row["height"], row["width"])
        assert int((mask == OIL).sum()) == int(row["n_px_oil"])

    def test_feature_channels_reach_the_store(self, tmp_path):
        raw = fixtures.make_zenodo_s1(tmp_path / "raw", n=1, size=256)
        build_source(
            "zenodo_s1",
            raw,
            tmp_path / "ard",
            BuildConfig(tile=TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)),
        )
        reader = TileReader(tmp_path / "ard" / "zenodo_s1")
        assert reader.band_names == [
            "vv_db",
            "vh_db",
            "incidence_deg",
            "wind_ms",
            "damping_db",
            "local_var_db",
        ]

    def test_uncalibrated_source_skips_the_physics_channels(self, tmp_path):
        """MKLab ships 8 bit renderings. Running CMOD5.N on them would produce
        numbers that look fine and mean nothing."""
        raw = fixtures.make_mklab(tmp_path / "mk", n=1, size=256)
        build_source(
            "mklab",
            raw,
            tmp_path / "ard",
            BuildConfig(tile=TileSpec(size=128, overlap=0, negative_keep_ratio=1.0)),
        )
        reader = TileReader(tmp_path / "ard" / "mklab")
        assert reader.band_names == ["vv_render_norm"]

    def test_build_many_then_split_then_report(self, tmp_path):
        sources = {
            "zenodo_s1": fixtures.make_zenodo_s1(tmp_path / "z", n=2, size=256),
            "pangaea_emed": fixtures.make_pangaea_emed(
                tmp_path / "p", n_oil=2, n_clean=1, size=256
            ),
        }
        frame = build_many(
            sources,
            tmp_path / "ard",
            BuildConfig(tile=TileSpec(size=128, overlap=32, negative_keep_ratio=1.0)),
        )
        assert set(frame["source"]) == {"zenodo_s1", "pangaea_emed"}

        config = SplitConfig(strategy="cross_source", test_sources=("pangaea_emed",))
        frame = assign_splits(frame, config)
        report = verify_disjoint(frame, config)
        assert report["n_leaked_groups"] == 0
        assert set(frame.loc[frame["source"] == "pangaea_emed", "split"]) == {"test"}

        summary = summarise(frame)
        assert len(summary) == 2
        # The two sources supervise different things; that must be visible.
        zen = summary[summary["source"] == "zenodo_s1"].iloc[0]
        pan = summary[summary["source"] == "pangaea_emed"].iloc[0]
        assert zen["pct_sea"] == 0
        assert pan["pct_sea"] > 0
        assert integrity_report(frame)["errors"] == []


# ---------------------------------------------------------------------------
# CDSE query construction (no network)
# ---------------------------------------------------------------------------


class TestCdseFilter:
    def test_filter_includes_every_constraint(self):
        f = CdseClient.build_filter(
            start="2024-01-01T00:00:00.000Z",
            end="2024-01-31T00:00:00.000Z",
            bbox=(72.0, 18.0, 73.5, 19.5),
        )
        assert "Collection/Name eq 'SENTINEL-1'" in f
        assert "'productType'" in f and "'GRD'" in f
        assert "'sensorMode'" in f and "'IW'" in f
        assert "ContentDate/Start gt 2024-01-01T00:00:00.000Z" in f
        assert "OData.CSC.Intersects" in f

    def test_polygon_ring_is_closed(self):
        f = CdseClient.build_filter(bbox=(0.0, 0.0, 1.0, 1.0))
        ring = f.split("POLYGON((")[1].split("))")[0]
        points = [p.strip() for p in ring.split(",")]
        assert points[0] == points[-1], "OData requires a closed ring"
        assert len(points) == 5

    def test_polarisation_can_be_left_unconstrained(self):
        assert "polarisationChannels" not in CdseClient.build_filter(polarisation=None)

    def test_download_without_credentials_explains_how_to_get_them(self, monkeypatch):
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)
        client = CdseClient()
        with pytest.raises(Exception, match="CDSE_USERNAME"):
            client.download_product("abc", "/tmp/x.zip")
