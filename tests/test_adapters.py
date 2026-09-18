"""Adapter contract tests, including the decision D2 supervision boundary."""

from __future__ import annotations

import numpy as np
import pytest

from oilspill.adapters import MKLabAdapter, PangaeaEMedAdapter, ZenodoS1Adapter, registered
from oilspill.schema import (
    IGNORE_ID,
    CLASS_ID_BY_NAME,
    SchemaError,
    class_pixel_counts,
    load_source_spec,
)
from tests import fixtures


def test_all_expected_adapters_are_registered():
    assert set(registered()) >= {"mklab", "pangaea_emed", "zenodo_s1"}


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------


class TestZenodoS1:
    @pytest.fixture
    def root(self, tmp_path):
        return fixtures.make_zenodo_s1(tmp_path / "zen", n=3, size=128)

    def test_discovers_every_pair(self, root):
        adapter = ZenodoS1Adapter(root)
        assert len(adapter) == 3

    def test_discovery_is_deterministic(self, root):
        a = [d["sample_id"] for d in ZenodoS1Adapter(root).discover()]
        b = [d["sample_id"] for d in ZenodoS1Adapter(root).discover()]
        assert a == b == sorted(a)

    def test_sample_shape_and_bands(self, root):
        sample = next(iter(ZenodoS1Adapter(root)))
        assert sample.bands.shape == (2, 128, 128)
        assert sample.band_names == ["vv_db", "vh_db"]
        assert sample.mask.shape == (128, 128)
        assert sample.mask.dtype == np.uint8

    def test_binary_background_becomes_ignore_not_sea(self, root):
        """Decision D2. The single most damaging silent bug available here.

        Zenodo masks are binary. Their zero class contains land, ships and
        look-alikes, so calling it sea would teach the model that biogenic films
        are open water.
        """
        sample = next(iter(ZenodoS1Adapter(root)))
        counts = class_pixel_counts(sample.mask)

        assert counts["oil"] > 0, "fixture should contain oil"
        assert counts["ignore"] > 0, "non-oil pixels must be ignore"
        assert counts["sea"] == 0, (
            "binary background leaked into the sea class; this teaches the model "
            "that look-alikes are open water"
        )
        assert counts["look_alike"] == counts["ship"] == counts["land"] == 0

    def test_only_oil_and_ignore_are_present(self, root):
        sample = next(iter(ZenodoS1Adapter(root)))
        present = set(np.unique(sample.mask).tolist())
        assert present <= {CLASS_ID_BY_NAME["oil"], IGNORE_ID}

    def test_255_encoded_positives_are_normalised(self, tmp_path):
        import tifffile

        root = fixtures.make_zenodo_s1(tmp_path / "z255", n=1, size=64)
        mask_path = next((root / "train" / "masks").iterdir())
        mask = tifffile.imread(str(mask_path))
        tifffile.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

        sample = next(iter(ZenodoS1Adapter(root)))
        assert set(np.unique(sample.mask).tolist()) <= {CLASS_ID_BY_NAME["oil"], IGNORE_ID}

    def test_undeclared_label_value_raises(self, tmp_path):
        """Decision D7: never default an unknown value."""
        import tifffile

        root = fixtures.make_zenodo_s1(tmp_path / "zbad", n=1, size=64)
        mask_path = next((root / "train" / "masks").iterdir())
        mask = tifffile.imread(str(mask_path))
        mask[0, 0] = 7
        tifffile.imwrite(str(mask_path), mask)

        with pytest.raises(SchemaError, match="undeclared label values"):
            next(iter(ZenodoS1Adapter(root)))

    def test_missing_mask_raises_rather_than_emitting_unlabelled_scene(self, tmp_path):
        root = fixtures.make_zenodo_s1(tmp_path / "zmiss", n=2, size=64)
        next((root / "train" / "masks").iterdir()).unlink()
        with pytest.raises(FileNotFoundError, match="no matching mask"):
            ZenodoS1Adapter(root).discover()

    def test_missing_root_error_names_the_fetch_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="oilspill fetch"):
            ZenodoS1Adapter(tmp_path / "nope").discover()


# ---------------------------------------------------------------------------
# MKLab
# ---------------------------------------------------------------------------


class TestMKLab:
    def test_indexed_labels_map_to_all_five_classes(self, tmp_path):
        root = fixtures.make_mklab(tmp_path / "mk", n=2, size=128, rgb_labels=False)
        sample = next(iter(MKLabAdapter(root)))
        counts = class_pixel_counts(sample.mask)
        for name in ("sea", "oil", "look_alike", "ship", "land"):
            assert counts[name] > 0, f"class {name} missing from fixture round trip"
        assert counts["ignore"] == 0

    def test_rgb_labels_give_the_same_mask_as_indexed(self, tmp_path):
        idx_root = fixtures.make_mklab(tmp_path / "a", n=1, size=96, rgb_labels=False)
        rgb_root = fixtures.make_mklab(tmp_path / "b", n=1, size=96, rgb_labels=True)
        idx_mask = next(iter(MKLabAdapter(idx_root))).mask
        rgb_mask = next(iter(MKLabAdapter(rgb_root))).mask
        assert np.array_equal(idx_mask, rgb_mask), (
            "the palette in classes.yaml disagrees with itself between index_map "
            "and rgb_map"
        )

    def test_unknown_colour_raises_and_flags_provisional_palette(self, tmp_path):
        from PIL import Image

        root = fixtures.make_mklab(tmp_path / "mkbad", n=1, size=96, rgb_labels=True)
        lbl = next((root / "train" / "labels").iterdir())
        arr = np.array(Image.open(lbl).convert("RGB"))
        arr[0, 0] = (12, 34, 56)
        Image.fromarray(arr).save(lbl)

        with pytest.raises(SchemaError) as exc:
            next(iter(MKLabAdapter(root)))
        assert "12,34,56" in str(exc.value)
        assert "provisional" in str(exc.value)

    def test_radiometry_is_labelled_as_uncalibrated(self, tmp_path):
        """MKLab ships 8 bit renderings, not sigma0. Mixing them with dB silently
        would be a units bug that no metric would reveal."""
        root = fixtures.make_mklab(tmp_path / "mk2", n=1, size=64)
        sample = next(iter(MKLabAdapter(root)))
        assert sample.band_names == ["vv_render_norm"]
        assert sample.attrs["radiometry"] == "8bit_render_not_calibrated"
        assert 0.0 <= float(sample.bands.min()) and float(sample.bands.max()) <= 1.0


# ---------------------------------------------------------------------------
# PANGAEA
# ---------------------------------------------------------------------------


class TestPangaeaEMed:
    @pytest.fixture
    def root(self, tmp_path):
        return fixtures.make_pangaea_emed(tmp_path / "pg", n_oil=2, n_clean=2, size=256)

    def test_discovers_oil_and_clean_patches(self, root):
        items = PangaeaEMedAdapter(root).discover()
        assert len(items) == 4
        assert sum(1 for i in items if i["has_objects"]) == 2

    def test_objects_rasterise_to_oil_and_lookalike(self, root):
        samples = {s.sample_id: s for s in PangaeaEMedAdapter(root)}
        oil_sample = samples["oil_000"]
        counts = class_pixel_counts(oil_sample.mask)
        assert counts["oil"] > 0
        assert counts["look_alike"] > 0
        assert counts["sea"] > 0

    def test_clean_patch_is_all_sea_not_ignore(self, root):
        """background_policy for this source is `sea`, unlike Zenodo."""
        samples = {s.sample_id: s for s in PangaeaEMedAdapter(root)}
        counts = class_pixel_counts(samples["clean_000"].mask)
        assert counts["sea"] == 256 * 256
        assert counts["ignore"] == 0

    def test_land_is_not_asserted(self, root):
        """This source does not annotate land, so it must never emit the class."""
        for sample in PangaeaEMedAdapter(root):
            assert class_pixel_counts(sample.mask)["land"] == 0

    def test_unknown_object_label_raises(self, root):
        import json

        path = root / "annotations.geojson"
        doc = json.loads(path.read_text())
        doc["features"][0]["properties"]["class"] = "mystery_phenomenon"
        path.write_text(json.dumps(doc))

        with pytest.raises(ValueError, match="absent from object_class_map"):
            list(PangaeaEMedAdapter(root))

    def test_missing_annotation_file_lists_directory_contents(self, tmp_path):
        root = fixtures.make_pangaea_emed(tmp_path / "pg2", n_oil=1, n_clean=1, size=64)
        (root / "annotations.geojson").unlink()
        with pytest.raises(FileNotFoundError, match="no annotation file"):
            PangaeaEMedAdapter(root).discover()


# ---------------------------------------------------------------------------
# Cross cutting
# ---------------------------------------------------------------------------


def test_expressible_mask_is_enforced_per_source():
    spec = load_source_spec("zenodo_s1")
    assert spec.expressible_ids() == {CLASS_ID_BY_NAME["oil"]}
    assert spec.background_id == IGNORE_ID

    spec = load_source_spec("pangaea_emed")
    assert spec.background_id == CLASS_ID_BY_NAME["sea"]
    assert CLASS_ID_BY_NAME["land"] not in spec.expressible_ids()


def test_every_adapter_emits_validating_samples(tmp_path):
    roots = {
        ZenodoS1Adapter: fixtures.make_zenodo_s1(tmp_path / "z", n=1, size=64),
        MKLabAdapter: fixtures.make_mklab(tmp_path / "m", n=1, size=64),
        PangaeaEMedAdapter: fixtures.make_pangaea_emed(tmp_path / "p", n_oil=1, n_clean=1, size=64),
    }
    for cls, root in roots.items():
        for sample in cls(root):
            sample.validate()  # raises on any inconsistency
            assert sample.source == cls.name
