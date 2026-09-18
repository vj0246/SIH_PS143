"""The physics channels must recover the quantities they claim to measure."""

from __future__ import annotations

import numpy as np
import pytest

from oilspill.ard.features import (
    build_feature_stack,
    damping_ratio_db,
    distance_to_land,
    local_variance_db,
    scene_wind_summary,
    synthesize_incidence,
    wind_channel,
)
from oilspill.ard.wind import IW_INCIDENCE_MAX, IW_INCIDENCE_MIN
from tests.fixtures import synthetic_scene


def test_synthesised_incidence_spans_the_iw_swath():
    inc = synthesize_incidence((64, 128))
    assert inc.shape == (64, 128)
    assert inc[0, 0] == pytest.approx(IW_INCIDENCE_MIN)
    assert inc[0, -1] == pytest.approx(IW_INCIDENCE_MAX)
    # Constant down azimuth, varying across ground range.
    assert np.allclose(inc[0], inc[-1])
    assert inc[0, 0] < inc[0, -1]


def test_damping_ratio_recovers_the_injected_contrast():
    """A 9 dB slick must read back as roughly 9 dB, not as an arbitrary number."""
    vv, _, oil = synthetic_scene(512, 512, slick_damping_db=9.0, seed=7)
    damping = damping_ratio_db(vv, block=64)

    # Speckle is heavy tailed, so compare medians rather than means.
    inside = float(np.median(damping[oil]))
    outside = float(np.median(damping[~oil]))

    assert 6.0 < inside < 12.0, f"slick damping read back as {inside:.2f} dB"
    assert abs(outside) < 2.5, f"clean water reads {outside:.2f} dB of damping"
    assert inside - outside > 5.0


def test_damping_ratio_is_near_zero_on_clean_water():
    vv, _, _ = synthetic_scene(384, 384, slick_damping_db=0.0, seed=11)
    damping = damping_ratio_db(vv, block=64)
    assert abs(float(np.median(damping))) < 2.5


def test_background_block_must_exceed_the_slick_scale():
    """Documents the failure mode called out in the background_sigma0 docstring.

    With a block far smaller than the slick, the slick becomes its own
    background and the measured damping collapses. This is a real trap, so it is
    pinned rather than left as prose.
    """
    vv, _, oil = synthetic_scene(512, 512, slick_damping_db=9.0, seed=7)
    good = float(np.median(damping_ratio_db(vv, block=64)[oil]))
    too_small = float(np.median(damping_ratio_db(vv, block=8)[oil]))
    assert too_small < good / 2.0


def test_wind_channel_recovers_the_injected_wind_speed():
    """The whole observability gate rests on this being right."""
    for truth in (4.0, 7.0, 9.5):
        vv, _, _ = synthetic_scene(384, 384, wind_ms=truth, slick_damping_db=0.0, seed=3)
        inc = np.full(vv.shape, 35.0, dtype=np.float32)
        field = wind_channel(vv, inc)
        assert float(np.median(field)) == pytest.approx(truth, abs=0.9), (
            f"injected {truth} m/s, recovered {float(np.median(field)):.2f}"
        )


def test_wind_channel_is_not_dragged_down_inside_the_slick():
    """Inverting raw pixels would report calm water inside every slick, which
    would teach the model that oil implies unobservable conditions."""
    vv, _, oil = synthetic_scene(384, 384, wind_ms=7.0, slick_damping_db=10.0, seed=5)
    inc = np.full(vv.shape, 35.0, dtype=np.float32)
    field = wind_channel(vv, inc)
    inside = float(np.median(field[oil]))
    assert inside > 5.0, f"wind inside the slick collapsed to {inside:.2f} m/s"


def test_local_variance_separates_a_film_from_a_calm_patch():
    """A damped film flattens speckle statistics; a low wind cell does not."""
    rng = np.random.default_rng(0)
    size = 256
    base = rng.exponential(1.0, size=(size, size))

    # Fully developed speckle at a lower mean, which is what a calm cell is.
    calm = 10.0 * np.log10(base * 10 ** (-8 / 10.0))
    # A damped film: same mean drop but suppressed fluctuation.
    film = 10.0 * np.log10((0.15 * base + 0.85) * 10 ** (-8 / 10.0))

    assert float(np.mean(local_variance_db(film))) < float(np.mean(local_variance_db(calm)))


def test_distance_to_land_is_zero_on_land_and_finite_everywhere():
    land = np.zeros((64, 64), dtype=bool)
    land[:4, :] = True
    dist = distance_to_land(land, pixel_metres=10.0)
    assert float(dist[0, 0]) == 0.0
    assert np.all(np.isfinite(dist))
    assert float(dist[-1, 0]) == pytest.approx((64 - 4) * 10.0 / 1000.0, rel=0.02)


def test_distance_to_land_on_an_all_water_scene_stays_finite():
    dist = distance_to_land(np.zeros((32, 32), dtype=bool))
    assert np.all(np.isfinite(dist))
    assert float(dist.min()) == 1000.0


class TestFeatureStack:
    def test_default_stack_contains_every_available_channel(self):
        vv, vh, _ = synthetic_scene(128, 128)
        stack, names = build_feature_stack(vv, vh)
        assert names == ["vv_db", "vh_db", "incidence_deg", "wind_ms", "damping_db", "local_var_db"]
        assert stack.shape == (len(names), 128, 128)
        assert stack.dtype == np.float32
        assert np.all(np.isfinite(stack))

    def test_channels_can_be_selected_for_ablation(self):
        vv, vh, _ = synthetic_scene(96, 96)
        stack, names = build_feature_stack(vv, vh, channels=["vv_db", "vh_db"])
        assert names == ["vv_db", "vh_db"]
        assert stack.shape == (2, 96, 96)

    def test_requesting_a_channel_without_its_input_fails_loudly(self):
        vv, _, _ = synthetic_scene(64, 64)
        with pytest.raises(ValueError, match="were requested but their inputs"):
            build_feature_stack(vv, channels=["vv_db", "vh_db"])
        with pytest.raises(ValueError, match="were requested but their inputs"):
            build_feature_stack(vv, channels=["vv_db", "dist_land_km"])

    def test_unknown_channel_name_fails_loudly(self):
        vv, _, _ = synthetic_scene(64, 64)
        with pytest.raises(ValueError, match="unknown feature channels"):
            build_feature_stack(vv, channels=["vv_db", "magic"])

    def test_land_mask_adds_the_distance_channel(self):
        vv, vh, _ = synthetic_scene(96, 96)
        land = np.zeros((96, 96), dtype=bool)
        land[:6, :] = True
        _, names = build_feature_stack(vv, vh, land_mask=land)
        assert "dist_land_km" in names


class TestSceneWindSummary:
    def test_reports_observable_for_a_normal_scene(self):
        vv, _, _ = synthetic_scene(256, 256, wind_ms=7.0, slick_damping_db=0.0, seed=2)
        summary = scene_wind_summary(vv, np.full(vv.shape, 35.0))
        assert summary["observability"] == "observable"
        assert summary["wind_ms_scene"] == pytest.approx(7.0, abs=1.0)
        assert 0.0 <= summary["wind_fraction_observable"] <= 1.0

    def test_reports_not_observable_on_a_glassy_sea(self):
        """Decision D4: this must be distinguishable from 'no oil found'."""
        vv, _, _ = synthetic_scene(256, 256, wind_ms=1.5, slick_damping_db=0.0, seed=4)
        summary = scene_wind_summary(vv, np.full(vv.shape, 35.0))
        assert summary["observability"] == "not_observable"

    def test_records_the_assumed_wind_direction(self):
        """The phi assumption must travel with the number it produced."""
        vv, _, _ = synthetic_scene(128, 128)
        summary = scene_wind_summary(vv, np.full(vv.shape, 35.0), phi=90.0)
        assert summary["wind_phi_assumed_deg"] == 90.0
