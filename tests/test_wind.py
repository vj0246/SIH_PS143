"""Verification of the CMOD5.N implementation and the observability gate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from oilspill.ard.wind import (
    INVERSION_SPEED_MAX,
    IW_INCIDENCE_MAX,
    IW_INCIDENCE_MIN,
    ObservabilityGate,
    background_wind_speed,
    cmod5n_forward,
    db_to_linear,
    direction_sensitivity,
    linear_to_db,
    wind_speed_from_sigma0,
)
from tests.reference_cmod5n import cmod5n_forward_scalar

# Sentinel-1 IW incidence angles span roughly 29 to 46 degrees. The wider grid
# checks that the vectorised branch handling holds outside the operational band
# too, which is where the np.where guards could plausibly go wrong.
INCIDENCES = [17.0, 22.0, 29.1, 33.0, 38.0, 41.0, 46.0, 50.0]
SPEEDS = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 50.0]
PHIS = [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 250.0, 359.0]


def test_forward_matches_scalar_reference_elementwise():
    """Vectorised CMOD5.N must equal the scalar reference at every grid point."""
    for inc, spd, phi in itertools.product(INCIDENCES, SPEEDS, PHIS):
        expected = cmod5n_forward_scalar(inc, spd, phi)
        got = float(cmod5n_forward(spd, phi, inc))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15), (
            f"mismatch at inc={inc} wspd={spd} phi={phi}: {got} vs {expected}"
        )


def test_forward_broadcasts_over_arrays():
    """A full grid evaluated at once must equal the same grid evaluated pointwise."""
    inc = np.array(INCIDENCES)[:, None, None]
    spd = np.array(SPEEDS)[None, :, None]
    phi = np.array(PHIS)[None, None, :]

    grid = cmod5n_forward(spd, phi, inc)
    assert grid.shape == (len(INCIDENCES), len(SPEEDS), len(PHIS))

    for i, j, k in itertools.product(
        range(len(INCIDENCES)), range(len(SPEEDS)), range(len(PHIS))
    ):
        expected = cmod5n_forward_scalar(INCIDENCES[i], SPEEDS[j], PHIS[k])
        assert float(grid[i, j, k]) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_forward_emits_no_warnings():
    """The np.where guards must not evaluate invalid operands."""
    with np.errstate(all="raise"):
        cmod5n_forward(
            np.array(SPEEDS)[:, None], np.array(PHIS)[None, :], 35.0
        )


def test_sigma0_is_monotonic_in_validated_domain():
    """Bisection is only valid where the forward model is strictly increasing.

    The validated domain is the Sentinel-1 IW incidence range and wind speeds up
    to :data:`INVERSION_SPEED_MAX`.
    """
    speeds = np.linspace(0.2, INVERSION_SPEED_MAX, 2000)
    incidences = np.linspace(IW_INCIDENCE_MIN, IW_INCIDENCE_MAX, 25)
    for inc in incidences:
        for phi in np.arange(0.0, 360.0, 5.0):
            sig = cmod5n_forward(speeds, phi, inc)
            assert np.all(np.diff(sig) > 0), f"not monotonic at inc={inc} phi={phi}"


def test_monotonicity_boundary_is_where_we_think():
    """Pin the turnover point so a coefficient edit cannot move it unnoticed.

    CMOD5.N turns over first at the low edge of the IW incidence range, upwind.
    If this test fails, INVERSION_SPEED_MAX in ard/wind.py must be revisited.
    """
    speeds = np.linspace(0.2, 50.0, 4000)
    sig = cmod5n_forward(speeds, 0.0, IW_INCIDENCE_MIN)
    turnover = speeds[np.argmax(np.diff(sig) <= 0)]
    assert 31.0 < turnover < 33.0, f"turnover moved to {turnover:.2f} m/s"
    assert INVERSION_SPEED_MAX < turnover


def test_inversion_rejects_geometry_outside_validated_domain():
    """Low incidence must fail loudly rather than return a wrong root."""
    with pytest.raises(ValueError, match="outside the validated"):
        wind_speed_from_sigma0(np.array([0.05]), inc=15.0)
    with pytest.raises(ValueError, match="exceeds the validated"):
        wind_speed_from_sigma0(np.array([0.05]), inc=35.0, hi=45.0)


def test_inversion_round_trips():
    """forward then inverse must recover the original wind speed."""
    truth = np.array([1.0, 3.0, 4.5, 6.0, 8.0, 10.0, 14.0, 25.0])
    for inc in [29.1, 35.0, 41.0, 46.0]:
        for phi in [0.0, 45.0, 90.0, 180.0]:
            sig = cmod5n_forward(truth, phi, inc)
            recovered = wind_speed_from_sigma0(sig, inc, phi)
            assert np.allclose(recovered, truth, atol=1e-6), (
                f"round trip failed at inc={inc} phi={phi}: {recovered} vs {truth}"
            )


def test_inversion_clamps_out_of_range_targets():
    """Unachievable sigma0 clamps to the search bounds rather than diverging."""
    inc = 35.0
    tiny = wind_speed_from_sigma0(np.array([1e-12]), inc, 45.0)
    huge = wind_speed_from_sigma0(np.array([1e6]), inc, 45.0)
    assert tiny[0] == pytest.approx(0.2, abs=1e-6)
    assert huge[0] == pytest.approx(INVERSION_SPEED_MAX, abs=1e-6)


def test_db_linear_round_trip():
    values = np.array([-30.0, -22.5, -15.0, -5.0, 0.0, 3.0])
    assert np.allclose(linear_to_db(db_to_linear(values)), values, atol=1e-10)


def test_direction_sensitivity_brackets_the_default():
    """Not knowing wind direction must be reported as a real spread."""
    inc = 35.0
    sig = float(cmod5n_forward(7.0, 45.0, inc))
    lo, hi = direction_sensitivity(sig, inc)
    assert lo < 7.0 < hi
    # The upwind/downwind ambiguity is worth several m/s. If this ever collapses
    # to near zero the retrieval has silently stopped depending on phi.
    assert hi - lo > 1.0


def test_background_wind_is_robust_to_slick_coverage():
    """A dark film covering part of the window must not drag the estimate down."""
    inc = 35.0
    phi = 45.0
    true_wind = 7.0
    clean = np.full(10000, float(cmod5n_forward(true_wind, phi, inc)))

    # Damp 8 percent of the window by 8 dB, a realistic slick contrast.
    slicked = clean.copy()
    slicked[:800] *= 10 ** (-8.0 / 10.0)

    naive = wind_speed_from_sigma0(np.array([slicked.mean()]), inc, phi)[0]
    robust = background_wind_speed(slicked, inc, phi, percentile=90.0)

    assert robust == pytest.approx(true_wind, abs=0.05)
    assert naive < robust, "the mean should be biased low by the slick"


def test_percentile_background_on_raw_speckle_is_biased():
    """Pin the trap documented in background_wind_speed's warning.

    Feeding single-look intensity to a percentile estimator inflates the wind by
    roughly half. This test exists so that anyone tempted to drop the
    multilooking step sees the consequence spelled out as a number.
    """
    inc, phi, true_wind = 35.0, 45.0, 7.0
    rng = np.random.default_rng(0)
    sigma0 = float(cmod5n_forward(true_wind, phi, inc))
    single_look = rng.exponential(sigma0, size=200_000)

    biased = background_wind_speed(single_look, inc, phi, percentile=85.0)
    assert biased > true_wind * 1.3, (
        f"expected a large positive bias, got {biased:.2f} m/s against {true_wind}"
    )

    # Heavy multilooking shrinks the bias but does not remove it: a percentile
    # of any distribution with spread still sits above its mean.
    looks = 64
    multilooked = single_look[: (single_look.size // looks) * looks].reshape(-1, looks).mean(axis=1)
    partly = background_wind_speed(multilooked, inc, phi, percentile=85.0)
    assert true_wind < partly < biased


def test_background_wind_on_block_means_recovers_the_truth():
    """The intended usage: block means, some of them contaminated by a slick.

    This is what oilspill.ard.features.background_sigma0 actually feeds it. Block
    means of a homogeneous sea are nearly identical, so the upper percentile is
    driven by which blocks are slick free rather than by speckle spread.
    """
    inc, phi, true_wind = 35.0, 45.0, 7.0
    rng = np.random.default_rng(1)
    sigma0 = float(cmod5n_forward(true_wind, phi, inc))

    looks = 4096
    block_means = rng.gamma(shape=looks, scale=sigma0 / looks, size=200)
    block_means[:40] *= 10 ** (-10.0 / 10.0)  # 20 percent of blocks slicked

    recovered = background_wind_speed(block_means, inc, phi, percentile=85.0)
    assert recovered == pytest.approx(true_wind, abs=0.2)


class TestObservabilityGate:
    def test_band_edges_are_inclusive(self):
        gate = ObservabilityGate(3.0, 10.0)
        assert gate.mask(np.array([3.0, 10.0])).all()
        assert not gate.mask(np.array([2.999, 10.001])).any()

    def test_nan_is_never_observable(self):
        gate = ObservabilityGate()
        assert not gate.mask(np.array([np.nan])).any()
        assert gate.fraction_observable(np.array([np.nan, np.nan])) == 0.0

    def test_verdicts(self):
        gate = ObservabilityGate(3.0, 10.0)
        assert gate.verdict(np.full(100, 6.0)) == "observable"
        assert gate.verdict(np.full(100, 1.0)) == "not_observable"
        mixed = np.concatenate([np.full(30, 6.0), np.full(70, 1.0)])
        assert gate.verdict(mixed) == "marginal"

    def test_calm_sea_is_not_observable_not_negative(self):
        """The distinction decision D4 exists to preserve."""
        inc = 35.0
        calm = cmod5n_forward(np.full(100, 1.5), 45.0, inc)
        wind = wind_speed_from_sigma0(calm, inc, 45.0)
        assert ObservabilityGate().verdict(wind) == "not_observable"
