"""CMOD5.N geophysical model function and the observability gate.

Decision D4. Wind is computed for every scene and stored as tile metadata; it is
never used to silently drop data at ingest.

The forward model is the CMOD5.N (neutral equivalent 10 m wind) coefficient set
from Hersbach. This module implements it vectorised over numpy arrays. The
implementation is unit tested against the scalar reference shipped in IFREMER's
``xsarsea`` package over a grid of incidence angle, wind speed and relative wind
direction; see ``tests/test_wind.py``.

Two caveats that matter operationally and are deliberately surfaced in the API
rather than buried:

1. CMOD inverts VV backscatter to wind speed given a wind *direction*. A single
   SAR image does not supply direction. ``wind_speed_from_sigma0`` therefore
   takes ``phi`` explicitly. Supply it from a reanalysis field (ERA5 10 m u/v)
   where available. The ``phi=45`` default is a documented compromise, not a
   physical claim, and the direction sensitivity is reported by
   :func:`direction_sensitivity`.

2. Inside a slick the backscatter is damped, so inverting it yields a spuriously
   low wind. The gate must be driven by *background* wind, which is what
   :func:`background_wind_speed` estimates by taking an upper percentile over a
   neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CMOD5N_COEFFS",
    "cmod5n_forward",
    "wind_speed_from_sigma0",
    "background_wind_speed",
    "direction_sensitivity",
    "ObservabilityGate",
    "db_to_linear",
    "linear_to_db",
]

# CMOD5.N coefficient vector, one based indexing preserved from the published
# formulation (index 0 is unused padding).
CMOD5N_COEFFS = np.array(
    [
        0.0,
        -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
        6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 0.0066, 0.3222,
        0.0120, 22.7000, 2.0813, 3.0000, 8.3659, -3.3428, 1.3236, 6.2437,
        2.3893, 0.3249, 4.1590, 1.6930,
    ],
    dtype=np.float64,
)

_ZPOW = 1.6
_THETM = 40.0
_THETHR = 25.0

WIND_SPEED_MIN = 0.2
WIND_SPEED_MAX = 50.0

# Validated inversion domain.
#
# CMOD5.N is NOT monotonic in wind speed everywhere, so bisection is not
# unconditionally valid. Sweeping the coefficient set over incidence angle and
# relative wind direction shows sigma0 turning over at high wind:
#
#   incidence 29.1 deg  first turnover at 31.8 m/s (phi = 0, upwind)
#   incidence 35.0 deg  first turnover at 36.3 m/s
#   incidence 40.0 deg  first turnover at 45.4 m/s
#   incidence 46.0 deg  monotonic to 50 m/s
#   incidence 17.0 deg  first turnover at 24.4 m/s
#   incidence 15.0 deg  first turnover at 13.0 m/s
#
# Inside the Sentinel-1 IW incidence range the model is strictly increasing up
# to 31.8 m/s, so the inversion ceiling is set below that. 30 m/s is Beaufort 11
# and is three times the upper edge of the oil detection band, so nothing
# operationally relevant is lost. Test ``test_sigma0_is_monotonic_in_validated_domain``
# enforces this and ``test_monotonicity_boundary_is_where_we_think`` fails if a
# coefficient edit moves the turnover.
INVERSION_SPEED_MAX = 30.0

# Sentinel-1 Interferometric Wide swath incidence angle range.
IW_INCIDENCE_MIN = 29.1
IW_INCIDENCE_MAX = 46.0


def db_to_linear(x_db: np.ndarray | float) -> np.ndarray:
    """Convert decibels to linear power."""
    return np.power(10.0, np.asarray(x_db, dtype=np.float64) / 10.0)


def linear_to_db(x_lin: np.ndarray | float, floor: float = 1e-12) -> np.ndarray:
    """Convert linear power to decibels, clamping non positive values."""
    arr = np.asarray(x_lin, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(arr, floor))


def cmod5n_forward(
    wspd: np.ndarray | float,
    phi: np.ndarray | float,
    inc: np.ndarray | float,
    coeffs: np.ndarray = CMOD5N_COEFFS,
) -> np.ndarray:
    """CMOD5.N forward model: wind to VV sigma0.

    Args:
        wspd: Neutral equivalent 10 m wind speed, m/s.
        phi: Angle between wind direction and radar look direction, degrees.
        inc: Radar incidence angle, degrees.
        coeffs: Coefficient vector, defaults to CMOD5.N.

    Returns:
        VV sigma0 in linear units, broadcast over the inputs.

    Notes:
        Fully vectorised. The two conditional branches in the published scalar
        formulation (``s < s0`` and ``v2 < y0``) are expressed with
        :func:`numpy.where` and are evaluated on safe operands so that the
        unused branch cannot emit warnings on out of domain values.
    """
    c = np.asarray(coeffs, dtype=np.float64)
    v = np.asarray(wspd, dtype=np.float64)
    p = np.asarray(phi, dtype=np.float64)
    t = np.asarray(inc, dtype=np.float64)

    y0 = c[19]
    pn = c[20]
    a = y0 - (y0 - 1.0) / pn
    b = 1.0 / (pn * (y0 - 1.0) ** (pn - 1.0))

    cosphi = np.cos(np.deg2rad(p))
    x = (t - _THETM) / _THETHR
    x2 = x * x

    # B0
    a0 = c[1] + c[2] * x + c[3] * x2 + c[4] * x * x2
    a1 = c[5] + c[6] * x
    a2 = c[7] + c[8] * x
    gam = c[9] + c[10] * x + c[11] * x2
    s0 = c[12] + c[13] * x
    s = a2 * v

    a3_base = 1.0 / (1.0 + np.exp(-s0))
    # Guard the power so the discarded branch never sees a division by zero or a
    # negative base; np.where evaluates both operands.
    ratio = np.where(s0 != 0.0, s / np.where(s0 == 0.0, 1.0, s0), 1.0)
    ratio = np.maximum(ratio, 1e-12)
    a3_low = a3_base * ratio ** (s0 * (1.0 - a3_base))
    a3_high = 1.0 / (1.0 + np.exp(-s))
    a3 = np.where(s < s0, a3_low, a3_high)

    b0 = (a3**gam) * np.power(10.0, a0 + a1 * v)

    # B1
    b1 = c[15] * v * (0.5 + x - np.tanh(4.0 * (x + c[16] + c[17] * v)))
    b1 = (c[14] * (1.0 + x) - b1) / (np.exp(0.34 * (v - c[18])) + 1.0)

    # B2
    v0 = c[21] + c[22] * x + c[23] * x2
    d1 = c[24] + c[25] * x + c[26] * x2
    d2 = c[27] + c[28] * x
    v2 = v / v0 + 1.0
    v2_low = a + b * np.maximum(v2 - 1.0, 0.0) ** pn
    v2 = np.where(v2 < y0, v2_low, v2)
    b2 = (-d1 + d2 * v2) * np.exp(-v2)

    return b0 * (1.0 + b1 * cosphi + b2 * (2.0 * cosphi**2 - 1.0)) ** _ZPOW


def wind_speed_from_sigma0(
    sigma0_vv: np.ndarray,
    inc: np.ndarray | float,
    phi: np.ndarray | float = 45.0,
    iterations: int = 40,
    lo: float = WIND_SPEED_MIN,
    hi: float = INVERSION_SPEED_MAX,
    strict_domain: bool = True,
) -> np.ndarray:
    """Invert CMOD5.N for wind speed by bisection.

    Args:
        sigma0_vv: VV sigma0 in **linear** units. Use :func:`db_to_linear` first
            if the raster is in dB.
        inc: Incidence angle in degrees, scalar or per pixel.
        phi: Angle between wind direction and look direction, degrees. See the
            module docstring: this is not recoverable from a single image and
            the default is a compromise.
        iterations: Bisection steps. 40 gives roughly 1e-11 m/s bracket width
            over the default range, so the result is deterministic.
        lo: Lower search bound, m/s.
        hi: Upper search bound, m/s. Defaults to
            :data:`INVERSION_SPEED_MAX`, above which CMOD5.N stops being
            monotonic at Sentinel-1 IW incidence angles and bisection would
            silently return the wrong root.
        strict_domain: Raise if ``inc`` falls outside the Sentinel-1 IW range or
            ``hi`` exceeds the validated monotonic ceiling. Set False only for a
            deliberate experiment with another sensor geometry, and re-verify
            monotonicity for that geometry first.

    Returns:
        Neutral equivalent 10 m wind speed in m/s. Pixels whose sigma0 lies
        outside the achievable range are clamped to ``lo`` or ``hi``; they are
        exactly the pixels the observability gate then rejects.

    Raises:
        ValueError: If ``strict_domain`` and the geometry is outside the
            validated domain.
    """
    target = np.asarray(sigma0_vv, dtype=np.float64)
    inc_a = np.broadcast_to(np.asarray(inc, dtype=np.float64), target.shape)
    phi_a = np.broadcast_to(np.asarray(phi, dtype=np.float64), target.shape)

    if strict_domain:
        if hi > INVERSION_SPEED_MAX:
            raise ValueError(
                f"inversion ceiling {hi} m/s exceeds the validated monotonic "
                f"domain of {INVERSION_SPEED_MAX} m/s for CMOD5.N at IW "
                "incidence angles; bisection is not valid above it"
            )
        finite_inc = inc_a[np.isfinite(inc_a)]
        if finite_inc.size:
            lo_i, hi_i = float(finite_inc.min()), float(finite_inc.max())
            if lo_i < IW_INCIDENCE_MIN or hi_i > IW_INCIDENCE_MAX:
                raise ValueError(
                    f"incidence range [{lo_i:.2f}, {hi_i:.2f}] deg falls outside the "
                    f"validated Sentinel-1 IW range [{IW_INCIDENCE_MIN}, "
                    f"{IW_INCIDENCE_MAX}] deg. CMOD5.N loses monotonicity at low "
                    "incidence (turnover at 13 m/s by 15 deg), so pass "
                    "strict_domain=False only after re-verifying this geometry."
                )

    low = np.full(target.shape, lo, dtype=np.float64)
    high = np.full(target.shape, hi, dtype=np.float64)

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        pred = cmod5n_forward(mid, phi_a, inc_a)
        too_small = pred < target
        low = np.where(too_small, mid, low)
        high = np.where(too_small, high, mid)

    return 0.5 * (low + high)


def background_wind_speed(
    sigma0_vv_linear: np.ndarray,
    inc: np.ndarray | float,
    phi: np.ndarray | float = 45.0,
    percentile: float = 90.0,
) -> float:
    """Scene level background wind, robust to the presence of dark films.

    A slick suppresses backscatter, so inverting the scene mean underestimates
    the true wind and would wrongly declare the scene unobservable. Taking an
    upper percentile of sigma0 over the water estimates the undamped background.

    .. warning::
        The input must already be **multilooked**. A percentile of single-look
        intensity is biased by the speckle distribution itself: the 85th
        percentile of a single-look exponential sits at 1.90 times its mean, a
        2.8 dB error that silently inflates the retrieved wind by roughly half.
        Use :func:`oilspill.ard.features.multilook`, or prefer
        :func:`oilspill.ard.features.wind_channel`, which handles this.

    Args:
        sigma0_vv_linear: Multilooked VV sigma0 in linear units over water only.
            Mask land and no data before calling.
        inc: Incidence angle in degrees.
        phi: Wind to look angle in degrees.
        percentile: Percentile of sigma0 taken as the undamped background.
            90 tolerates up to roughly 10 percent slick coverage in the window.

    Returns:
        Background wind speed in m/s.
    """
    arr = np.asarray(sigma0_vv_linear, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return float("nan")
    ref = float(np.percentile(finite, percentile))
    inc_ref = float(np.nanmean(np.asarray(inc, dtype=np.float64)))
    return float(wind_speed_from_sigma0(np.array([ref]), inc_ref, phi)[0])


def direction_sensitivity(
    sigma0_vv_linear: float, inc: float, phis: np.ndarray | None = None
) -> tuple[float, float]:
    """Spread of retrieved wind speed across all possible wind directions.

    Reports the cost of not knowing the wind direction, so that a retrieval can
    be presented with an honest uncertainty band rather than a single number.

    Returns:
        ``(min_speed, max_speed)`` in m/s over the sampled directions.
    """
    if phis is None:
        phis = np.arange(0.0, 360.0, 10.0)
    speeds = wind_speed_from_sigma0(
        np.full(phis.shape, float(sigma0_vv_linear)), inc, phis
    )
    return float(np.nanmin(speeds)), float(np.nanmax(speeds))


@dataclass(frozen=True)
class ObservabilityGate:
    """Wind band inside which SAR oil detection is physically meaningful.

    Below the lower bound the sea surface is too smooth and the whole scene is
    dark, so a slick has nothing to contrast against. Above the upper bound
    wave breaking overwhelms the Marangoni damping and the slick signature
    disappears. Outside the band the correct output is ``not_observable``, which
    is a different statement from "no oil detected".
    """

    min_ms: float = 3.0
    max_ms: float = 10.0

    def mask(self, wind_ms: np.ndarray) -> np.ndarray:
        """Boolean array, True where detection is considered valid."""
        w = np.asarray(wind_ms, dtype=np.float64)
        return np.isfinite(w) & (w >= self.min_ms) & (w <= self.max_ms)

    def fraction_observable(self, wind_ms: np.ndarray) -> float:
        """Fraction of finite pixels lying inside the band."""
        w = np.asarray(wind_ms, dtype=np.float64)
        finite = np.isfinite(w)
        if not finite.any():
            return 0.0
        return float(self.mask(w).sum() / finite.sum())

    def verdict(self, wind_ms: np.ndarray, min_fraction: float = 0.5) -> str:
        """Scene level verdict: ``observable``, ``marginal`` or ``not_observable``."""
        frac = self.fraction_observable(wind_ms)
        if frac >= min_fraction:
            return "observable"
        if frac > 0.0:
            return "marginal"
        return "not_observable"
