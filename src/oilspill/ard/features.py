"""Physics channels stacked alongside backscatter.

Decision D5. Slick contrast is not an absolute quantity: the same film produces
different damping at 30 and 45 degrees of incidence and at 4 and 9 m/s of wind.
A network fed only intensity has to infer that relationship from a few thousand
patches, which is a large part of why cross-basin transfer collapses in the
published literature. Handing it the conditioning variables directly is cheap.

Every function here works on a single scene and returns a ``(H, W)`` float32
array so the channels stack without special cases.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .wind import (
    IW_INCIDENCE_MAX,
    IW_INCIDENCE_MIN,
    ObservabilityGate,
    background_wind_speed,
    db_to_linear,
    linear_to_db,
    wind_speed_from_sigma0,
)

__all__ = [
    "synthesize_incidence",
    "background_sigma0",
    "damping_ratio_db",
    "local_variance_db",
    "wind_channel",
    "distance_to_land",
    "build_feature_stack",
    "FEATURE_BUILDERS",
]


def synthesize_incidence(
    shape: tuple[int, int],
    near: float = IW_INCIDENCE_MIN,
    far: float = IW_INCIDENCE_MAX,
    range_axis: int = 1,
) -> np.ndarray:
    """Linear incidence ramp across the ground range axis.

    A fallback for sources that ship no incidence raster, which is the case for
    every public benchmark dataset found. It is an approximation: the true ramp
    across an IW sub-swath is not exactly linear and sub-swath boundaries step.

    Args:
        shape: ``(H, W)`` of the scene.
        near: Near range incidence in degrees.
        far: Far range incidence in degrees.
        range_axis: 1 for columns (the usual GRD convention), 0 for rows.

    Returns:
        ``(H, W)`` float32 incidence angles in degrees.
    """
    height, width = shape
    n = width if range_axis == 1 else height
    ramp = np.linspace(near, far, n, dtype=np.float32)
    if range_axis == 1:
        return np.broadcast_to(ramp[None, :], shape).astype(np.float32)
    return np.broadcast_to(ramp[:, None], shape).astype(np.float32)


def multilook(sigma0_linear: np.ndarray, size: int = 5) -> np.ndarray:
    """Boxcar multilooking in the linear domain.

    Averaging must happen in linear power, never in dB. SAR intensity is
    exponentially distributed for a single look, so the mean of the logarithm is
    biased low by 2.5 dB relative to the logarithm of the mean, and that bias
    propagates straight into any ratio computed from it.
    """
    arr = np.asarray(sigma0_linear, dtype=np.float64)
    if size <= 1:
        return arr
    return ndimage.uniform_filter(arr, size=size, mode="nearest")


def background_sigma0(
    sigma0_linear: np.ndarray,
    block: int = 64,
    percentile: float = 85.0,
    neighbourhood: int = 5,
) -> np.ndarray:
    """Undamped background backscatter field.

    Two stages, and both are necessary:

    1. **Block means.** The block mean of linear intensity is an unbiased
       estimator of sigma0 and collapses speckle by the square root of the block
       area. Taking a percentile of raw single-look pixels instead would be
       biased by the speckle distribution itself: the 85th percentile of a
       single-look exponential sits at 1.90 times the mean, a 2.8 dB error that
       is invisible in the output but shifts every retrieved wind speed by
       roughly 50 percent. This was found by the wind recovery test, not
       reasoned about in advance.

    2. **Upper percentile across neighbouring blocks.** A block sitting inside a
       slick has a genuinely low mean, so it must not become its own background.
       Taking an upper percentile over a neighbourhood of blocks rejects those,
       while still tracking real large-scale wind variation across the scene,
       which a single scene-wide scalar would flatten.

    Args:
        sigma0_linear: ``(H, W)`` sigma0 in linear units.
        block: Block edge in pixels. Must be smaller than the scale over which
            wind varies, and the neighbourhood must span more than the widest
            slick, or the slick becomes its own background and the damping ratio
            collapses toward zero.
        percentile: Percentile taken across the block neighbourhood.
        neighbourhood: Neighbourhood edge, in blocks.

    Returns:
        ``(H, W)`` float32 background field in linear units.
    """
    arr = np.asarray(sigma0_linear, dtype=np.float64)
    height, width = arr.shape
    ny = max(1, int(np.ceil(height / block)))
    nx = max(1, int(np.ceil(width / block)))

    pad_y = ny * block - height
    pad_x = nx * block - width
    padded = np.pad(arr, ((0, pad_y), (0, pad_x)), mode="edge")

    tiles = padded.reshape(ny, block, nx, block).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    coarse = tiles.mean(axis=2)

    if coarse.size > 1:
        coarse = ndimage.percentile_filter(
            coarse, percentile=percentile, size=min(neighbourhood, max(coarse.shape)), mode="nearest"
        )

    if coarse.shape == (1, 1):
        return np.full((height, width), float(coarse[0, 0]), dtype=np.float32)

    zoom = (height / coarse.shape[0], width / coarse.shape[1])
    upsampled = ndimage.zoom(coarse, zoom, order=1, mode="nearest")
    return upsampled[:height, :width].astype(np.float32)


def damping_ratio_db(
    vv_db: np.ndarray,
    block: int = 64,
    percentile: float = 85.0,
    looks: int = 5,
    neighbourhood: int = 5,
) -> np.ndarray:
    """Backscatter suppression relative to the local background, in dB.

    This is the one genuinely physical handle SAR gives on a surface film: a
    monomolecular or oil film damps Bragg scale capillary waves through Marangoni
    damping, and the amount of damping depends on the film's elasticity and
    viscosity. It is not an oil type classifier on its own, and this codebase
    does not claim it is.

    Both terms of the ratio are multilooked in the linear domain before the
    logarithm is taken, so the estimate is not contaminated by the log-of-speckle
    bias described in :func:`multilook`.

    Args:
        vv_db: VV sigma0 in dB.
        block: Background block edge in pixels.
        percentile: Percentile across the block neighbourhood.
        looks: Boxcar edge for multilooking the numerator, in pixels.
        neighbourhood: Background neighbourhood edge, in blocks.

    Returns:
        ``(H, W)`` float32, positive where the pixel is darker than its
        surroundings. Clean water sits near 0; a visible slick runs roughly
        5 to 15 dB.
    """
    lin = db_to_linear(np.asarray(vv_db, dtype=np.float64))
    local = multilook(lin, size=looks)
    bg = background_sigma0(
        lin, block=block, percentile=percentile, neighbourhood=neighbourhood
    ).astype(np.float64)
    return (linear_to_db(bg) - linear_to_db(local)).astype(np.float32)


def local_variance_db(vv_db: np.ndarray, size: int = 7) -> np.ndarray:
    """Local variance of the dB image.

    Separates two things that look equally dark in a single pixel: a genuine
    film, which suppresses the Bragg resonance and so flattens the speckle
    statistics, and a low wind cell, which stays fully developed speckle at a
    lower mean. Cheap, and it captures most of what a full GLCM stack would.
    """
    arr = np.asarray(vv_db, dtype=np.float64)
    mean = ndimage.uniform_filter(arr, size=size, mode="nearest")
    mean_sq = ndimage.uniform_filter(arr * arr, size=size, mode="nearest")
    return np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32)


def wind_channel(
    vv_db: np.ndarray,
    incidence: np.ndarray,
    phi: float = 45.0,
    block: int = 64,
    percentile: float = 85.0,
) -> np.ndarray:
    """CMOD5.N wind speed field derived from the background backscatter.

    Inverting the raw pixel would report a spuriously calm wind inside every
    slick, which is exactly backwards: it would tell the model that oil implies
    unobservable conditions. The inversion therefore runs on the background
    field from :func:`background_sigma0`.

    Returns:
        ``(H, W)`` float32 neutral equivalent 10 m wind speed, m/s.
    """
    lin = db_to_linear(np.asarray(vv_db, dtype=np.float64))
    bg = background_sigma0(lin, block=block, percentile=percentile)
    inc = np.asarray(incidence, dtype=np.float64)
    inc = np.clip(inc, IW_INCIDENCE_MIN, IW_INCIDENCE_MAX)
    inc = np.broadcast_to(inc, bg.shape)
    return wind_speed_from_sigma0(bg.astype(np.float64), inc, phi).astype(np.float32)


def distance_to_land(land_mask: np.ndarray, pixel_metres: float = 10.0) -> np.ndarray:
    """Euclidean distance from each pixel to the nearest land pixel, in km.

    Coastal proximity is a strong prior in both directions. Ports, outfalls and
    shipping lanes concentrate real discharges near the coast, while river
    plumes, algal blooms and shallow-water bathymetric signatures concentrate
    look-alikes there too.

    Args:
        land_mask: Boolean array, True on land.
        pixel_metres: Ground sample distance. Sentinel-1 IW GRDH is 10 m.

    Returns:
        ``(H, W)`` float32 distance in kilometres. All-water scenes return a
        constant large value rather than infinity, so the channel stays finite.
    """
    land = np.asarray(land_mask, dtype=bool)
    if not land.any():
        return np.full(land.shape, 1000.0, dtype=np.float32)
    dist_px = ndimage.distance_transform_edt(~land)
    return (dist_px * pixel_metres / 1000.0).astype(np.float32)


#: Channel name to builder. Keys are the names written into the manifest and
#: selected in training configs, so they are part of the public contract.
FEATURE_BUILDERS = {
    "incidence_deg": "synthesised or supplied incidence angle",
    "wind_ms": "CMOD5.N wind speed from background backscatter",
    "damping_db": "backscatter suppression versus local background",
    "local_var_db": "local variance of the dB image",
    "dist_land_km": "distance to nearest land pixel",
}


def build_feature_stack(
    vv_db: np.ndarray,
    vh_db: np.ndarray | None = None,
    incidence: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    channels: list[str] | None = None,
    phi: float = 45.0,
    block: int = 64,
    pixel_metres: float = 10.0,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the selected channels into a ``(C, H, W)`` float32 stack.

    Args:
        vv_db: VV sigma0 in dB. Required.
        vh_db: VH sigma0 in dB, optional.
        incidence: Per pixel incidence in degrees. Synthesised if absent.
        land_mask: Boolean land mask. ``dist_land_km`` is skipped if absent.
        channels: Subset of :data:`FEATURE_BUILDERS` keys plus ``vv_db`` and
            ``vh_db``. Defaults to everything available. Keeping this selectable
            is what preserves the ablation promised in decision D5.
        phi: Wind to look angle in degrees for the CMOD inversion.
        block: Block edge for the background estimate.
        pixel_metres: Ground sample distance for the land distance channel.

    Returns:
        ``(stack, names)``.

    Raises:
        ValueError: If a requested channel is unknown or its inputs are missing.
    """
    vv = np.asarray(vv_db, dtype=np.float32)
    shape = vv.shape

    if incidence is None:
        incidence = synthesize_incidence(shape)
    incidence = np.broadcast_to(np.asarray(incidence, dtype=np.float32), shape)

    available = ["vv_db"]
    if vh_db is not None:
        available.append("vh_db")
    available += ["incidence_deg", "wind_ms", "damping_db", "local_var_db"]
    if land_mask is not None:
        available.append("dist_land_km")

    requested = list(channels) if channels is not None else available
    unknown = [c for c in requested if c not in (*available, "vh_db", "dist_land_km")]
    if unknown:
        raise ValueError(f"unknown feature channels {unknown}; available are {available}")
    missing = [c for c in requested if c not in available]
    if missing:
        raise ValueError(
            f"channels {missing} were requested but their inputs were not supplied "
            "(vh_db needs vh_db=..., dist_land_km needs land_mask=...)"
        )

    computed: dict[str, np.ndarray] = {}
    for name in requested:
        if name == "vv_db":
            computed[name] = vv
        elif name == "vh_db":
            computed[name] = np.asarray(vh_db, dtype=np.float32)
        elif name == "incidence_deg":
            computed[name] = incidence.astype(np.float32)
        elif name == "wind_ms":
            computed[name] = wind_channel(vv, incidence, phi=phi, block=block)
        elif name == "damping_db":
            computed[name] = damping_ratio_db(vv, block=block)
        elif name == "local_var_db":
            computed[name] = local_variance_db(vv)
        elif name == "dist_land_km":
            computed[name] = distance_to_land(land_mask, pixel_metres=pixel_metres)

    stack = np.stack([computed[n] for n in requested], axis=0).astype(np.float32)
    return stack, requested


def scene_wind_summary(
    vv_db: np.ndarray,
    incidence: np.ndarray | None = None,
    phi: float = 45.0,
    gate: ObservabilityGate | None = None,
) -> dict:
    """Scene level wind statistics for the manifest.

    Decision D4: this is recorded, never used to drop the scene.
    """
    gate = gate or ObservabilityGate()
    vv = np.asarray(vv_db, dtype=np.float32)
    inc = synthesize_incidence(vv.shape) if incidence is None else incidence
    field = wind_channel(vv, inc, phi=phi)
    # The scene figure is the median of the already debiased background field,
    # not a percentile of raw pixels: see background_sigma0 for why the latter
    # is off by the speckle distribution.
    scene_wind = float(np.nanmedian(field))
    return {
        "wind_ms_mean": float(np.nanmean(field)),
        "wind_ms_min": float(np.nanmin(field)),
        "wind_ms_max": float(np.nanmax(field)),
        "wind_ms_scene": float(scene_wind),
        "wind_fraction_observable": gate.fraction_observable(field),
        "observability": gate.verdict(field),
        "wind_phi_assumed_deg": float(phi),
    }
