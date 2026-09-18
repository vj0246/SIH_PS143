"""Synthetic datasets reproducing each source's on-disk layout.

Decision D9: the authoring sandbox cannot reach any of the real data hosts, so
correctness is established against fixtures that mimic the distributed layouts
byte for byte in structure, if not in content. When the real archives land,
running the same tests against them is the acceptance check.

The synthetic SAR content is not noise: it is a plausible sea surface with
speckle plus a damped elliptical slick, so that tiling, damping ratio and wind
inversion all see something with the right statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oilspill.ard.wind import cmod5n_forward, linear_to_db


def synthetic_scene(
    height: int = 256,
    width: int = 256,
    wind_ms: float = 7.0,
    incidence: float = 35.0,
    slick_damping_db: float = 9.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a VV/VH dB scene with one elliptical slick.

    Returns:
        ``(vv_db, vh_db, oil_mask)`` where ``oil_mask`` is a boolean array.
    """
    rng = np.random.default_rng(seed)

    sigma0_vv = float(cmod5n_forward(wind_ms, 45.0, incidence))
    # VH sits far below VV over the ocean; the offset is representative, not
    # a calibrated model.
    sigma0_vh = sigma0_vv * 10 ** (-9.0 / 10.0)

    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height * 0.45, width * 0.55
    ry, rx = height * 0.18, width * 0.30
    theta = np.deg2rad(25.0)
    yr = (yy - cy) * np.cos(theta) + (xx - cx) * np.sin(theta)
    xr = -(yy - cy) * np.sin(theta) + (xx - cx) * np.cos(theta)
    oil = (yr / ry) ** 2 + (xr / rx) ** 2 <= 1.0

    damp = np.where(oil, 10 ** (-slick_damping_db / 10.0), 1.0)

    # Multiplicative speckle, single look intensity is exponentially distributed.
    speckle_vv = rng.exponential(1.0, size=(height, width))
    speckle_vh = rng.exponential(1.0, size=(height, width))

    vv = linear_to_db(sigma0_vv * damp * speckle_vv).astype(np.float32)
    vh = linear_to_db(sigma0_vh * damp * speckle_vh).astype(np.float32)
    return vv, vh, oil


# ---------------------------------------------------------------------------
# Zenodo layout
# ---------------------------------------------------------------------------


def make_zenodo_s1(root: Path, n: int = 3, size: int = 256, partition: str = "train") -> Path:
    """Write ``<root>/<partition>/{images,masks}/*.tif`` in the Zenodo layout."""
    import tifffile

    img_dir = root / partition / "images"
    mask_dir = root / partition / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        vv, vh, oil = synthetic_scene(size, size, seed=i)
        stack = np.stack([vv, vh], axis=-1).astype(np.float32)
        tifffile.imwrite(str(img_dir / f"scene_{i:03d}.tif"), stack)
        tifffile.imwrite(str(mask_dir / f"scene_{i:03d}.tif"), oil.astype(np.uint8))
    return root


# ---------------------------------------------------------------------------
# MKLab layout
# ---------------------------------------------------------------------------

#: Palette matching configs/classes.yaml `mklab.rgb_map`.
MKLAB_PALETTE = {
    0: (0, 0, 0),
    1: (0, 255, 255),
    2: (255, 0, 0),
    3: (153, 76, 0),
    4: (0, 153, 0),
}


def make_mklab(
    root: Path, n: int = 3, size: int = 256, partition: str = "train", rgb_labels: bool = False
) -> Path:
    """Write the MKLab layout with either indexed or RGB label rasters."""
    from PIL import Image

    img_dir = root / partition / "images"
    lbl_dir = root / partition / ("labels" if rgb_labels else "labels_1D")
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        vv, _, oil = synthetic_scene(size, size, seed=100 + i)
        render = np.clip((vv + 30.0) / 30.0, 0.0, 1.0)
        Image.fromarray((render * 255).astype(np.uint8), mode="L").save(
            img_dir / f"patch_{i:03d}.jpg"
        )

        label = np.zeros((size, size), dtype=np.uint8)
        label[oil] = 1
        label[: size // 12, :] = 4          # land strip
        label[size // 12 : size // 8, :] = 2  # look-alike band
        label[size - 6 : size - 2, size - 6 : size - 2] = 3  # ship

        if rgb_labels:
            rgb = np.zeros((size, size, 3), dtype=np.uint8)
            for idx, colour in MKLAB_PALETTE.items():
                rgb[label == idx] = colour
            Image.fromarray(rgb, mode="RGB").save(lbl_dir / f"patch_{i:03d}.png")
        else:
            im = Image.fromarray(label, mode="P")
            flat: list[int] = []
            for idx in range(256):
                flat.extend(MKLAB_PALETTE.get(idx, (0, 0, 0)))
            im.putpalette(flat)
            im.save(lbl_dir / f"patch_{i:03d}.png")
    return root


# ---------------------------------------------------------------------------
# PANGAEA layout
# ---------------------------------------------------------------------------


def make_pangaea_emed(root: Path, n_oil: int = 2, n_clean: int = 2, size: int = 256) -> Path:
    """Write patch rasters plus a GeoJSON annotation file in pixel coordinates."""
    import tifffile

    patch_dir = root / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for i in range(n_oil):
        vv, _, _ = synthetic_scene(size, size, seed=200 + i)
        tifffile.imwrite(str(patch_dir / f"oil_{i:03d}.tif"), vv.astype(np.float32))
        features.append(
            {
                "type": "Feature",
                "properties": {"patch_id": f"oil_{i:03d}", "class": "oil_slick"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[40, 40], [200, 60], [210, 150], [60, 140], [40, 40]]
                    ],
                },
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": {"patch_id": f"oil_{i:03d}", "class": "biogenic"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 190], [90, 195], [85, 240], [12, 235], [10, 190]]],
                },
            }
        )

    for i in range(n_clean):
        vv, _, _ = synthetic_scene(size, size, slick_damping_db=0.0, seed=300 + i)
        tifffile.imwrite(str(patch_dir / f"clean_{i:03d}.tif"), vv.astype(np.float32))

    with (root / "annotations.geojson").open("w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    return root
