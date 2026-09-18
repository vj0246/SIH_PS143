"""Adapter for the Zenodo "Sentinel-1 SAR Oil spill image dataset", Parts I to III.

Distributed layout, as documented on the Zenodo records:

    <root>/
        images/      *.tif   float32, (2048, 2048, 2), sigma0 in dB, [VV, VH]
        masks/       *.tif   uint8,   (2048, 2048),    1 = oil, 0 = everything else

Filenames pair by stem. Some releases nest these under ``train/`` and ``test/``
subdirectories; both layouts are handled.

The critical behaviour here is decision D2. These masks are binary. The zero
class is not sea, it is "not annotated as oil", and it demonstrably contains
land, ships and look-alikes. Mapping it to sea would teach the model that
biogenic films are open water, which is precisely the failure this project
exists to avoid. It is therefore mapped to ignore, driven by
``background_policy: ignore`` in ``configs/classes.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import apply_expressible_mask, remap_index_mask
from .base import DatasetAdapter, Sample, register

_IMAGE_DIRS = ("images", "image", "img")
_MASK_DIRS = ("masks", "mask", "labels", "label")
_RASTER_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


def _find_pair_dirs(root: Path) -> list[tuple[Path, Path]]:
    """Locate every (images, masks) directory pair under ``root``.

    Handles both the flat layout and the train/test split layout.
    """
    pairs: list[tuple[Path, Path]] = []
    candidates = [root, *(p for p in sorted(root.iterdir()) if p.is_dir())] if root.is_dir() else []
    for base in candidates:
        img = next((base / d for d in _IMAGE_DIRS if (base / d).is_dir()), None)
        msk = next((base / d for d in _MASK_DIRS if (base / d).is_dir()), None)
        if img and msk:
            pairs.append((img, msk))
    return pairs


def _read_tiff(path: Path) -> np.ndarray:
    import tifffile

    return tifffile.imread(str(path))


@register
class ZenodoS1Adapter(DatasetAdapter):
    """Binary-labelled Sentinel-1 dual polarisation scenes."""

    name = "zenodo_s1"

    def discover(self) -> list[dict]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name}: {self.root} does not exist. Run `oilspill fetch "
                f"--source zenodo_s1_part1` on a machine with network access."
            )

        pairs = _find_pair_dirs(self.root)
        if not pairs:
            raise FileNotFoundError(
                f"{self.name}: no images/ + masks/ directory pair found under {self.root}. "
                f"Looked for {_IMAGE_DIRS} alongside {_MASK_DIRS}, both directly under "
                "the root and one level down."
            )

        items: list[dict] = []
        for img_dir, mask_dir in pairs:
            masks = {
                p.stem: p for p in mask_dir.iterdir() if p.suffix in _RASTER_SUFFIXES
            }
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix not in _RASTER_SUFFIXES:
                    continue
                mask_path = masks.get(img_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(
                        f"{self.name}: image {img_path.name} has no matching mask in "
                        f"{mask_dir}. Refusing to emit an unlabelled scene."
                    )
                items.append(
                    {
                        "sample_id": f"{img_dir.parent.name}/{img_path.stem}",
                        "image": str(img_path),
                        "mask": str(mask_path),
                        "partition": img_dir.parent.name,
                    }
                )
        return sorted(items, key=lambda d: d["sample_id"])

    def load_one(self, item: dict) -> Sample:
        img = _read_tiff(Path(item["image"]))
        raw_mask = _read_tiff(Path(item["mask"]))

        if img.ndim == 2:
            img = img[..., None]
        if img.ndim != 3:
            raise ValueError(
                f"{item['sample_id']}: expected (H, W, C) image, got shape {img.shape}"
            )

        n_ch = img.shape[2]
        if n_ch == 2:
            band_names = ["vv_db", "vh_db"]
        elif n_ch == 1:
            band_names = ["vv_db"]
        else:
            raise ValueError(
                f"{item['sample_id']}: expected 1 or 2 polarisation channels, got {n_ch}"
            )

        bands = np.moveaxis(img, 2, 0).astype(np.float32, copy=False)

        if raw_mask.ndim == 3:
            # Some releases store the binary mask as a single-channel raster with
            # a trailing dimension, or triplicated to RGB.
            flat = raw_mask.reshape(raw_mask.shape[0], raw_mask.shape[1], -1)
            if flat.shape[2] > 1 and not np.all(flat[..., :1] == flat):
                raise ValueError(
                    f"{item['sample_id']}: multi channel mask whose channels differ; "
                    "this source is documented as binary single band"
                )
            raw_mask = flat[..., 0]

        # Some releases store the positive class as 255 rather than 1.
        uniques = set(np.unique(raw_mask).tolist())
        if uniques <= {0, 255} and 255 in uniques:
            raw_mask = (raw_mask > 0).astype(np.uint8)

        mask = remap_index_mask(raw_mask, self.spec)
        mask = apply_expressible_mask(mask, self.spec)

        return Sample(
            sample_id=item["sample_id"],
            source=self.name,
            bands=bands,
            band_names=band_names,
            mask=mask,
            attrs={
                "partition": item.get("partition"),
                "label_completeness": "binary_oil_only",
                "image_path": item["image"],
                "mask_path": item["mask"],
            },
        )
