"""Adapter for the MKLab / m4d.iti.gr oil spill detection dataset.

Krestenitis et al., Remote Sensing 11(15) 1762, 2019. The reference benchmark
for five class oil spill segmentation.

Distributed layout:

    <root>/
        train/
            images/  *.jpg   Sentinel-1 VV, rendered to 8 bit
            labels_1D/ *.png palette indexed masks
        test/
            images/
            labels_1D/

Some mirrors ship ``labels/`` with RGB colour coded masks instead of, or
alongside, ``labels_1D/``. Both are handled: an RGB mask goes through the
``rgb_map`` table and an indexed mask through ``index_map``.

Label caveat, decision D7. The palette in ``configs/classes.yaml`` for this
source is marked ``provisional``: it was transcribed from the publication rather
than confirmed against the distributed files, because the distribution is behind
a request form. Any value not in the table raises rather than defaulting, and
``oilspill verify --source mklab`` reports every unmapped value at once. Run
that before training on this source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schema import apply_expressible_mask, remap_index_mask, remap_rgb_mask
from .base import DatasetAdapter, Sample, register

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_LABEL_DIRS_INDEXED = ("labels_1D", "labels_1d", "labels_index")
_LABEL_DIRS_RGB = ("labels", "labels_rgb", "masks")


def _read_image(path: Path) -> np.ndarray:
    import tifffile
    from skimage import io as skio

    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(str(path))
    return skio.imread(str(path))


def _read_label(path: Path) -> tuple[np.ndarray, bool]:
    """Read a label raster.

    Returns:
        ``(array, is_rgb)``. A palette PNG is read as its index array when it
        has no colour dimension, and as RGB otherwise.
    """
    from PIL import Image

    with Image.open(path) as im:
        if im.mode == "P":
            # Preserve the palette indices rather than resolving them, so the
            # index_map applies. Resolving would depend on the writer's palette.
            return np.array(im, dtype=np.uint8), False
        if im.mode in ("L", "I;16", "I"):
            return np.array(im), False
        return np.array(im.convert("RGB")), True


@register
class MKLabAdapter(DatasetAdapter):
    """Five class oil spill segmentation benchmark."""

    name = "mklab"

    def discover(self) -> list[dict]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name}: {self.root} does not exist. This dataset is distributed "
                "via a request form at https://m4d.iti.gr/oil-spill-detection-dataset/ ; "
                "see configs/datasets.yaml manual_instructions."
            )

        partitions = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        if not partitions:
            partitions = [self.root]

        items: list[dict] = []
        for part in partitions:
            img_dir = part / "images"
            if not img_dir.is_dir():
                continue
            lbl_dir = next(
                (part / d for d in (*_LABEL_DIRS_INDEXED, *_LABEL_DIRS_RGB) if (part / d).is_dir()),
                None,
            )
            if lbl_dir is None:
                raise FileNotFoundError(
                    f"{self.name}: {part} has images/ but no label directory. "
                    f"Looked for {_LABEL_DIRS_INDEXED + _LABEL_DIRS_RGB}."
                )
            indexed = lbl_dir.name in _LABEL_DIRS_INDEXED
            labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES}

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                lbl_path = labels.get(img_path.stem)
                if lbl_path is None:
                    raise FileNotFoundError(
                        f"{self.name}: image {img_path.name} has no label in {lbl_dir}"
                    )
                items.append(
                    {
                        "sample_id": f"{part.name}/{img_path.stem}",
                        "image": str(img_path),
                        "label": str(lbl_path),
                        "indexed": indexed,
                        "partition": part.name,
                    }
                )
        if not items:
            raise FileNotFoundError(f"{self.name}: no image/label pairs found under {self.root}")
        return sorted(items, key=lambda d: d["sample_id"])

    def load_one(self, item: dict) -> Sample:
        img = _read_image(Path(item["image"]))
        raw, is_rgb = _read_label(Path(item["label"]))

        if img.ndim == 3:
            # The distributed images are 8 bit renderings of a single VV band,
            # triplicated to RGB by the export. Collapse only if identical.
            chans = img.reshape(img.shape[0], img.shape[1], -1)
            if chans.shape[2] >= 3 and np.array_equal(chans[..., 0], chans[..., 1]):
                img = chans[..., 0]
            else:
                img = chans[..., 0]
        img = np.asarray(img)

        # These are 8 bit renderings, not calibrated sigma0. Naming the channel
        # honestly prevents it from being silently mixed with dB rasters.
        bands = img.astype(np.float32)[None, ...] / 255.0
        band_names = ["vv_render_norm"]

        if is_rgb:
            mask = remap_rgb_mask(raw, self.spec)
        else:
            mask = remap_index_mask(raw, self.spec)
        mask = apply_expressible_mask(mask, self.spec)

        return Sample(
            sample_id=item["sample_id"],
            source=self.name,
            bands=bands,
            band_names=band_names,
            mask=mask,
            attrs={
                "partition": item.get("partition"),
                "label_completeness": "full_5class",
                "radiometry": "8bit_render_not_calibrated",
                "provisional_palette": self.spec.provisional,
                "image_path": item["image"],
                "label_path": item["label"],
            },
        )
