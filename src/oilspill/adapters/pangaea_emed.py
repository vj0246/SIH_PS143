"""Adapter for the PANGAEA Eastern Mediterranean oil slick and look-alike dataset.

Earth Syst. Sci. Data 17, 6807 (2025), doi:10.1594/PANGAEA.980773. CC-BY-4.0.
3225 oil objects across 1365 Sentinel-1 patches, plus 2290 no-oil patches whose
look-alikes are clustered into 12 water and 5 coastal groups. It carries the
richest public look-alike taxonomy found, which is what makes it the most
valuable source for the false alarm problem.

This source is **object annotated**, not pixel annotated, so the adapter
rasterises object geometry into masks. Whether the resulting boundaries are
tight enough to supervise segmentation, as opposed to only scoring
detection, is an open question recorded in decisions.md and must be settled by
looking at the real files.

Expected layout, probed in this order:

    <root>/
        patches/ or images/       *.tif | *.png     patch rasters
        annotations.geojson  |  annotations.json  |  annotations.csv

Annotation records must supply, per object: the patch identifier, a class label
resolvable through ``object_class_map`` in configs/classes.yaml, and a geometry.
Geometry may be a polygon ring or a bounding box, in pixel coordinates by
default or in the patch CRS when ``coord_space`` says ``crs``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..schema import IGNORE_ID, apply_expressible_mask
from .base import DatasetAdapter, Sample, register

_PATCH_DIRS = ("patches", "images", "img", "scenes")
_RASTER_SUFFIXES = (".tif", ".tiff", ".png")
_ANNOTATION_NAMES = (
    "annotations.geojson",
    "annotations.json",
    "annotations.csv",
    "objects.geojson",
    "objects.csv",
    "labels.csv",
)


def _read_raster(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(path))
    from skimage import io as skio

    return skio.imread(str(path))


def _load_annotations(path: Path) -> list[dict]:
    """Normalise GeoJSON or CSV annotations into a flat record list."""
    if path.suffix.lower() in (".geojson", ".json"):
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        features = doc.get("features", doc if isinstance(doc, list) else [])
        records = []
        for feat in features:
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            records.append(
                {
                    "patch_id": str(
                        props.get("patch_id")
                        or props.get("patch")
                        or props.get("image_id")
                        or props.get("scene")
                        or ""
                    ),
                    "label": str(props.get("class") or props.get("label") or props.get("type") or ""),
                    "geometry": geom,
                    "coord_space": str(props.get("coord_space", "pixel")),
                }
            )
        return records

    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    records = []
    for row in rows:
        norm = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        patch = norm.get("patch_id") or norm.get("patch") or norm.get("image_id") or ""
        label = norm.get("class") or norm.get("label") or norm.get("type") or ""
        if "wkt" in norm and norm["wkt"]:
            geom = {"type": "WKT", "wkt": norm["wkt"]}
        elif all(k in norm for k in ("xmin", "ymin", "xmax", "ymax")):
            geom = {
                "type": "BBox",
                "bbox": [
                    float(norm["xmin"]),
                    float(norm["ymin"]),
                    float(norm["xmax"]),
                    float(norm["ymax"]),
                ],
            }
        else:
            raise ValueError(
                f"annotation row for patch {patch!r} has neither a wkt column nor "
                "xmin/ymin/xmax/ymax columns"
            )
        records.append(
            {
                "patch_id": patch,
                "label": label,
                "geometry": geom,
                "coord_space": norm.get("coord_space", "pixel"),
            }
        )
    return records


def _rings_from_geometry(geom: dict) -> list[np.ndarray]:
    """Extract polygon rings as ``(N, 2)`` arrays of ``(x, y)``."""
    gtype = str(geom.get("type", "")).lower()

    if gtype == "bbox":
        x0, y0, x1, y1 = geom["bbox"]
        return [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)]

    if gtype == "wkt":
        text = geom["wkt"].strip()
        head, _, body = text.partition("(")
        if "POLYGON" not in head.upper():
            raise ValueError(f"only POLYGON WKT is supported, got {head.strip()!r}")
        body = body.rstrip(")")
        ring_texts = [t for t in body.replace("(", "").split(")") if t.strip()]
        rings = []
        for rt in ring_texts:
            pts = [p.strip() for p in rt.split(",") if p.strip()]
            rings.append(np.array([[float(v) for v in p.split()[:2]] for p in pts]))
        return rings

    if gtype == "polygon":
        return [np.asarray(ring, dtype=np.float64) for ring in geom["coordinates"]]

    if gtype == "multipolygon":
        rings = []
        for poly in geom["coordinates"]:
            rings.extend(np.asarray(r, dtype=np.float64) for r in poly)
        return rings

    raise ValueError(f"unsupported geometry type {geom.get('type')!r}")


def _crs_to_pixel(
    xy: np.ndarray, geotransform: tuple[float, float, float, float, float, float]
) -> np.ndarray:
    """Invert a GDAL affine geotransform. Assumes no rotation terms."""
    x0, dx, rx, y0, ry, dy = geotransform
    if rx != 0.0 or ry != 0.0:
        raise ValueError("rotated geotransforms are not supported")
    col = (xy[:, 0] - x0) / dx
    row = (xy[:, 1] - y0) / dy
    return np.column_stack([col, row])


@register
class PangaeaEMedAdapter(DatasetAdapter):
    """Object annotated Eastern Mediterranean slicks and look-alikes."""

    name = "pangaea_emed"

    def _annotation_path(self) -> Path:
        for candidate in _ANNOTATION_NAMES:
            path = self.root / candidate
            if path.is_file():
                return path
        found = sorted(p.name for p in self.root.iterdir()) if self.root.is_dir() else []
        raise FileNotFoundError(
            f"{self.name}: no annotation file found under {self.root}. "
            f"Looked for {_ANNOTATION_NAMES}. Directory contains: {found[:40]}"
        )

    def _patch_dir(self) -> Path:
        for candidate in _PATCH_DIRS:
            path = self.root / candidate
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"{self.name}: no patch directory found under {self.root}. "
            f"Looked for {_PATCH_DIRS}."
        )

    def discover(self) -> list[dict]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.name}: {self.root} does not exist. Fetch it from "
                "https://doi.pangaea.de/10.1594/PANGAEA.980773"
            )

        patch_dir = self._patch_dir()
        records = _load_annotations(self._annotation_path())

        by_patch: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            by_patch[rec["patch_id"]].append(rec)

        patches = {
            p.stem: p for p in sorted(patch_dir.iterdir()) if p.suffix.lower() in _RASTER_SUFFIXES
        }

        unknown = sorted(set(by_patch) - set(patches))
        if unknown:
            raise FileNotFoundError(
                f"{self.name}: annotations reference {len(unknown)} patches with no raster, "
                f"first few: {unknown[:5]}"
            )

        items = []
        for stem, path in patches.items():
            objects = by_patch.get(stem, [])
            items.append(
                {
                    "sample_id": stem,
                    "raster": str(path),
                    "objects": objects,
                    # A patch with no annotated objects is a genuine no-oil patch
                    # from the look-alike set, not a missing label.
                    "has_objects": bool(objects),
                }
            )
        return sorted(items, key=lambda d: d["sample_id"])

    def load_one(self, item: dict) -> Sample:
        from skimage.draw import polygon as sk_polygon

        raster = _read_raster(Path(item["raster"]))
        if raster.ndim == 2:
            raster = raster[..., None]
        height, width = raster.shape[:2]
        bands = np.moveaxis(raster, 2, 0).astype(np.float32, copy=False)
        band_names = ["vv_db"] if bands.shape[0] == 1 else ["vv_db", "vh_db"][: bands.shape[0]]

        # background_policy for this source is `sea`: objects mark oil and
        # look-alikes, so unmarked water inside an annotated patch is real sea.
        mask = np.full((height, width), self.spec.background_id, dtype=np.uint8)

        unmapped: set[str] = set()
        for obj in item["objects"]:
            label = obj["label"].strip().lower().replace(" ", "_").replace("-", "_")
            class_id = self.spec.object_class_map.get(label)
            if class_id is None:
                unmapped.add(obj["label"])
                continue
            for ring in _rings_from_geometry(obj["geometry"]):
                if obj.get("coord_space") == "crs":
                    raise ValueError(
                        f"{item['sample_id']}: object geometry is in CRS coordinates but the "
                        "patch carries no geotransform; supply pixel coordinates or extend "
                        "the adapter to read the raster's affine transform"
                    )
                rr, cc = sk_polygon(ring[:, 1], ring[:, 0], shape=(height, width))
                mask[rr, cc] = class_id

        if unmapped:
            raise ValueError(
                f"{self.name}: patch {item['sample_id']} has object labels absent from "
                f"object_class_map: {sorted(unmapped)}. Add them to configs/classes.yaml "
                "rather than dropping them silently."
            )

        mask = apply_expressible_mask(mask, self.spec)

        return Sample(
            sample_id=item["sample_id"],
            source=self.name,
            bands=bands,
            band_names=band_names,
            mask=mask,
            attrs={
                "label_completeness": "objects_oil_and_lookalike",
                "n_objects": len(item["objects"]),
                "is_no_oil_patch": not item["has_objects"],
                "raster_path": item["raster"],
                "note": "land is not annotated by this source and stays sea; supply an "
                "external land mask before using coastal patches",
            },
        )
