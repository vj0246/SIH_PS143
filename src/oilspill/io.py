"""Sharded tile store.

Tiles are written into compressed ``.npz`` shards of a fixed number of tiles, and
addressed from the manifest by ``(shard, index)``. Shards keep the file count
manageable at global scale, where a full Sentinel-1 ingest reaches millions of
tiles and one file per tile would make the filesystem the bottleneck.

The format is deliberately boring. No custom binary layout, no database, nothing
that needs a migration path. A shard can be opened with numpy alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .ard.tiling import Tile


@dataclass
class ShardRef:
    """Where one tile lives.

    ``store`` is part of the address, not decoration. Each source writes into its
    own store directory, so shard filenames restart at ``tiles_00000.npz`` for
    every source and ``(shard, index)`` alone collides the moment a second source
    is built. That collision is silent: the manifest looks fine and training
    reads the wrong pixels for some fraction of tiles.
    """

    store: str
    shard: str
    index: int


class TileStoreSet:
    """Resolve manifest addresses across several per-source stores.

    Training reads the manifest, which spans every source, so it needs one object
    that can turn ``(store, shard, index)`` into pixels regardless of which store
    the row came from. Readers are opened lazily and cached.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._readers: dict[str, "TileReader"] = {}

    def reader(self, store: str) -> "TileReader":
        if store not in self._readers:
            self._readers[store] = TileReader(self.root / store)
        return self._readers[store]

    def read(self, store: str, shard: str, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        return self.reader(store).read(shard, index)

    def read_row(self, row) -> tuple[np.ndarray, np.ndarray, str]:
        """Read the tile described by one manifest row."""
        return self.read(str(row["store"]), str(row["shard"]), int(row["shard_index"]))

    def stores(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/store.json"))


class TileWriter:
    """Accumulate tiles and flush them into shards.

    Use as a context manager so the final partial shard is always flushed::

        with TileWriter(root, shard_size=512) as writer:
            for tile in tiles:
                ref = writer.add(tile)
    """

    def __init__(
        self,
        root: str | Path,
        shard_size: int = 512,
        prefix: str = "tiles",
        store_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store_name = store_name or self.root.name
        self.shard_size = shard_size
        self.prefix = prefix
        self._buffer: list[Tile] = []
        self._shard_index = 0
        self._refs: dict[str, ShardRef] = {}
        self._band_names: list[str] | None = None

    def __enter__(self) -> "TileWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, tile: Tile) -> ShardRef:
        """Buffer a tile, flushing when the shard fills.

        Raises:
            ValueError: If band names differ between tiles. A store with mixed
                channel layouts cannot be read back coherently, and finding out
                at training time would be a long debugging session.
        """
        if self._band_names is None:
            self._band_names = list(tile.band_names)
        elif list(tile.band_names) != self._band_names:
            raise ValueError(
                f"tile {tile.tile_id} has bands {tile.band_names} but this store was "
                f"opened with {self._band_names}; write mixed layouts to separate stores"
            )

        ref = ShardRef(
            store=self.store_name, shard=self._shard_name(), index=len(self._buffer)
        )
        self._refs[tile.tile_id] = ref
        self._buffer.append(tile)
        if len(self._buffer) >= self.shard_size:
            self._flush()
        return ref

    def extend(self, tiles: Iterable[Tile]) -> list[ShardRef]:
        return [self.add(t) for t in tiles]

    def close(self) -> None:
        """Flush any buffered tiles and write the store metadata sidecar."""
        if self._buffer:
            self._flush()
        meta = {
            "store_name": self.store_name,
            "prefix": self.prefix,
            "shard_size": self.shard_size,
            "band_names": self._band_names or [],
            "n_shards": self._shard_index,
            "n_tiles": len(self._refs),
        }
        (self.root / "store.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @property
    def refs(self) -> dict[str, ShardRef]:
        return dict(self._refs)

    def _shard_name(self) -> str:
        return f"{self.prefix}_{self._shard_index:05d}.npz"

    def _flush(self) -> None:
        if not self._buffer:
            return
        path = self.root / self._shard_name()
        np.savez_compressed(
            path,
            bands=np.stack([t.bands for t in self._buffer]).astype(np.float32),
            masks=np.stack([t.mask for t in self._buffer]).astype(np.uint8),
            tile_ids=np.array([t.tile_id for t in self._buffer], dtype=object),
        )
        self._buffer.clear()
        self._shard_index += 1


class TileReader:
    """Random access into a tile store, caching one shard at a time.

    Sequential access ordered by shard is dramatically faster than random access
    across shards, so shuffle within a shard and iterate shards in random order
    rather than shuffling globally.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "store.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"no store.json in {self.root}; was TileWriter closed?")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.band_names: list[str] = list(self.meta.get("band_names", []))
        self._cached_shard: str | None = None
        self._cache: dict | None = None

    def _load(self, shard: str) -> dict:
        if shard != self._cached_shard:
            with np.load(self.root / shard, allow_pickle=True) as data:
                self._cache = {
                    "bands": data["bands"],
                    "masks": data["masks"],
                    "tile_ids": data["tile_ids"],
                }
            self._cached_shard = shard
        assert self._cache is not None
        return self._cache

    def read(self, shard: str, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        """Return ``(bands, mask, tile_id)`` for one tile."""
        data = self._load(shard)
        return (
            data["bands"][index],
            data["masks"][index],
            str(data["tile_ids"][index]),
        )

    def iter_shard(self, shard: str) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        data = self._load(shard)
        for i in range(len(data["tile_ids"])):
            yield data["bands"][i], data["masks"][i], str(data["tile_ids"][i])

    def shards(self) -> list[str]:
        return sorted(p.name for p in self.root.glob(f"{self.meta['prefix']}_*.npz"))

    def __len__(self) -> int:
        return int(self.meta.get("n_tiles", 0))
