"""Command line interface.

    oilspill sources                 list registered sources and their label reach
    oilspill disk    --source S      remote size versus free space, before fetching
    oilspill fetch   --source S      resumable checksum verified download
    oilspill verify  --source S      report undeclared label values before any build
    oilspill build   --source S      adapt, featurise, tile, write store and manifest
    oilspill split                   assign blocked train/val/test
    oilspill stats                   per source manifest summary and integrity report
    oilspill search                  query CDSE for real Sentinel-1 scenes
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .ard.tiling import TileSpec
from .build import BuildConfig, build_source
from .manifest import integrity_report, read_manifest, summarise, write_manifest
from .schema import CLASSES, list_sources, load_source_spec, scan_unmapped
from .splits import SplitConfig, assign_splits, verify_disjoint

app = typer.Typer(add_completion=False, help="Global SAR oil slick detection: ARD layer")
console = Console()


@app.command()
def sources() -> None:
    """List declared sources and what each can supervise."""
    table = Table(title="Declared sources")
    table.add_column("source")
    table.add_column("encoding")
    table.add_column("expresses")
    table.add_column("background")
    table.add_column("provisional")

    for name in list_sources():
        spec = load_source_spec(name)
        table.add_row(
            name,
            spec.encoding,
            ", ".join(sorted(spec.expresses)) or "-",
            spec.background_policy,
            "yes" if spec.provisional else "",
        )
    console.print(table)

    schema = Table(title="Unified schema")
    schema.add_column("id")
    schema.add_column("name")
    schema.add_column("description")
    for c in CLASSES:
        schema.add_row(str(c.id), c.name, c.description)
    schema.add_row("255", "ignore", "not expressible by the source; excluded from loss and metrics")
    console.print(schema)


@app.command()
def disk(source: str = typer.Option(..., "--source", "-s")) -> None:
    """Report remote size against free space before committing to a download."""
    from .fetch import estimate_disk

    report = estimate_disk(source)
    console.print_json(json.dumps(report))
    if report.get("known") and report["gigabytes"] > report["free_gigabytes"]:
        console.print(
            f"[red]Not enough space: needs {report['gigabytes']} GB, "
            f"{report['free_gigabytes']} GB free[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def fetch(
    source: str = typer.Option(..., "--source", "-s"),
    root: str = typer.Option("data/raw", "--root"),
    verify_existing: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Download one registry source, resuming and verifying."""
    from .fetch import fetch_source

    with console.status(f"fetching {source}"):
        report = fetch_source(source, root=root, verify_existing=verify_existing)

    if report["status"] in ("manual", "unavailable"):
        console.print(f"[yellow]{source} is not automatically downloadable[/yellow]")
        console.print(report.get("manual_instructions") or "")
        if report.get("landing"):
            console.print(f"Landing page: {report['landing']}")
        raise typer.Exit(code=0)

    console.print(
        f"[green]{source}[/green]: {len(report['downloaded'])} downloaded, "
        f"{len(report['verified'])} verified, {len(report['skipped'])} skipped"
    )
    if report.get("citation"):
        console.print(f"Cite: {report['citation']}")


@app.command()
def verify(
    source: str = typer.Option(..., "--source", "-s"),
    root: str = typer.Option(..., "--root"),
    limit: int = typer.Option(50, "--limit"),
) -> None:
    """Report every undeclared label value across a source, without converting.

    Decision D7. Run this before the first build of any source whose palette is
    marked provisional, which currently means MKLab.
    """
    from .adapters import get_adapter

    adapter = get_adapter(source)(root)
    spec = adapter.spec
    items = adapter.discover()[:limit]

    masks = []
    for item in items:
        try:
            masks.append(adapter.load_one(item).mask)
        except Exception:
            # load_one raises on the first bad value; read the raw label instead
            # so the scan sees every source file rather than stopping at one.
            key = next((k for k in ("label", "mask") if k in item), None)
            if key is None:
                raise
            import numpy as np
            from PIL import Image

            path = Path(item[key])
            if path.suffix.lower() in (".tif", ".tiff"):
                import tifffile

                masks.append(tifffile.imread(str(path)))
            else:
                with Image.open(path) as im:
                    masks.append(np.array(im if im.mode == "P" else im.convert("RGB")))

    findings = scan_unmapped(masks, spec)
    console.print(f"scanned {len(masks)} label rasters from [bold]{source}[/bold]")
    if not findings:
        console.print("[green]no undeclared label values[/green]")
        return

    table = Table(title="Undeclared label values")
    table.add_column("value")
    table.add_column("pixels", justify="right")
    for key, count in sorted(findings.items(), key=lambda kv: -kv[1]):
        table.add_row(key, f"{count:,}")
    console.print(table)
    console.print(
        "[red]Add these to configs/classes.yaml before building. "
        "They are not defaulted, by design.[/red]"
    )
    raise typer.Exit(code=1)


@app.command()
def build(
    source: str = typer.Option(..., "--source", "-s"),
    root: str = typer.Option(..., "--root", help="raw files for this source"),
    out: str = typer.Option("data/ard", "--out"),
    manifest: str = typer.Option("data/ard/manifest.parquet", "--manifest"),
    tile_size: int = typer.Option(256, "--tile-size"),
    overlap: int = typer.Option(32, "--overlap"),
    negative_ratio: float = typer.Option(0.15, "--negative-ratio"),
    limit: int = typer.Option(0, "--limit", help="0 means all scenes"),
    features: bool = typer.Option(True, "--features/--no-features"),
    append: bool = typer.Option(False, "--append", help="merge into an existing manifest"),
) -> None:
    """Adapt, featurise and tile one source, then write the manifest."""
    import pandas as pd

    config = BuildConfig(
        tile=TileSpec(size=tile_size, overlap=overlap, negative_keep_ratio=negative_ratio),
        compute_features=features,
    )

    def progress(i: int, n: int, sample_id: str) -> None:
        if i % 10 == 0 or i == n:
            console.print(f"  {i}/{n}  {sample_id}")

    frame = build_source(
        source, root, out, config, limit=limit or None, progress=progress
    )

    manifest_path = Path(manifest)
    if append and manifest_path.is_file():
        existing = read_manifest(manifest_path)
        existing = existing[existing["source"] != source]
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.sort_values("tile_id").reset_index(drop=True)

    write_manifest(frame, manifest_path)
    console.print(f"[green]{len(frame)} tiles[/green] -> {manifest_path}")

    report = integrity_report(frame)
    for err in report["errors"]:
        console.print(f"[red]ERROR {err}[/red]")
    for warn in report["warnings"]:
        console.print(f"[yellow]warning: {warn}[/yellow]")
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command()
def split(
    manifest: str = typer.Option("data/ard/manifest.parquet", "--manifest"),
    strategy: str = typer.Option("scene", "--strategy"),
    train: float = typer.Option(0.7, "--train"),
    val: float = typer.Option(0.15, "--val"),
    cell_degrees: float = typer.Option(1.0, "--cell-degrees"),
    test_sources: str = typer.Option("", "--test-sources", help="comma separated"),
    seed: int = typer.Option(20260910, "--seed"),
) -> None:
    """Assign blocked splits and verify that no group straddles two of them."""
    config = SplitConfig(
        strategy=strategy,
        train=train,
        val=val,
        cell_degrees=cell_degrees,
        seed=seed,
        test_sources=tuple(s for s in test_sources.split(",") if s),
    )
    frame = read_manifest(manifest)
    frame = assign_splits(frame, config)
    write_manifest(frame, manifest)

    report = verify_disjoint(frame, config)
    console.print_json(json.dumps(report, indent=2))
    if report["n_leaked_groups"]:
        console.print("[red]groups straddle splits; every metric on this split is inflated[/red]")
        raise typer.Exit(code=1)
    console.print("[green]splits are disjoint[/green]")


@app.command()
def stats(manifest: str = typer.Option("data/ard/manifest.parquet", "--manifest")) -> None:
    """Per source summary plus the integrity report."""
    import pandas as pd

    frame = read_manifest(manifest)
    summary = summarise(frame)

    # A wide table wraps into illegibility in a normal terminal, so print one
    # block per source with the metrics stacked.
    for _, row in summary.iterrows():
        table = Table(title=f"source: {row['source']}", show_header=False)
        table.add_column("metric", style="bold")
        table.add_column("value", justify="right")
        for column in summary.columns:
            if column == "source":
                continue
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}"
            table.add_row(str(column), str(value))
        console.print(table)

    if "split" in frame.columns and frame["split"].ne("").any():
        pivot = pd.pivot_table(
            frame,
            index="source",
            columns="split",
            values="tile_id",
            aggfunc="count",
            fill_value=0,
        )
        split_table = Table(title="tiles per source and split")
        split_table.add_column("source")
        for column in pivot.columns:
            split_table.add_column(str(column), justify="right")
        for source, row in pivot.iterrows():
            split_table.add_row(str(source), *[str(int(v)) for v in row.tolist()])
        console.print(split_table)

    report = integrity_report(frame)
    for err in report["errors"]:
        console.print(f"[red]ERROR {err}[/red]")
    for warn in report["warnings"]:
        console.print(f"[yellow]warning: {warn}[/yellow]")
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command()
def search(
    bbox: str = typer.Option(..., "--bbox", help="min_lon,min_lat,max_lon,max_lat"),
    start: str = typer.Option(..., "--start", help="2024-01-01T00:00:00.000Z"),
    end: str = typer.Option(..., "--end"),
    product_type: str = typer.Option("GRD", "--product-type"),
    sensor_mode: str = typer.Option("IW", "--mode"),
    top: int = typer.Option(20, "--top"),
) -> None:
    """Query CDSE for Sentinel-1 scenes. No credentials needed to search."""
    from .cdse import CdseClient, summarise_products

    coords = tuple(float(v) for v in bbox.split(","))
    if len(coords) != 4:
        raise typer.BadParameter("bbox needs four comma separated numbers")

    client = CdseClient()
    products = client.search_area(
        coords, start, end, top=top, product_type=product_type, sensor_mode=sensor_mode
    )
    rows = summarise_products(products)

    table = Table(title=f"{len(rows)} Sentinel-1 products")
    for column in ("name", "start", "size_gb", "orbit_direction", "relative_orbit", "online"):
        table.add_column(column)
    for row in rows:
        table.add_row(*[str(row.get(c, "")) for c in
                        ("name", "start", "size_gb", "orbit_direction", "relative_orbit", "online")])
    console.print(table)
    console.print("Product ids:")
    for row in rows:
        console.print(f"  {row['id']}  {row['name']}")


if __name__ == "__main__":
    app()
