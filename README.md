# Global SAR oil slick detection: analysis ready data layer

Stage 1 of an oil spill detection and vessel attribution pipeline. This
repository covers **detection only**: turning heterogeneous public SAR datasets
and live Sentinel-1 scenes into one consistent, honestly labelled, honestly
split training corpus. Characterisation, drift hindcast and AIS attribution are
Stage 2 and Stage 3 and are not here.

Read `decisions.md` first. It records every design decision with the rejected
alternative and the reason, including three bugs that the test suite caught and
that would otherwise have quietly degraded every result downstream.

## Why the data layer first

The published evidence says the bottleneck is not architecture. A model trained
on Mediterranean Sentinel-1 falls from 67.8 to 51.8 percent mIoU when applied to
Peruvian waters, and the largest reported gains at global scale come from
co-training oil against look-alikes on a big, diverse corpus rather than from a
better decoder. So this layer is built around four things that most published
pipelines get wrong:

1. **Look-alikes are a class, not noise.** Biogenic films, low wind cells, rain
   cells and internal waves outnumber real slicks by roughly an order of
   magnitude in any real archive. The schema has five classes and the loss sees
   the oil versus look-alike boundary directly.
2. **Sources supervise different things, and pretending otherwise is a bug.**
   Binary-labelled datasets do not say "this is sea", they say "this is not
   annotated as oil", and that remainder contains land, ships and look-alikes.
   Merging it into the sea class teaches the model that biogenic films are open
   water. Those pixels become `ignore` and are excluded from loss and metrics.
3. **Random splits leak and inflate.** Tiles from one scene share a slick, a sea
   state, a wind field, an incidence ramp and a speckle realisation. Splits are
   blocked by scene, by geographic cell, or by whole source.
4. **Detection has a physical validity window.** Below roughly 3 m/s the sea is
   too smooth for a slick to contrast against, above roughly 10 m/s wave
   breaking overwhelms the damping. Outside it the correct output is
   `not_observable`, which is a different statement from "no oil detected".

## Install

```bash
pip install -e ".[dev]"
pytest -q          # 95 tests, no network needed
```

## Pipeline

```
fetch  ->  verify  ->  build  ->  split  ->  stats
```

```bash
# What is declared, and what each source can supervise
oilspill sources

# Check remote size against free disk before committing to a multi-GB download
oilspill disk --source zenodo_s1_part1

# Resumable, checksum verified download
oilspill fetch --source zenodo_s1_part1 --root data/raw

# Report every undeclared label value at once, before converting anything.
# Mandatory for any source whose palette is marked provisional.
oilspill verify --source mklab --root data/raw/mklab

# Adapt, featurise, tile, write the store and the manifest
oilspill build --source zenodo_s1 --root data/raw/zenodo_s1 \
               --out data/ard --manifest data/ard/manifest.parquet \
               --tile-size 256 --overlap 32 --negative-ratio 0.15

# Add another source into the same manifest
oilspill build --source pangaea_emed --root data/raw/pangaea_emed \
               --out data/ard --manifest data/ard/manifest.parquet --append

# Blocked splits, verified disjoint
oilspill split --manifest data/ard/manifest.parquet --strategy scene
oilspill split --manifest data/ard/manifest.parquet \
               --strategy cross_source --test-sources pangaea_emed

# Per source summary plus integrity checks
oilspill stats --manifest data/ard/manifest.parquet

# Query real Sentinel-1 over any area on earth. No credentials needed to search.
oilspill search --bbox 72.0,18.0,73.5,19.5 \
                --start 2024-01-01T00:00:00.000Z \
                --end   2024-01-31T00:00:00.000Z --top 20
```

## Label schema

| id | name | meaning |
|----|------|---------|
| 0 | sea | open water, no film |
| 1 | oil | oil slick or oil-like anthropogenic film |
| 2 | look_alike | biogenic film, low wind, rain cell, upwelling, internal wave, wake |
| 3 | ship | vessel or bright point target |
| 4 | land | land, island, fixed structure |
| 255 | ignore | not expressible by the source; excluded from loss and metrics |

What each source can actually supervise is declared in `configs/classes.yaml`
and enforced at load time. `oilspill stats` shows the consequence directly: a
binary source reports two thirds of its pixels as `ignore` and zero percent
`sea`, while an object-annotated source reports 95 percent `sea`. Those are not
comparable corpora and the manifest never pretends they are.

Any label value or colour not declared in the config raises rather than
defaulting. A silently mismapped class would delete a category from training and
the only symptom would be a mildly disappointing metric.

## Channels

Alongside VV and VH in dB, each tile can carry physical conditioning variables,
all selectable so the ablation stays available:

| channel | what it is |
|---------|-----------|
| `incidence_deg` | incidence angle, supplied or synthesised across the swath |
| `wind_ms` | CMOD5.N wind speed inverted from the background backscatter |
| `damping_db` | backscatter suppression against the local background |
| `local_var_db` | local variance, separating a damped film from a calm cell |
| `dist_land_km` | distance to nearest land |

Slick contrast is not absolute: the same film reads differently at 30 and 45
degrees of incidence and at 4 and 9 m/s of wind. Handing the network those
variables is cheaper than making it infer them from a few thousand patches.

Sources whose radiometry is an 8 bit rendering rather than calibrated sigma0
skip the physics channels entirely, because CMOD5.N and the damping ratio are
defined on sigma0 and applying them to a rendering produces numbers that look
plausible and mean nothing.

## Data sources

| source | what it gives | access |
|--------|---------------|--------|
| Zenodo S1 oil spill, Parts I to III | 1200 scenes each, 2048 x 2048, VV and VH in dB, binary oil masks | automatic, CC-BY-4.0 |
| PANGAEA Eastern Mediterranean | 3225 oil objects over 1365 patches plus 2290 no-oil patches, 12 water and 5 coastal look-alike clusters | automatic, CC-BY-4.0 |
| MKLab / m4d.iti.gr | the reference 5 class benchmark | request form |
| GlobalOSD-SAR | over 100000 annotated slick and look-alike images from 500000+ global scenes, 2014 to 2021 | announced public, no repository URL found; worth an email to the authors |
| CDSE Sentinel-1 | live global GRD and SLC | free registration |

Credentials for CDSE are read from `CDSE_USERNAME` and `CDSE_PASSWORD` in the
environment. They are never written to disk or into any file in this repository.

## Where the downloads run

`oilspill fetch` is designed to run on a machine with open network access. The
sandbox this code was authored in permits only GitHub and PyPI; zenodo.org,
pangaea.de, m4d.iti.gr, dataspace.copernicus.eu, ASF, Kaggle and Hugging Face
are all refused at the proxy. The whole pipeline is therefore verified against
synthetic fixtures that reproduce each source's exact on-disk layout, so
correctness never depended on having the real bytes present. Running the same
tests against the real archives is the acceptance check once they land.

## Layout

```
configs/
  classes.yaml        unified schema and per-source remapping, the contract
  datasets.yaml       source registry: urls, licences, citations, notes
src/oilspill/
  schema.py           class schema, remapping, loud failure on unknown values
  adapters/           one file per source, all yielding the same Sample
  ard/
    wind.py           CMOD5.N forward and inverse, observability gate
    features.py       physics channels
    tiling.py         deterministic content aware tiler
  io.py               sharded tile store, cross-store address resolution
  manifest.py         parquet manifest and integrity checks
  splits.py           blocked split assignment and leak verification
  build.py            adapter to features to tiles to manifest
  fetch.py            resumable checksum verified downloader
  cdse.py             Copernicus Data Space query and download
  cli.py              command line interface
tests/
  fixtures.py         synthetic datasets in each source's real layout
```

## What is deliberately not claimed

- The damping ratio is a physical measure of surface film damping. It is not an
  oil type classifier, and nothing here treats it as one.
- CMOD inverts backscatter to wind given a wind *direction*, which a single SAR
  image does not supply. The assumed direction travels with every retrieval as
  `wind_phi_assumed_deg`, and `direction_sensitivity` reports the resulting
  spread.
- Wind inversion is capped at 30 m/s because CMOD5.N stops being monotonic above
  roughly 31.8 m/s at the near edge of the IW swath, where bisection would return
  a plausible wrong root. See decisions.md D11.
- The MKLab palette is transcribed from a publication, not confirmed against the
  distributed files. It is flagged `provisional` and `oilspill verify` exists
  precisely to catch it being wrong.
