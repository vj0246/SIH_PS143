# Decisions log: oil spill detection, Stage 1 (detection only)

Scope of this stage: detect and delineate oil slicks in Sentinel-1 SAR, globally,
separating them from look-alikes. Characterisation, drift hindcast and AIS
attribution are explicitly out of scope and are Stage 2 and Stage 3.

Every decision below is recorded with the alternative that was rejected and the
reason, so that a reviewer can attack the reasoning rather than guess at it.

---

## D1. Unified label schema is 5 classes, not binary

**Decision.** All sources are remapped to a single schema:

| id | name | meaning |
|----|------|---------|
| 0 | sea | open water, no film |
| 1 | oil | oil slick or oil-like anthropogenic film |
| 2 | look_alike | biogenic film, low wind cell, rain cell, upwelling, internal wave, wake |
| 3 | ship | vessel or bright point target |
| 4 | land | land, island, fixed structure |
| 255 | ignore | not expressible by the source, excluded from loss and metrics |

**Rejected.** Binary oil versus background, with a separate downstream look-alike
classifier.

**Reason.** The dominant published failure mode is not missing slicks, it is
false alarms on look-alikes. A binary segmenter never sees the discriminative
boundary between a slick and a biogenic film during training, because both are
just "dark". Recent global-scale work reports that co-training oil and look-alike
samples together is what produced their headline gains. The two-stage
alternative remains available: the multiclass head degrades to binary by argmax
merging, so nothing is lost by starting here.

---

## D2. The ignore class is load bearing, not a convenience

**Decision.** Every source declares which unified classes it can express.
Classes a source cannot express are written as 255 (ignore) in that source's
masks, and the loss and the metrics both skip them.

**Rejected.** Merging every source's "background" into class 0 (sea).

**Reason.** This is the single most damaging silent bug available in this
project. The Zenodo Sentinel-1 sets are binary: oil is 1, everything else is 0.
That "everything else" contains land, ships and, critically, look-alikes. If it
is merged into sea, the model is explicitly taught that biogenic films are open
water, which destroys exactly the capability D1 exists to build. Sources that
cannot express look-alike therefore contribute supervision for oil and sea only,
and their unlabelled remainder is masked out.

Consequence to accept: effective supervision per pixel varies by source, and
metrics must always be reported per source as well as pooled.

---

## D3. Splits are geographically and temporally blocked, never random

**Decision.** Tiles are grouped into spatial blocks (default 1.0 degree cells)
and whole blocks are assigned to train, validation or test. A separate
`leave-one-region-out` mode holds out an entire basin. A separate
`cross-source` mode trains on one source and tests on another.

**Rejected.** Random per-tile splitting.

**Reason.** Neighbouring tiles from the same Sentinel-1 scene share the same
slick, the same sea state, the same wind field, the same incidence angle ramp and
the same speckle realisation. Random splitting leaks all of that across the
split boundary and inflates the reported score without improving the model.
Published cross-domain evidence shows a model at 67.8 percent mIoU in its
training basin falling to 51.8 percent in another basin. A random split cannot
see that 16 point gap; a blocked split reports it as the headline number.

This is a deliberate choice to report a lower, honest number.

---

## D4. Wind is metadata and a gate, not a filter applied at ingest

**Decision.** A CMOD5.N wind field is computed from VV backscatter for every
scene and stored per tile as statistics (mean, min, max, fraction of pixels
inside the detectable band). Tiles are never dropped at ingest. A configurable
observability gate is applied at inference time and at metric-reporting time.

**Rejected.** Dropping low-wind and high-wind tiles during dataset construction.

**Reason.** Two reasons. First, dropping them at ingest makes the decision
irreversible and hides it from the evaluation; a reader cannot then ask how the
detector behaves at 2 m/s. Second, low-wind cells are themselves a look-alike
class the model must learn to reject, so removing them removes the hard negatives.
Storing wind as metadata lets the same dataset answer "how good is detection"
and "when is detection valid" separately.

Default band, configurable: 3.0 to 10.0 m/s at 10 m neutral. Outside it the
system reports `not_observable` rather than a negative.

**Verification note.** The CMOD5.N implementation in `ard/wind.py` is vectorised
in numpy and is unit-tested against the scalar reference implementation shipped
in IFREMER's `xsarsea` package, over a grid of incidence angle, wind speed and
relative wind direction. The coefficient vector is the CMOD5.N (neutral) set.

---

## D5. Physics channels are stacked alongside backscatter

**Decision.** The tile tensor carries, in addition to VV and VH in dB:
local incidence angle, CMOD5.N wind speed, damping ratio relative to a local
background estimate, and a coastal distance proxy. Channels are declared in
config and any subset can be selected at training time.

**Rejected.** Feeding VV and VH alone and relying on the network to infer context.

**Reason.** Slick contrast is not an absolute quantity. The same slick produces
different contrast at 20 and 45 degrees of incidence and at 4 and 9 m/s of wind.
A network given only intensity has to learn that relationship from a few thousand
patches, which is what makes cross-basin transfer collapse. Giving it the
conditioning variables directly is cheap and is the intervention with the best
evidence behind it. The ablation is preserved by making channels selectable.

---

## D6. One adapter per source, one Sample contract

**Decision.** Each dataset gets a single file under `adapters/` that yields the
common `Sample` dataclass. No source-specific logic exists anywhere else in the
codebase.

**Rejected.** A single loader with per-source branches.

**Reason.** Sources will keep being added, including scenes the team labels
themselves. Adding one means writing one file and registering it, with no risk
to the others. It also makes per-source metric reporting (required by D2) fall
out naturally.

---

## D7. Unknown label values fail loudly

**Decision.** Palette and value remapping tables are declared in
`configs/classes.yaml`. Any pixel value or RGB triple encountered in a source
mask that is not in that source's table raises, and `oilspill verify` reports
every unmapped value with its pixel count before any conversion is committed.

**Rejected.** Mapping unknown values to sea or to ignore.

**Reason.** The MKLab palette is documented but was transcribed from a paper.
If a single RGB triple is wrong, silently mapping it to sea would delete an
entire class from the training set and the only symptom would be a mildly
disappointing metric. Failing loudly converts a silent data bug into a startup
error.

---

## D8. Manifest is a parquet table and is the single source of truth

**Decision.** One row per tile, holding tile id, source, on-disk shard and index,
geometry, acquisition time, per-class pixel counts, wind statistics, and split
assignment. Training reads the manifest, never the directory tree.

**Rejected.** Directory-structure-as-metadata, or a JSON index.

**Reason.** Class balance, split integrity, per-source metrics and stratified
sampling are all one query away, and stay correct as sources are added. Parquet
because these tables reach millions of rows once global Sentinel-1 scenes are
ingested and column pruning matters.

---

## D9. Downloads run on the operator's machine, not in the build sandbox

**Decision.** `oilspill fetch` is a resumable, checksum-verified downloader
driven by `configs/datasets.yaml`. It is designed to run wherever the operator
has network access.

**Reason.** Not a design preference. The sandbox this code was authored in has an
egress policy that permits only GitHub and PyPI; zenodo.org, pangaea.de,
m4d.iti.gr, dataspace.copernicus.eu, ASF, Kaggle and Hugging Face are all
refused at the proxy. The fetcher is therefore written to be run elsewhere, and
the entire pipeline is verified against synthetic fixtures that reproduce each
source's exact on-disk layout so that correctness does not depend on having the
real bytes present at authoring time.

---

## D10. Tiling is deterministic and content aware

**Decision.** Fixed 256 pixel tiles with configurable overlap, generated in a
deterministic raster order so a given scene always yields the same tile ids.
Tiles containing oil are retained in full; pure-sea tiles are subsampled to a
configured ratio and the subsampling is seeded and recorded.

**Rejected.** Random crops at training time.

**Reason.** Determinism makes the manifest meaningful and makes a run
reproducible. Oil occupies a very small pixel fraction; without subsampling of
negatives the manifest is 99 percent sea and every downstream statistic is
dominated by empty water. Recording the seed and ratio keeps the discarded
negatives recoverable.

---

## D11. Wind inversion is capped at 30 m/s because CMOD5.N stops being monotonic

**Decision.** `wind_speed_from_sigma0` bisects over 0.2 to 30.0 m/s and refuses,
by default, any incidence angle outside the Sentinel-1 IW range of 29.1 to 46
degrees.

**Reason.** Found while writing the test suite, not assumed. Bisection requires
the forward model to be strictly increasing in wind speed, so the test swept the
coefficient set over incidence and relative wind direction. CMOD5.N turns over:

| incidence | first turnover |
|-----------|----------------|
| 15.0 deg  | 13.0 m/s |
| 17.0 deg  | 24.4 m/s |
| 29.1 deg  | 31.8 m/s |
| 35.0 deg  | 36.3 m/s |
| 40.0 deg  | 45.4 m/s |
| 46.0 deg  | monotonic to 50 m/s |

Above the turnover, bisection converges to a plausible looking but wrong root
and nothing in the output announces it. The cap at 30 m/s sits below the worst
case turnover inside the IW swath. Nothing operational is lost: 30 m/s is
Beaufort 11 and three times the upper edge of the oil detection band.

Two tests hold this in place. One asserts strict monotonicity across the whole
validated domain. The other pins the turnover to between 31 and 33 m/s so that
any future edit to the coefficient vector fails loudly instead of quietly
widening the domain.

**Rejected.** Inverting over the full 0.2 to 50 m/s range published for the GMF,
and trusting the operational wind band to keep the solver away from the bad
region. Rejected because the solver is not confined to the band: it runs on
every pixel, including bright wave-breaking pixels, before any gating happens.

---

## D12. The background estimator uses block means, not a percentile of raw pixels

**Decision.** `background_sigma0` averages linear intensity into blocks first,
then takes an upper percentile across a neighbourhood of blocks. Both terms of
the damping ratio are multilooked in the linear domain before any logarithm.

**Rejected.** The obvious and wrong version, which was written first: take the
85th percentile of raw single-look pixels inside each block.

**Reason.** Found by the test that checks whether the wind channel recovers an
injected wind speed, not by reasoning about it beforehand. SAR intensity is
exponentially distributed at one look, so its 85th percentile sits at 1.90 times
its mean. That is a fixed 2.8 dB overestimate of sigma0, which propagates to
roughly 50 percent overestimate of wind speed: a scene generated at 7.0 m/s came
back as 10.4 m/s, which pushed it past the 10 m/s ceiling and made the pipeline
label a perfectly ordinary scene `not_observable`. Nothing in the output looked
wrong. The values were smooth, plausible and entirely incorrect.

The correction has two parts, and both are needed:

- The block mean of linear intensity is an unbiased estimator of sigma0 and
  collapses speckle by the square root of the block area, which removes the bias.
- An upper percentile taken *across* neighbouring blocks then rejects blocks
  that sit inside a slick, while still tracking genuine large-scale wind
  variation that a single scene-wide scalar would flatten.

Averaging must happen in linear power, never in dB. The mean of the logarithm of
a single-look intensity is 2.5 dB below the logarithm of its mean, so a boxcar
filter applied to a dB raster introduces its own separate bias.

Two tests pin this: one asserts the wind channel recovers 4.0, 7.0 and 9.5 m/s to
within 0.9 m/s through the full path, and one spells out the size of the bias if
the multilooking step is ever removed.

**Generalisation worth carrying forward.** Every ratio, threshold and normaliser
in this pipeline is computed on speckled data. The default assumption should be
that a statistic is biased until a test says otherwise.

---

## D13. The store name is part of a tile's address

**Decision.** A tile is addressed by `(store, shard, shard_index)`. Each source
writes into its own store directory, and the store name travels in the manifest.

**Rejected.** Addressing by `(shard, shard_index)` alone, which is what the first
implementation did.

**Reason.** Found by the multi-source end to end test, not by inspection. Shard
filenames restart at `tiles_00000.npz` inside every store, so the moment a second
source is built, its first shard collides with the first source's. The manifest
looked entirely healthy: unique tile ids, sensible class counts, valid splits.
Training would simply have read the wrong pixels for a large fraction of rows,
with correct labels attached to them, which is close to the worst possible bug
because it degrades results without ever producing an error.

`TileStoreSet` resolves addresses across stores, and the manifest integrity check
now tests uniqueness on the full triple.

---

## Open questions carried into Stage 1 implementation

1. MKLab distribution requires a request form. Until the real masks are in hand,
   the palette in `configs/classes.yaml` is provisional and D7 is what protects
   against it being wrong.
2. The PANGAEA Eastern Mediterranean set is object-annotated rather than
   pixel-annotated. The adapter rasterises object geometry; whether the
   resulting masks are tight enough for segmentation supervision, or should only
   be used for detection-level evaluation, is unresolved and needs a look at the
   real files.
3. GlobalOSD-SAR is announced as public but no repository URL has been located.
   It is the largest and most globally distributed set found, so it is worth an
   email to the authors.
