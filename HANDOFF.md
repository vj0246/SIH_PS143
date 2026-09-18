# HANDOFF: SAR Slick Attribution Pipeline

Smart India Hackathon 2026. Marine oil spill detection and vessel attribution
from satellite radar.

**How to use this file.** Paste the whole thing into a fresh AI chat as the first
message, with a line on top saying "this is a project handoff, read it and then
help me continue from the Next Actions section". That single paste rebuilds the
full working context: the problem, the reasoning, what is built, what is not,
and what to do next. A human collaborator can read it top to bottom in about
fifteen minutes.

Everything referenced here travels in the zip.

---

## 1. What the problem actually asks for

The official statement wants an automated pipeline that does four things:
detect and characterise an oil spill from satellite imagery, trace the slick
backward to its origin point and time and forward to predict its spread, attribute
the spill to a vessel using historic AIS traffic around that origin window, and
present all of it in a visual interface.

Our reading of it, which shaped every decision:

The statement sounds like a detection problem. It is not. It is a **false
positive control problem wrapped in an inverse problem wrapped in a ranking
problem**. Finding dark patches on radar is easy. Knowing which dark patches are
oil is hard. Running drift backward through an ocean is an inverse problem with
growing error. Naming a ship from that is a ranking under uncertainty that can
never become proof.

## 2. Scope decision: what we detect

Source agnostic. We detect any oil film on water.

Synthetic aperture radar physically cannot separate bunker fuel from crude cargo.
Their surface signatures overlap almost completely. Anyone who claims otherwise
from radar alone is wrong, and a judge who knows the field will catch it.

Source class is inferred later, from vessel identity and slick volume, never from
the image. A container ship carries no oil cargo, so a slick attributed to one is
bunker fuel or bilge waste. A laden crude tanker could be either, and volume
separates them, because 200 cubic metres is not a bunker leak.

What the system will actually see, by frequency:

1. Operational discharge from ordinary ships. Bilge water, tank washings, sludge,
   engine room waste, bunker fuel. Most detections by count. Over 90 percent of
   global chronic ocean oiling is human caused and it concentrates along shipping
   lanes.
2. Cargo crude from tanker casualties. Rare, very large, high visibility.
3. Natural seeps, wrecks, platforms. Fixed in position, and the single biggest
   false positive category for vessel attribution.

## 3. The three constraints that shaped the architecture

**Look-alikes outnumber real slicks by roughly ten to one.** Biogenic surfactant
films from algal blooms, low wind cells, rain cells, upwelling, internal waves,
ship wakes and grease ice all damp radar backscatter and appear as dark patches.

**Detectors do not transfer between oceans.** A published model scoring 67.8 mean
intersection over union in the Mediterranean falls to 51.8 on Peruvian waters.
Sixteen points lost to geography alone, caused by different sea state, different
current regime producing thin fragmented slicks instead of compact ones, and
higher background texture complexity.

**Attribution can never be proof.** The only defensible output is a ranked list
of candidate vessels with calibrated confidence, plus an explicit
"origin unresolvable" state when the drift error is too large to narrow the
field.

## 4. Architecture in four stages

See `docs/architecture.svg` and `ARCHITECTURE.md` for the full picture.

**Stage 1, Analysis Ready Data. BUILT.**
Turn heterogeneous radar scenes and mismatched public datasets into one
consistent, honestly labelled training corpus. Calibration, multilooking,
geocoding, land masking, physics channel construction, wind retrieval, the
observability gate, label unification, deterministic tiling, manifest and blocked
splits.

**Stage 2, Detection and Verification. DESIGNED.**
Five class segmentation with oil and look-alike co-trained in one label space,
then object level verification using shape, damping, texture, shipping lane
proximity and archive recurrence.

**Stage 3, Characterisation and Drift. DESIGNED.**
Geometry metrics, thickness banding where a coincident optical scene exists, a
posterior over four behavioural oil classes, and a backward drift ensemble
producing an origin probability field over space and time.

**Stage 4, Attribution and Interface. DESIGNED.**
Query historic AIS over that spacetime volume, filter irrelevant traffic, forward
re-simulate a discharge from every surviving candidate track, score the overlap
against the slick actually observed, and present ranked suspects with an evidence
trail.

### The coupling that makes it more than a pipeline

Oil type inference and vessel attribution are **not sequential**. They are
coupled and should run as a joint probabilistic model.

Vessel type constrains oil type. Oil type constrains survival time and drift
behaviour. Drift constrains the origin field. The origin field constrains the
candidate set. Run it in the other direction too: slick volume constrains which
vessel classes are even plausible.

Most teams will build detect, then track, then match. This loop is the
differentiator and it is the thing to lead with when presenting.

## 5. What is built, precisely

Stage 1 only. About 5000 lines across 17 modules. 95 tests, all passing, none
requiring network access.

```
configs/classes.yaml       unified label schema and per source remapping
configs/datasets.yaml      source registry with licences and citations
src/oilspill/schema.py     class schema, remapping, loud failure on unknown values
src/oilspill/adapters/     one file per dataset, all yielding the same Sample
src/oilspill/ard/wind.py   CMOD5.N forward and inverse, observability gate
src/oilspill/ard/features.py   physics channels
src/oilspill/ard/tiling.py     deterministic content aware tiler
src/oilspill/io.py         sharded tile store, cross store addressing
src/oilspill/manifest.py   parquet manifest and integrity checks
src/oilspill/splits.py     blocked split assignment and leak verification
src/oilspill/build.py      adapter to features to tiles to manifest
src/oilspill/fetch.py      resumable checksum verified downloader
src/oilspill/cdse.py       Copernicus Data Space query and download
src/oilspill/cli.py        command line interface
tests/fixtures.py          synthetic datasets in each source's real on disk layout
```

Stages 2, 3 and 4 are specified in detail in `ARCHITECTURE.md` and not written.

## 6. The design decisions, and why

Full versions with rejected alternatives are in `decisions.md`. The five that
matter most:

**Five classes, not binary.** Sea, oil, look-alike, ship, land, plus ignore. Oil
and look-alike must be learned together in one label space. A binary segmenter
never sees the discriminative boundary, so it learns "dark equals oil" and fires
on every biogenic film. The strongest published global result attributes its
gains specifically to this co-training.

**The ignore class is load bearing, not a convenience.** Several public datasets
ship binary masks. Their zero class does not mean sea, it means "not annotated as
oil", and it provably contains land, ships and look-alikes. Merging that into the
sea class explicitly teaches the model that biogenic films are open water, which
destroys the exact capability the five class schema exists to build. Each source
therefore declares which classes it can express and abstains on the rest. Those
pixels are excluded from both loss and metrics.

This is the single most damaging silent bug available in this project. Nothing in
the output would look wrong.

**Splits are blocked, never random.** Neighbouring tiles from one scene share the
same slick, sea state, wind field, incidence angle ramp and speckle realisation.
Random splitting puts them on both sides of the boundary and inflates the score
without improving the model. We block by scene, by one degree geographic cell, or
by whole source. This deliberately produces a lower reported number than a
comparable paper using random splits.

**Wind is metadata and a gate, never a filter at ingest.** Below roughly 3 metres
per second the whole sea is dark and a slick has nothing to contrast against.
Above roughly 10 the wave breaking overwhelms the damping. Outside that band the
system emits `not_observable`, which is a different statement from "no oil
detected". Tiles are never dropped at ingest, for two reasons: dropping them
hides the decision from the evaluation, and low wind cells are themselves a
look-alike class the model must learn to reject.

**Physics channels are stacked alongside backscatter.** Incidence angle, wind
speed, damping ratio, local variance, distance to land. Slick contrast is not an
absolute quantity: the same film reads differently at 30 and at 45 degrees of
incidence and at 4 and at 9 metres per second of wind. A network given only
intensity must infer that relationship from a few thousand patches, which is a
large part of why cross basin transfer collapses. Every channel is selectable so
the ablation is still available.

## 7. Three bugs the tests caught

These are worth telling, because they show the test suite is real and they are
each a general lesson.

**CMOD5.N is not monotonic.** The wind retrieval inverts a geophysical model
function by bisection, which requires the function to be strictly increasing. It
is not. Sweeping the coefficient set over incidence angle and relative wind
direction showed backscatter turning over at 31.8 metres per second at the near
edge of the Sentinel-1 swath, and as low as 13 metres per second at 15 degrees of
incidence. Above the turnover, bisection converges to a plausible looking wrong
root and nothing announces it. Inversion is now capped at 30 metres per second
with the geometry validated, and two tests pin the turnover so a coefficient edit
cannot quietly widen the domain.

**Single look speckle biased the wind retrieval by 50 percent.** The first
background estimator took the 85th percentile of raw pixels. Radar intensity at
one look is exponentially distributed, so that percentile sits at 1.90 times the
mean, a fixed 2.8 decibel error. A scene generated at 7.0 metres per second came
back as 10.4, past the detection ceiling, so the pipeline labelled a perfectly
ordinary scene unobservable. The output looked smooth and plausible throughout.
Fixed by averaging in linear power into blocks first, then taking a percentile
across a neighbourhood of blocks.

General lesson worth carrying: every ratio, threshold and normaliser in this
project is computed on speckled data. Assume a statistic is biased until a test
says otherwise. Averaging must happen in linear power, never in decibels, because
the mean of the logarithm sits 2.5 decibels below the logarithm of the mean.

**Tile addresses collided across sources.** Shard filenames restart inside each
source's store, so the second source's first shard shadowed the first source's.
The manifest looked entirely healthy: unique identifiers, sensible class counts,
valid splits. Training would simply have read the wrong pixels with correct
labels attached, degrading results without ever producing an error.

## 8. Data sources and their real access status

**Imagery, all free.**
Copernicus Data Space Ecosystem for Sentinel-1 Interferometric Wide swath ground
range detected products, dual polarisation, 10 metre, 250 kilometre swath. This
is the workhorse. Alaska Satellite Facility for the Sentinel-1 archive and for
NISAR L band, which became public in July 2026 and covers acquisitions from June
2026. ISRO Bhoonidhi for RISAT and EOS-04 over the Indian Ocean.

**Labelled training data.**
The MKLab five class benchmark from m4d.iti.gr, behind a request form, and its
palette in our config is transcribed from the publication rather than confirmed
against real files, so it is flagged provisional and `oilspill verify` exists to
catch it being wrong. The Zenodo Sentinel-1 oil spill sets, parts one through
three, roughly 3600 scenes with dual polarisation backscatter in decibels but
binary labels only. The PANGAEA Eastern Mediterranean set, 3225 oil objects
across 1365 patches plus 2290 no oil patches clustered into twelve water and five
coastal look-alike types, which is the richest public look-alike taxonomy we
found.

**The prize, not yet obtained.**
GlobalOSD-SAR. Over 100,000 annotated oil slick and look-alike images mined from
more than 500,000 globally distributed Sentinel-1 scenes spanning 2014 to 2021.
Announced as public, but no repository URL has been located. Reported
intersection over union of 82.63 to 95.99 using an ordinary UNet++ with oil and
look-alike co-training plus an iterative snowballing loop.

**Action item: email the corresponding authors.** That one dataset is worth more
than everything else on this list combined, and it is the difference between a
regional demo and a globally credible model.

**Environmental forcing.**
Copernicus Marine global analysis and forecast for currents at one twelfth of a
degree, the wave product for Stokes drift, ERA5 for 10 metre winds, and the NOAA
ADIOS oil library for weathering properties. INCOIS regional products are higher
resolution over the Indian Ocean and are the right upgrade for the Indian
sub-domain.

**AIS, the awkward one.**
Global Fishing Watch offers free research access. The Danish Maritime Authority
publishes full raw AIS, which is Denmark only but is the best dataset on earth
for building and testing attribution logic. MarineCadastre gives free historic US
data. Indian authoritative AIS sits with the Directorate General of Shipping and
the Coast Guard and is not publicly available.

Practical consequence: run the radar, drift and interface work on Indian areas of
interest, and validate attribution quantitatively where full raw AIS exists. State
that split openly rather than hiding it.

## 9. Running the code

```bash
pip install -e ".[dev]"
pytest -q                      # 95 tests, no network needed

oilspill sources               # what is declared and what each can supervise
oilspill disk --source zenodo_s1_part1      # check size before downloading
oilspill fetch --source zenodo_s1_part1 --root data/raw
oilspill verify --source mklab --root data/raw/mklab   # before first build
oilspill build --source zenodo_s1 --root data/raw/zenodo_s1 \
               --out data/ard --manifest data/ard/manifest.parquet
oilspill split --manifest data/ard/manifest.parquet --strategy scene
oilspill stats --manifest data/ard/manifest.parquet
oilspill search --bbox 72.0,18.0,73.5,19.5 \
                --start 2024-01-01T00:00:00.000Z --end 2024-01-31T00:00:00.000Z
```

Credentials for Copernicus come from `CDSE_USERNAME` and `CDSE_PASSWORD` in the
environment. Never written to disk, never committed.

One warning from experience: the sandbox this was authored in blocked every data
host at the network proxy, so nothing was ever downloaded during development. The
whole pipeline is verified against synthetic fixtures that reproduce each source's
exact on disk layout. Running the same tests against the real archives is the
acceptance check, and it has not happened yet. Expect the MKLab palette and the
PANGAEA annotation layout to need adjustment on first contact with real files.
Both fail loudly rather than silently, which is by design.

## 10. Honest assessment

**What is genuinely strong.**
The label schema and the ignore class. The blocked split policy. The physics
channels. The observability gate as a first class output. The decision log with
rejected alternatives. The evaluation protocol, especially the synthetic closed
loop for attribution. Together these are the difference between a project that
scores well on a benchmark and one that would survive contact with a new ocean.

**What is weak.**
No model exists yet. No real data has been through the pipeline. The drift and
attribution stages are specified but unwritten, and they are the parts with the
most unknown unknowns. The oil type posterior is the most speculative component
in the whole design and I would not defend it hard.

**What would worry me most.**
Building a beautiful segmenter that fires on every algal bloom in the Arabian
Sea. Every structural choice above exists to attack that one failure, but none of
them have been tested against real data yet.

Second worry: scope. The problem statement asks for four hard things. A team that
does all four badly loses to a team that does two of them credibly. If time gets
tight, cut the forward forecast and the oil type posterior before cutting the
look-alike rejection or the evaluation protocol.

**What I would do differently with hindsight.**
Chase GlobalOSD-SAR on day one rather than building adapters for three smaller
datasets first. Data access lead time is the longest pole and it is not
parallelisable by working harder.

## 11. Next actions, in order

1. Register free accounts on Copernicus Data Space and Copernicus Marine. Five
   minutes each, and everything downstream needs them.
2. Email the GlobalOSD-SAR corresponding authors asking for the dataset. Longest
   lead time, start immediately.
3. Submit the MKLab request form.
4. Download the Zenodo and PANGAEA sets and run
   `oilspill verify` then `oilspill build` on real files. Expect adapter
   adjustments. This is the acceptance test for Stage 1.
5. Build the evaluation harness before the model. Cross region hold out, false
   alarms per 1000 square kilometres at fixed recall, wind stratified curves.
   A model without this harness produces numbers nobody should trust.
6. Train the five class baseline. UNet++ or SegFormer, ImageNet initialisation,
   Dice plus focal loss with ignore index. Compare against two baselines: a
   classical adaptive threshold with morphological cleanup, and a plain U-Net on
   backscatter only. If the physics channels do not beat both, they are not
   earning their place.
7. Only then start Stage 3.

## 12. Never claim these

- That the damping ratio identifies oil type. It measures surface film damping.
- That radar gives slick thickness. Damping saturates. Thickness needs optical
  colour banding and is uncertain by a factor of several even then.
- That the system identifies the guilty vessel. It ranks candidate vessels with
  calibrated confidence. Chemical fingerprinting of oil against a vessel's tanks
  is what stands up legally. Our contribution is narrowing four hundred
  candidates to three.
- Any end to end runtime or accuracy figure. Neither has been measured.
- That NISAR L band will help. It is an open question, and that is exactly what
  makes it worth investigating.

## 13. Numbers worth knowing

| Value | What it is |
|---|---|
| 3 to 10 m/s | Wind band where radar oil detection is valid |
| about 5 cm | Bragg resonant wavelength, C band at 35 degrees |
| 67.8 to 51.8 | Mean intersection over union, Mediterranean model tested on Peru |
| 10 to 1 | Look-alikes versus real slicks in a real archive |
| over 90 percent | Share of chronic ocean oiling that is human caused |
| 29.1 to 46 degrees | Sentinel-1 Interferometric Wide incidence angle range |
| 31.8 m/s | Where CMOD5.N stops being monotonic |
| 2.8 dB | Speckle bias in a percentile estimator on single look data |
| 1 to 4.5 percent | Windage, as a fraction of wind speed |
| 12 to 24 hours | Useful backward drift horizon |
| under 1 percent | Oil share of pixels in a labelled corpus |
| about 1 GB | One Sentinel-1 ground range detected scene |

## 14. Reading order for the repo

1. This file.
2. `ARCHITECTURE.md` for the full four stage specification.
3. `docs/architecture.svg` open alongside it.
4. `decisions.md` for every design decision with its rejected alternative.
5. `README.md` for how to run things.
6. `src/oilspill/schema.py` and `configs/classes.yaml` together, because the
   label schema is the contract everything else obeys.
7. `tests/test_wind.py` to see how the CMOD5.N implementation was verified against
   an independent reference.

---

Companion materials outside the zip: an architecture web page and a question and
answer defence brief, both published as private links that can be shared from the
page's own share menu.
