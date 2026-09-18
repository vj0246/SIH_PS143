# Architecture

Automated detection of marine oil slicks from satellite SAR, backward drift to an
origin, and probabilistic attribution to a vessel using historic AIS.

Status: **Stage 1 is built and tested. Stages 2 to 4 are designed, not built.**

---

## The problem, stated precisely

Given a satellite radar image of the ocean, decide whether a dark patch is oil or
one of a dozen natural phenomena that look identical to a radar. If it is oil,
work out where it came from and when, then work out which ship put it there.

Three things make this hard, and the architecture is shaped by all three.

1. **Look-alikes outnumber slicks by roughly ten to one.** Biogenic films, low
   wind cells, rain cells, upwelling, internal waves and ship wakes all suppress
   radar backscatter. The hard problem is not finding dark patches. It is
   rejecting the ones that are not oil.
2. **Detectors do not transfer between oceans.** A model trained on the
   Mediterranean drops from 67.8 to 51.8 mIoU on Peruvian waters. Sea state,
   current regime, slick morphology and background texture all shift.
3. **Attribution is probabilistic and can never be proof.** The correct output is
   a ranked list of suspects with calibrated confidence, not a verdict.

## What is being detected

Source agnostic. The radar cannot distinguish bunker fuel from crude cargo, and
the pipeline does not pretend otherwise. Source class is inferred later, from
vessel identity and slick volume, not from the image.

By count, most global detections are operational discharge from ordinary vessels:
bilge water, tank washings, sludge, bunker fuel. Over 90 percent of chronic ocean
oiling is anthropogenic and it clusters along shipping lanes. Cargo crude spills
from tanker casualties are rare and large. Natural seeps, wrecks and platforms
are fixed in position and are the top false positive for vessel attribution,
which is why a static exclusion layer is part of the design.

---

## Layer stack

### Layer 0. Ingest

| input | source | role |
|-------|--------|------|
| Sentinel-1 IW GRDH, VV+VH | Copernicus Data Space | primary detection imagery, global |
| NISAR L-band | ASF DAAC | second frequency, damping differs from C-band |
| RISAT / EOS-04 | ISRO Bhoonidhi | Indian Ocean sub-domain |
| ERA5 10 m wind | Copernicus CDS | observability gating, drift forcing |
| CMEMS currents and Stokes drift | Copernicus Marine | drift forcing |
| GEBCO bathymetry, GSHHG coastline | public | context channels, land mask |
| Static exclusion layer | seeps, platforms, wrecks | false positive suppression |
| Historic AIS | GFW, DMA, MarineCadastre, commercial | attribution |

### Layer 1. Analysis ready data

Orbit file, thermal noise removal, radiometric calibration to sigma0, speckle
filtering, geocoding, land masking, conversion to dB, deterministic tiling.

Each tile then carries physical conditioning channels alongside VV and VH:
incidence angle, CMOD5.N wind speed, damping ratio against the local background,
local variance, distance to land.

**Why the extra channels.** Slick contrast is not absolute. The same film reads
differently at 30 and at 45 degrees of incidence, and at 4 and at 9 m/s of wind.
A network given only intensity has to infer that relationship from a few thousand
patches, which is a large part of why cross-basin transfer collapses. Supplying
the conditioning variables directly is cheap and is the intervention with the
best evidence behind it.

### Layer 2. Observability gate

Below roughly 3 m/s the sea is too smooth for a slick to contrast against.
Above roughly 10 m/s wave breaking overwhelms the Marangoni damping and the
slick signature disappears.

Outside that band the system emits `not_observable`, which is a different
statement from "no oil detected". Wind is recorded per tile as metadata and never
used to silently drop data at ingest, so the same corpus can answer both "how
good is detection" and "when is detection valid".

### Layer 3. Segmentation

Five classes: sea, oil, look-alike, ship, land, plus an ignore class for
supervision a source cannot provide.

Oil and look-alike are learned **together**, in one label space. A binary
segmenter never sees the discriminative boundary during training and learns
"dark equals oil". The largest reported global result attributes its gains to
exactly this co-training, plus an iterative loop that grows and cleans the
training set, on ordinary UNet++.

**The ignore class is load bearing.** Binary labelled datasets do not say "this
is sea", they say "this is not annotated as oil", and that remainder provably
contains land, ships and look-alikes. Merging it into sea teaches the model that
biogenic films are open water. Those pixels are excluded from loss and metrics.

### Layer 4. Object verification

Vectorise the segmentation, compute per object geometry, damping ratio, texture
and context features, then reject residual look-alikes with a classifier that can
see object level evidence a pixel classifier cannot: shape, elongation,
proximity to a shipping lane, and recurrence at the same coordinates across the
archive. A dark patch that appears at the same position on 30 percent of all
passes is a seep, not a ship.

Output: a slick object with a calibrated confidence.

### Layer 5. Characterisation

Geometry: area, perimeter, major and minor axis, orientation, eccentricity,
shape complexity. Linear high aspect ratio slicks indicate a moving discharge;
compact slicks indicate a point release or a wreck.

Thickness and volume: Bonn Agreement Oil Appearance Code banding, only where a
coincident optical scene exists. SAR gives presence, never thickness.

Behavioural class: a **posterior over four classes**, not a label. Non persistent
volatile, persistent light, persistent heavy emulsifying, non petroleum
surfactant. Driven by damping ratio, spreading rate, multi pass persistence,
volume, and a vessel type prior from AIS. Persistence alone is powerful: a slick
still present after 48 hours at 8 m/s cannot be marine gas oil.

Age: no sensor measures it. Inferred by running the drift model until predicted
geometry matches observed geometry.

### Layer 6. Drift

OpenDrift OpenOil, ensemble, forced by CMEMS currents, ERA5 winds and WAVERYS
Stokes drift, with oil properties from the NOAA ADIOS library.

Backward: N particles by M oil type hypotheses by K forcing perturbations,
producing an origin probability field over space and time rather than a point.

Forward: predicted slick evolution, for response planning and for the scoring
step below.

**Horizon.** Beyond roughly 12 to 24 hours, current and wind error broaden the
origin field until the AIS candidate set is useless. The correct output there is
`origin unresolvable`, not a ranked list of four hundred vessels.

### Layer 7. Attribution

1. Query historic AIS over the spacetime volume where the origin field carries
   mass.
2. Filter irrelevant traffic.
3. For each surviving candidate, **forward simulate a discharge from its own
   track** and score the overlap with the actually observed slick.
4. Rank by a combined score: origin field mass at the candidate, track alignment
   with the slick major axis, forward simulation overlap, vessel type and laden
   state prior, speed and course anomaly, AIS gap coincident with the origin
   window, distance off standard route.

Step 3 is what makes this defensible. Backward drift alone gives a search region.
Forward re-simulation from each candidate, scored against the real slick shape,
is a hypothesis test.

### Layer 8. Interface

Map with slick polygons, time slider, drift cones, ranked suspect list, and an
evidence panel that shows why each suspect scored as it did. Observability state
is displayed as a first class output, so an operator can tell "we looked and saw
nothing" apart from "we could not look".

---

## The coupling that matters

Oil type inference and vessel attribution are **not sequential**. They are
coupled, and the architecture runs them as a joint probabilistic model.

Vessel type constrains oil type: a container ship carries no oil cargo, so any
slick attributed to it is bunker or bilge, which fixes a viscosity prior, which
changes the drift, which changes the origin field, which changes the candidate
set. Volume works the other way: a 200 cubic metre slick is not a bunker leak,
which downweights every non tanker candidate.

Most teams will build detect, then track, then match. The coupling is the
technically interesting part of the problem and the main differentiator.

---

## Evaluation

The evaluation protocol is the differentiator, not the model.

- **Cross region hold out.** Train on one basin, test on another, and report the
  drop as the headline number. A single-region score is close to meaningless for
  global deployment.
- **False alarms per 1000 square kilometres at fixed recall.** The metric that
  actually decides whether an operator would use the system.
- **Wind stratified performance curves.** Detection quality as a function of the
  conditioning variable that governs it.
- **Synthetic closed loop for attribution.** Take a real AIS track, forward
  simulate a discharge, render it into a realistic scene, then run the full
  backward pipeline and check whether it recovers the true vessel at rank one.
  This gives unlimited labelled attribution data and produces a recall at rank K
  curve against elapsed time, traffic density and forcing error. No public
  dataset provides that, and no other team will have the curve.

Splits are blocked by scene, by geographic cell, or by whole source. Random tile
splits leak: neighbouring tiles share a slick, a sea state, a wind field, an
incidence ramp and a speckle realisation.

---

## What is deliberately not claimed

- The damping ratio measures surface film damping. It is not an oil type
  classifier.
- CMOD inverts backscatter to wind given a wind direction, which a single SAR
  image does not supply. The assumed direction travels with every retrieval.
- Wind inversion is capped at 30 m/s because CMOD5.N stops being monotonic above
  roughly 31.8 m/s at the near edge of the IW swath.
- Attribution produces ranked suspects with calibrated confidence. It never
  produces a verdict.

See `decisions.md` for every design decision with its rejected alternative, and
for the three bugs the test suite caught in Stage 1.
