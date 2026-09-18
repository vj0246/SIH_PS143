"""Generate the full pipeline architecture diagram as a standalone SVG.

Hand-placing two hundred SVG elements invites drift; this computes the layout
from one grid so lanes, bands and arrows stay aligned.

Usage:
    python scripts/make_architecture_svg.py out.svg
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Canvas and grid
# ---------------------------------------------------------------------------

W, H = 2480, 1620
MARGIN = 36
LANES = 4
LANE_GAP = 96
LANE_W = (W - 2 * MARGIN - (LANES - 1) * LANE_GAP) / LANES

BAND_HEADER = 118
BAND_SOURCE = 176
BAND_PROCESS = 470
BAND_ARTIFACT = 1258
BAND_EVAL = 1400
BAND_LEGEND = 1536

SANS = "DejaVu Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "DejaVu Sans Mono, IBM Plex Mono, Menlo, Consolas, monospace"

C = {
    "bg": "#f2f5f4",
    "panel": "#ffffff",
    "ink": "#15212b",
    "soft": "#4a5b64",
    "faint": "#7d8d94",
    "rule": "#c0cac8",
    "depth": "#1a6485",
    "depth_s": "#dbe9ef",
    "steel": "#2f4a57",
    "steel_s": "#dee5e8",
    "slick": "#8a6626",
    "slick_s": "#f0e5cd",
    "caution": "#a82a6e",
    "caution_s": "#f5dde9",
    "moss": "#2c7361",
    "moss_s": "#dbeae5",
    "source": "#eae5d7",
    "source_b": "#b9ac8c",
}

LANE_ACCENT = ["depth", "steel", "slick", "caution"]

out: list[str] = []


def lane_x(i: int) -> float:
    return MARGIN + i * (LANE_W + LANE_GAP)


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def text(
    x, y, s, size=13, fill="ink", font=SANS, weight="normal", anchor="start",
    spacing=0, opacity=None,
):
    attrs = [
        f'x="{x:.1f}"', f'y="{y:.1f}"',
        f'font-family="{font}"', f'font-size="{size}"',
        f'fill="{C.get(fill, fill)}"',
    ]
    if weight != "normal":
        attrs.append(f'font-weight="{weight}"')
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if spacing:
        attrs.append(f'letter-spacing="{spacing}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    out.append(f'<text {" ".join(attrs)}>{esc(s)}</text>')


def rect(x, y, w, h, fill="panel", stroke="rule", sw=1.2, dash=None, rx=0):
    attrs = [
        f'x="{x:.1f}"', f'y="{y:.1f}"', f'width="{w:.1f}"', f'height="{h:.1f}"',
        f'fill="{C.get(fill, fill)}"', f'stroke="{C.get(stroke, stroke)}"',
        f'stroke-width="{sw}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if rx:
        attrs.append(f'rx="{rx}"')
    out.append(f'<rect {" ".join(attrs)}/>')


def line(x1, y1, x2, y2, stroke="ink", sw=1.4, dash=None, marker=True):
    attrs = [
        f'x1="{x1:.1f}"', f'y1="{y1:.1f}"', f'x2="{x2:.1f}"', f'y2="{y2:.1f}"',
        f'stroke="{C.get(stroke, stroke)}"', f'stroke-width="{sw}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker:
        attrs.append(f'marker-end="url(#a-{stroke})"')
    out.append(f'<line {" ".join(attrs)}/>')


def path(d, stroke="ink", sw=1.6, dash=None, marker=True, fill="none"):
    attrs = [f'd="{d}"', f'stroke="{C.get(stroke, stroke)}"',
             f'stroke-width="{sw}"', f'fill="{fill}"']
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker:
        attrs.append(f'marker-end="url(#a-{stroke})"')
    out.append(f'<path {" ".join(attrs)}/>')


def diamond(cx, cy, w, h, fill="depth_s", stroke="depth", sw=1.8):
    pts = f"{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}"
    out.append(
        f'<polygon points="{pts}" fill="{C[fill]}" stroke="{C[stroke]}" '
        f'stroke-width="{sw}"/>'
    )


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------


@dataclass
class Block:
    title: str
    lines: list[str] = field(default_factory=list)
    kind: str = "process"          # process | store | source | emphasis | out
    note: str = ""
    height: float = 0.0
    y: float = 0.0


LINE_H = 15.5
TITLE_H = 21


def block_height(b: Block) -> float:
    h = 14 + TITLE_H + LINE_H * len(b.lines)
    if b.note:
        h += 17
    return h + 10


def draw_block(b: Block, x: float, w: float, accent: str) -> None:
    styles = {
        "process":  ("panel", "rule", 1.3),
        "emphasis": (accent + "_s", accent, 2.4),
        "store":    ("panel", accent, 1.6),
        "source":   ("source", "source_b", 1.2),
        "out":      ("caution_s", "caution", 1.8),
    }
    fill, stroke, sw = styles[b.kind]
    dash = "5 4" if b.kind == "store" else None
    rect(x, b.y, w, b.height, fill=fill, stroke=stroke, sw=sw, dash=dash)

    if b.kind == "process":
        out.append(
            f'<rect x="{x:.1f}" y="{b.y:.1f}" width="4" '
            f'height="{b.height:.1f}" fill="{C[accent]}"/>'
        )

    tx = x + (16 if b.kind == "process" else 12)
    ty = b.y + 24
    tcol = accent if b.kind in ("emphasis", "store", "out") else "ink"
    tsize = 14.5 if b.kind in ("emphasis", "out") else 13.5
    text(tx, ty, b.title, size=tsize, weight="bold", fill=tcol, font=SANS)

    yy = ty + 19
    for ln in b.lines:
        mono = ln.startswith("`")
        clean = ln.strip("`")
        text(
            tx, yy, clean,
            size=11.4 if mono else 11.8,
            fill="soft",
            font=MONO if mono else SANS,
        )
        yy += LINE_H

    if b.note:
        text(tx, yy + 5, b.note, size=11, fill=accent, font=SANS, weight="bold")


def stack(blocks: list[Block], x: float, w: float, y0: float, accent: str,
          gap: float = 26, connect: bool = True) -> float:
    y = y0
    for b in blocks:
        b.height = block_height(b)
        b.y = y
        y += b.height + gap
    for b in blocks:
        draw_block(b, x, w, accent)
    if connect:
        for a, b in zip(blocks, blocks[1:]):
            line(x + w / 2, a.y + a.height, x + w / 2, b.y - 4,
                 stroke=accent, sw=1.6)
    return y


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

LANE_TITLES = [
    ("STAGE 1", "Analysis Ready Data", "BUILT  ·  95 tests"),
    ("STAGE 2", "Detection & Verification", "DESIGNED"),
    ("STAGE 3", "Characterisation & Drift", "DESIGNED"),
    ("STAGE 4", "Attribution & Interface", "DESIGNED"),
]

SOURCES = [
    [
        "Sentinel-1 IW GRDH  ·  CDSE OData/STAC  ·  VV+VH  ·  10 m  ·  250 km swath",
        "NISAR L-band  ·  ASF DAAC  ·  public since Jul 2026",
        "RISAT-1A / EOS-04  ·  ISRO Bhoonidhi  ·  Indian Ocean sub-domain",
        "ERA5 10 m wind  ·  Copernicus CDS      GEBCO bathy  ·  GSHHG coast",
    ],
    [
        "MKLab 5-class benchmark  ·  request form  ·  palette provisional",
        "Zenodo S1 Parts I–III  ·  3600 scenes  ·  binary labels only",
        "PANGAEA E. Med  ·  3225 oil objects  ·  12 look-alike clusters",
        "GlobalOSD-SAR  ·  100k+ global patches  ·  no repo URL, email authors",
        "Static exclusion layer  ·  natural seeps, platforms, known wrecks",
    ],
    [
        "CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024  ·  1/12° currents",
        "CMEMS WAV_001_027  ·  Stokes drift        ERA5  ·  10 m wind",
        "NOAA ADIOS oil library  ·  ~1000 characterised oils",
        "Sentinel-2 / 3, Landsat 8-9  ·  opportunistic optical, cloud-free only",
    ],
    [
        "Historic AIS  ·  Global Fishing Watch API  ·  free research access",
        "Danish Maritime Authority open AIS  ·  full raw, best for validation",
        "MarineCadastre US  ·  Spire / Kpler commercial for Indian Ocean",
        "Vessel registry  ·  type, deadweight, laden draught, prior offences",
    ],
]

L1 = [
    Block("Ingest & calibrate", [
        "Apply precise orbit file",
        "Thermal noise removal (VH sits near NESZ)",
        "Radiometric calibration → sigma-nought",
    ]),
    Block("Multilook & geocode", [
        "Boxcar average in LINEAR power, never in dB",
        "Speckle filter · ellipsoid/terrain correct · land mask",
        "Convert to dB only after averaging",
    ], note="mean of log(intensity) is 2.5 dB below log(mean)"),
    Block("CMOD5.N wind retrieval", [
        "Block means → percentile across block neighbourhood",
        "Bisection inverse, 40 iterations, deterministic",
        "phi assumed and recorded, never silently dropped",
    ], note="capped at 30 m/s: GMF turns over at 31.8 m/s"),
    Block("Observability gate", [
        "3 m/s floor: glassy sea, everything is dark",
        "10 m/s ceiling: wave breaking swamps damping",
        "outside → not_observable, which is not a negative",
    ], kind="emphasis"),
    Block("Feature stack builder", [
        "`vv_db   vh_db   incidence_deg   wind_ms`",
        "`damping_db   local_var_db   dist_land_km`",
        "every channel selectable, so the ablation survives",
    ]),
    Block("Label unification", [
        "5 classes + ignore(255) per source capability",
        "binary sources abstain, they do not assert sea",
        "undeclared value → hard error, never a default",
    ], note="D2: the ignore class is load bearing"),
    Block("Deterministic tiler & splits", [
        "256 px, 32 overlap, last tile flush to edge",
        "negatives kept at 0.15 by seeded blake2b hash",
        "splits blocked by scene / 1° cell / whole source",
    ]),
]

L2 = [
    Block("Five-class segmenter", [
        "UNet++ or SegFormer, ImageNet or SAR-SSL init",
        "input: 7-channel tile, not bare intensity",
        "oil and look-alike CO-TRAINED in one label space",
    ], kind="emphasis",
        note="binary framing never sees the discriminative boundary"),
    Block("Loss & sampling", [
        "Dice + focal, class weighted",
        "`ignore_index = 255` excludes unsupervised pixels",
        "oil is under 1 percent of pixels, sample accordingly",
    ]),
    Block("Vectorise → object features", [
        "area, perimeter, major/minor axis, orientation",
        "eccentricity, solidity, fractal dimension",
        "damping ratio, GLCM texture, distance to lane",
    ]),
    Block("Look-alike rejector", [
        "Gradient boosted trees on object features",
        "sees context a pixel classifier cannot",
        "linear high-aspect ⇒ moving discharge",
    ]),
    Block("Archive recurrence test", [
        "same coordinates on >30 percent of passes",
        "⇒ seep, platform or wreck, not a ship",
        "top false positive for vessel attribution",
    ], kind="emphasis"),
    Block("Confidence threshold", [
        "calibrated per wind bin and per region",
        "below tau → logged, not surfaced",
    ]),
]

L3 = [
    Block("Geometry metrics", [
        "area, axes, orientation, eccentricity",
        "shape complexity, perimeter-to-area ratio",
        "compact ⇒ point release or wreck",
    ]),
    Block("Thickness & volume", [
        "Bonn Agreement Oil Appearance Code banding",
        "sheen 0.04–0.3 µm … continuous colour >200 µm",
        "OPTICAL ONLY · SAR gives presence, not thickness",
    ], note="uncertain by a factor of several, say so"),
    Block("Behavioural class posterior", [
        "non-persistent volatile | persistent light",
        "persistent heavy emulsifying | non-petroleum",
        "from damping, spreading, persistence, volume",
    ], kind="emphasis", note="a posterior over 4 classes, never a label"),
    Block("OpenOil backward ensemble", [
        "N particles × M oil hypotheses × K forcing perturbations",
        "currents + windage(1–4.5%) + Stokes drift",
        "weathering from the ADIOS oil library",
    ]),
    Block("Horizon check", [
        "beyond 12–24 h the origin field is too broad",
        "→ origin_unresolvable, not 400 ranked ships",
    ], kind="emphasis"),
    Block("Age inference", [
        "no sensor measures slick age",
        "run forward until predicted geometry matches",
    ]),
]

L4 = [
    Block("Spacetime AIS query", [
        "select tracks intersecting the origin field support",
        "window set by the oil type's survival time",
    ]),
    Block("Traffic filter", [
        "drop tracks inconsistent with the origin mass",
        "drop fixed infrastructure and known seeps",
        "keep AIS gaps: absence is itself evidence",
    ]),
    Block("Forward re-simulation per candidate", [
        "simulate a discharge from each candidate's own track",
        "score IoU against the slick actually observed",
        "turns a lookup into a hypothesis test",
    ], kind="emphasis", note="this is what makes attribution defensible"),
    Block("Score fusion", [
        "overlap · origin mass · track alignment",
        "vessel type & laden prior · speed/course anomaly",
        "AIS gap at origin window · route deviation",
    ]),
    Block("Calibration", [
        "isotonic fit on the synthetic closed loop",
        "so a 0.8 score means 0.8 of the time",
    ]),
    Block("Operator interface", [
        "map, time slider, drift cones, evidence panel",
        "observability shown as a first-class state",
    ]),
]

ARTIFACTS = [
    ("ARD tile store + manifest", [
        "`npz shards, C×256×256 float32`",
        "`manifest.parquet · 1 row per tile`",
        "geometry, class counts, wind stats, split",
    ]),
    ("Slick objects", [
        "`GeoJSON polygons + confidence`",
        "geometry, damping and texture features",
        "look-alike probability retained",
    ]),
    ("Origin field + type posterior", [
        "`P(x, y, t) NetCDF probability volume`",
        "`4-class behavioural posterior`",
        "forward forecast for response planning",
    ]),
    ("Ranked suspects", [
        "`MMSI, score, confidence interval`",
        "`evidence trail per scoring term`",
        "never a verdict, always a ranking",
    ]),
]

EVAL = [
    ("Cross-region hold-out",
     "Train one basin, test another. The published drop is 67.8 to 51.8 mIoU. "
     "Report that number, never the in-domain one."),
    ("False alarms per 1000 km²",
     "At fixed recall. The number that decides whether an operator would "
     "actually run the system on a live feed."),
    ("Wind-stratified curves",
     "Precision and recall against the physical variable that governs "
     "detectability, not pooled across all conditions."),
    ("Synthetic closed loop",
     "Real AIS track → simulated discharge → rendered scene → full backward "
     "pipeline. Gives recall@rank-K that no public dataset provides."),
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render() -> str:
    out.clear()
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{SANS}">'
    )
    out.append("<defs>")
    for key in ("ink", "depth", "steel", "slick", "caution", "moss", "faint"):
        out.append(
            f'<marker id="a-{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{C[key]}"/></marker>'
        )
    out.append("</defs>")
    rect(0, 0, W, H, fill="bg", stroke="bg", sw=0)

    # ---- masthead ----
    text(MARGIN, 52, "SAR Slick Attribution Pipeline", size=32, weight="bold")
    text(MARGIN, 78,
         "Detect oil on the ocean from satellite radar · drift it backward to an origin "
         "· rank the vessels that could have put it there",
         size=14.5, fill="soft")
    text(W - MARGIN, 48, "SOURCE AGNOSTIC BY DESIGN", size=11.5, fill="caution",
         weight="bold", anchor="end", spacing=1.6)
    text(W - MARGIN, 68,
         "radar cannot separate bunker fuel from crude cargo",
         size=11.8, fill="soft", anchor="end")
    text(W - MARGIN, 86,
         "source class is inferred from vessel identity and volume, never from the image",
         size=11.8, fill="soft", anchor="end")
    line(MARGIN, 96, W - MARGIN, 96, stroke="ink", sw=2, marker=False)

    # ---- band labels ----
    for label, yy in (
        ("EXTERNAL DATA", BAND_SOURCE - 14),
        ("PROCESSING", BAND_PROCESS - 14),
        ("ARTIFACTS", BAND_ARTIFACT - 14),
        ("EVALUATION", BAND_EVAL - 14),
    ):
        text(MARGIN, yy, label, size=10.5, fill="faint", weight="bold",
             spacing=2.2, font=MONO)

    # ---- lanes ----
    lane_blocks = [L1, L2, L3, L4]
    for i, (stage, name, status) in enumerate(LANE_TITLES):
        x = lane_x(i)
        accent = LANE_ACCENT[i]

        rect(x, BAND_HEADER, LANE_W, 44, fill=accent, stroke=accent, sw=1)
        text(x + 14, BAND_HEADER + 20, stage, size=11, fill="#ffffff",
             weight="bold", font=MONO, spacing=1.8)
        text(x + 14, BAND_HEADER + 36, name, size=15, fill="#ffffff",
             weight="bold")
        text(x + LANE_W - 14, BAND_HEADER + 29, status, size=10.5,
             fill="#ffffff", font=MONO, anchor="end", spacing=1.2, opacity=0.88)

        # sources
        sy = BAND_SOURCE
        for s in SOURCES[i]:
            rect(x, sy, LANE_W, 26, fill="source", stroke="source_b", sw=1)
            text(x + 11, sy + 17.5, s, size=11.2, fill="#4b4436", font=MONO)
            sy += 30
        line(x + LANE_W / 2, sy + 2, x + LANE_W / 2, BAND_PROCESS - 6,
             stroke="source_b", sw=1.5, dash="4 4")

        # process
        stack(lane_blocks[i], x, LANE_W, BAND_PROCESS, accent)

        # artifact
        title, lines = ARTIFACTS[i]
        ah = 18 + TITLE_H + LINE_H * len(lines)
        rect(x, BAND_ARTIFACT, LANE_W, ah, fill="panel", stroke=accent,
             sw=1.8, dash="6 4")
        text(x + 14, BAND_ARTIFACT + 25, title, size=13.5, weight="bold",
             fill=accent)
        yy = BAND_ARTIFACT + 45
        for ln in lines:
            mono = ln.startswith("`")
            text(x + 14, yy, ln.strip("`"), size=11.4 if mono else 11.8,
                 fill="soft", font=MONO if mono else SANS)
            yy += LINE_H

        # last process block down into artifact
        last = lane_blocks[i][-1]
        line(x + LANE_W / 2, last.y + last.height,
             x + LANE_W / 2, BAND_ARTIFACT - 4, stroke=accent, sw=1.8)

    # ---- lane to lane hand-offs ----
    handoffs = [
        (0, ["ARD tiles", "+ manifest"]),
        (1, ["verified", "slick objects"]),
        (2, ["origin field", "+ posterior"]),
    ]
    for i, label in handoffs:
        x1 = lane_x(i) + LANE_W
        x2 = lane_x(i + 1)
        xm, y = (x1 + x2) / 2, BAND_ARTIFACT + 34
        line(x1 + 4, y, x2 - 6, y, stroke="ink", sw=2.4)
        for j, ln in enumerate(label):
            text(xm, y + 20 + j * 14, ln, size=10.2, fill="ink",
                 font=MONO, anchor="middle", weight="bold")

    # ---- the coupling loop: orthogonal, routed inside the lane gap ----
    gx = lane_x(3) - LANE_GAP / 2
    y_from = L4[3].y + L4[3].height / 2
    y_to = L3[2].y + L3[2].height / 2
    path(f"M {lane_x(3):.0f} {y_from:.0f} L {gx:.0f} {y_from:.0f} "
         f"L {gx:.0f} {y_to:.0f} L {lane_x(2) + LANE_W + 4:.0f} {y_to:.0f}",
         stroke="caution", sw=3)
    out.append(f'<circle cx="{gx:.0f}" cy="{y_from:.0f}" r="4.5" '
               f'fill="{C["caution"]}"/>')
    xm = lane_x(3) - LANE_GAP / 2
    text(xm, BAND_PROCESS - 74,
         "vessel type constrains oil type constrains drift constrains the candidate set",
         size=14, fill="caution", weight="bold", anchor="middle")
    text(xm, BAND_PROCESS - 55,
         "this loop is the architecture — not detect, then track, then match",
         size=12, fill="caution", anchor="middle")
    line(xm, BAND_PROCESS - 48, xm, y_from - 8, stroke="caution",
         sw=1.4, dash="4 3", marker=False)

    # ---- snowball loop: verified detections re-enter the corpus ----
    xs = lane_x(1) + LANE_W
    gs = xs + LANE_GAP / 2
    y_rec = L2[4].y + L2[4].height / 2
    y_src = BAND_SOURCE + 3 * 30 + 13
    path(f"M {xs:.0f} {y_rec:.0f} L {gs:.0f} {y_rec:.0f} "
         f"L {gs:.0f} {y_src:.0f} L {xs + 4:.0f} {y_src:.0f}",
         stroke="steel", sw=2, dash="6 4")
    text(xs - 12, BAND_PROCESS - 74, "snowball loop", size=12.5,
         fill="steel", weight="bold", anchor="end")
    text(xs - 12, BAND_PROCESS - 56,
         "verified detections re-enter the training corpus",
         size=11.5, fill="soft", anchor="end")
    line(xs - 4, BAND_PROCESS - 66, gs, BAND_PROCESS - 66, stroke="steel",
         sw=1.2, dash="4 3", marker=False)

    # ---- not_observable side exit, inside the gate block ----
    gate = L1[3]
    xg = lane_x(0)
    cw, cx = 182, xg + LANE_W - 196
    rect(cx, gate.y + 26, cw, 46, fill="caution_s", stroke="caution", sw=1.6)
    text(cx + 12, gate.y + 46, "not_observable", size=12,
         fill="caution", font=MONO, weight="bold")
    text(cx + 12, gate.y + 62, "an output, not a negative", size=9.8,
         fill="soft", font=MONO)
    line(cx - 16, gate.y + 49, cx - 2, gate.y + 49, stroke="caution", sw=1.6)

    # ---- evaluation strip ----
    ew = (W - 2 * MARGIN - 3 * 14) / 4
    for i, (title, body) in enumerate(EVAL):
        x = MARGIN + i * (ew + 14)
        rect(x, BAND_EVAL, ew, 104, fill="moss_s", stroke="moss", sw=1.5)
        text(x + 14, BAND_EVAL + 25, title, size=13.5, weight="bold",
             fill="moss")
        words, lines, cur = body.split(), [], ""
        for word in words:
            if len(cur) + len(word) + 1 > 58:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        lines.append(cur)
        yy = BAND_EVAL + 45
        for ln in lines[:4]:
            text(x + 14, yy, ln, size=11.5, fill="soft")
            yy += 15.5

    # ---- legend ----
    ly = BAND_LEGEND
    line(MARGIN, ly - 18, W - MARGIN, ly - 18, stroke="rule", sw=1,
         marker=False)
    lx = MARGIN

    def key(w, draw, label):
        nonlocal lx
        draw(lx)
        text(lx + w + 9, ly + 15, label, size=11.5, fill="soft")
        lx += w + 18 + len(label) * 6.1

    key(30, lambda x: (rect(x, ly + 3, 30, 16, fill="panel", stroke="rule"),
                       out.append(f'<rect x="{x}" y="{ly + 3}" width="3.5" '
                                  f'height="16" fill="{C["depth"]}"/>')),
        "process step")
    key(30, lambda x: rect(x, ly + 3, 30, 16, fill="depth_s", stroke="depth",
                           sw=2.2), "decisive mechanism")
    key(30, lambda x: rect(x, ly + 3, 30, 16, fill="panel", stroke="depth",
                           sw=1.8, dash="5 4"), "data artifact")
    key(30, lambda x: rect(x, ly + 3, 30, 16, fill="source",
                           stroke="source_b"), "external source")
    key(30, lambda x: rect(x, ly + 3, 30, 16, fill="moss_s", stroke="moss",
                           sw=1.5), "evaluation")
    key(34, lambda x: line(x, ly + 11, x + 34, ly + 11, stroke="ink", sw=2.2),
        "data flow")
    key(34, lambda x: line(x, ly + 11, x + 34, ly + 11, stroke="caution",
                           sw=3), "feedback: joint inference")
    key(34, lambda x: line(x, ly + 11, x + 34, ly + 11, stroke="steel", sw=2,
                           dash="6 4"), "label bootstrap")

    text(MARGIN, ly + 56,
         "Deliberately not claimed:  damping ratio is not an oil-type classifier  "
         "·  CMOD needs a wind direction a single image cannot supply  "
         "·  attribution is a calibrated ranking, never a verdict",
         size=11.8, fill="faint")
    text(W - MARGIN, ly + 56,
         "Rev. 1  ·  Stage 1 built and tested  ·  see decisions.md",
         size=11.5, fill="faint", font=MONO, anchor="end")

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "architecture.svg"
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(render())
    print(f"wrote {dest}")
