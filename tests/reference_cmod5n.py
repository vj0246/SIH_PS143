"""Scalar reference implementation of CMOD5.N, used only to verify our own.

Transcribed from IFREMER's ``xsarsea`` (xsarsea/windspeed/gmfs_impl.py,
``gmf_cmod5_generic(neutral=True)``, Apache-2.0). It is deliberately kept in the
scalar, branch-per-pixel form of the published formulation so that it is an
independent check on the vectorised numpy version in
``oilspill.ard.wind.cmod5n_forward`` rather than a copy of it.

Not imported by the package. Test scaffolding only.
"""

from __future__ import annotations

import math

C = [
    0.0,
    -0.6878, -0.7957, 0.338, -0.1728, 0.0, 0.004, 0.1103, 0.0159,
    6.7329, 2.7713, -2.2885, 0.4971, -0.725, 0.045, 0.0066, 0.3222,
    0.012, 22.7, 2.0813, 3.0, 8.3659, -3.3428, 1.3236, 6.2437,
    2.3893, 0.3249, 4.159, 1.693,
]


def cmod5n_forward_scalar(inc: float, wspd: float, phi: float) -> float:
    """VV sigma0 in linear units for one (incidence, wind speed, phi) triple."""
    zpow = 1.6
    thetm = 40.0
    thethr = 25.0
    y0 = C[19]
    pn = C[20]
    a = y0 - (y0 - 1.0) / pn
    b = 1.0 / (pn * (y0 - 1.0) ** (pn - 1.0))

    cosphi = math.cos(math.radians(phi))
    x = (inc - thetm) / thethr
    x2 = x**2.0

    a0 = C[1] + C[2] * x + C[3] * x2 + C[4] * x * x2
    a1 = C[5] + C[6] * x
    a2 = C[7] + C[8] * x
    gam = C[9] + C[10] * x + C[11] * x2
    s0 = C[12] + C[13] * x
    s = a2 * wspd
    a3 = 1.0 / (1.0 + math.exp(-s0))

    if s < s0:
        a3 = a3 * (s / s0) ** (s0 * (1.0 - a3))
    else:
        a3 = 1.0 / (1.0 + math.exp(-s))

    b0 = (a3**gam) * 10.0 ** (a0 + a1 * wspd)

    b1 = C[15] * wspd * (0.5 + x - math.tanh(4.0 * (x + C[16] + C[17] * wspd)))
    b1 = (C[14] * (1.0 + x) - b1) / (math.exp(0.34 * (wspd - C[18])) + 1.0)

    v0 = C[21] + C[22] * x + C[23] * x2
    d1 = C[24] + C[25] * x + C[26] * x2
    d2 = C[27] + C[28] * x
    v2 = wspd / v0 + 1.0
    if v2 < y0:
        v2 = a + b * (v2 - 1.0) ** pn

    b2 = (-d1 + d2 * v2) * math.exp(-v2)

    return b0 * (1.0 + b1 * cosphi + b2 * (2.0 * cosphi**2.0 - 1.0)) ** zpow
