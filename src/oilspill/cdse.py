"""Copernicus Data Space Ecosystem client: query and download real Sentinel-1.

This is the path to global, current data. Free registration at
dataspace.copernicus.eu; credentials come from the environment and are never
written to disk.

Two things about this API are worth knowing before using it:

1. Access tokens expire in minutes, while a GRD product is roughly 1 GB. The
   token is therefore refreshed lazily before every request rather than fetched
   once per session, otherwise long downloads die partway through.
2. The OData filter grammar is picky about quoting and about the geometry
   literal. :func:`build_filter` composes it rather than leaving callers to
   hand-write strings.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"


class CdseError(RuntimeError):
    pass


@dataclass
class CdseConfig:
    catalogue: str = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    download: str = "https://zipper.dataspace.copernicus.eu/odata/v1"
    client_id: str = "cdse-public"

    @classmethod
    def from_registry(cls, path: str | Path | None = None) -> "CdseConfig":
        with Path(path or _DEFAULT_REGISTRY).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh).get("cdse", {})
        return cls(
            catalogue=raw.get("catalogue", cls.catalogue),
            token_url=raw.get("token_url", cls.token_url),
            download=raw.get("download", cls.download),
            client_id=raw.get("client_id", cls.client_id),
        )


class CdseClient:
    """Minimal OData client for Sentinel-1 search and product download."""

    def __init__(
        self,
        config: CdseConfig | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.config = config or CdseConfig.from_registry()
        self._username = username or os.environ.get("CDSE_USERNAME")
        self._password = password or os.environ.get("CDSE_PASSWORD")
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "oilspill-ard/0.1"})

    # -- auth ---------------------------------------------------------------

    def _ensure_token(self) -> str:
        """Return a valid access token, refreshing with a safety margin.

        Search does not need a token; download does. Credentials are only
        required at the point they are used, so a query-only workflow runs
        without them.
        """
        if self._token and time.time() < self._token_expiry - 30:
            return self._token
        if not self._username or not self._password:
            raise CdseError(
                "CDSE credentials are required for download. Set CDSE_USERNAME and "
                "CDSE_PASSWORD in the environment. Register free at "
                "https://dataspace.copernicus.eu . Never commit them to this repository."
            )
        response = self._session.post(
            self.config.token_url,
            data={
                "grant_type": "password",
                "username": self._username,
                "password": self._password,
                "client_id": self.config.client_id,
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise CdseError(
                f"token request failed with HTTP {response.status_code}. "
                "Check the credentials, and note that CDSE rate limits repeated failures."
            )
        payload = response.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + float(payload.get("expires_in", 600))
        return self._token

    # -- search -------------------------------------------------------------

    @staticmethod
    def build_filter(
        collection: str = "SENTINEL-1",
        product_type: str = "GRD",
        sensor_mode: str = "IW",
        start: str | None = None,
        end: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        polarisation: str | None = "VV&VH",
    ) -> str:
        """Compose an OData ``$filter`` expression.

        Args:
            collection: Collection name.
            product_type: For example ``GRD`` or ``SLC``. Use GRD for detection;
                SLC is only needed if you intend to compute polarimetric
                features that require the complex data.
            sensor_mode: ``IW`` over coastal seas, ``EW`` over open ocean and
                polar regions.
            start: ISO 8601 UTC start, for example ``2024-01-01T00:00:00.000Z``.
            end: ISO 8601 UTC end.
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in degrees.
            polarisation: Attribute value, or None to leave unconstrained.

        Returns:
            The filter string.
        """
        clauses = [f"Collection/Name eq '{collection}'"]
        clauses.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        )
        clauses.append(
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'sensorMode' "
            f"and att/OData.CSC.StringAttribute/Value eq '{sensor_mode}')"
        )
        if polarisation:
            clauses.append(
                "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
                "'polarisationChannels' and att/OData.CSC.StringAttribute/Value eq "
                f"'{polarisation}')"
            )
        if start:
            clauses.append(f"ContentDate/Start gt {start}")
        if end:
            clauses.append(f"ContentDate/Start lt {end}")
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            ring = (
                f"{min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
                f"{min_lon} {max_lat},{min_lon} {min_lat}"
            )
            clauses.append(
                f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({ring}))')"
            )
        return " and ".join(clauses)

    def search(
        self,
        odata_filter: str,
        top: int = 100,
        order_by: str = "ContentDate/Start desc",
    ) -> list[dict]:
        """Run a catalogue query and return the product records.

        Follows ``@odata.nextLink`` so ``top`` is a total, not a page size.
        """
        url = f"{self.config.catalogue}/Products"
        params = {"$filter": odata_filter, "$top": min(top, 1000), "$orderby": order_by}
        results: list[dict] = []

        while url and len(results) < top:
            response = self._session.get(url, params=params, timeout=120)
            if response.status_code != 200:
                raise CdseError(
                    f"catalogue query failed with HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )
            payload = response.json()
            results.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
            params = None  # nextLink already carries the query string
        return results[:top]

    def search_area(
        self,
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        top: int = 100,
        **kwargs,
    ) -> list[dict]:
        """Convenience wrapper: everything over a bounding box in a date window."""
        return self.search(self.build_filter(start=start, end=end, bbox=bbox, **kwargs), top=top)

    # -- download -----------------------------------------------------------

    def download_product(
        self,
        product_id: str,
        dest: str | Path,
        chunk_bytes: int = 8 << 20,
        progress=None,
    ) -> Path:
        """Download one product zip.

        The token is refreshed immediately before the request and the transfer
        is streamed to a ``.part`` file, so a partial download never masquerades
        as a complete one.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")

        token = self._ensure_token()
        url = f"{self.config.download}/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {token}"}

        with self._session.get(url, headers=headers, stream=True, timeout=300) as resp:
            if resp.status_code != 200:
                raise CdseError(
                    f"download of {product_id} failed with HTTP {resp.status_code}"
                )
            length = resp.headers.get("Content-Length")
            total = int(length) if length else None
            done = 0
            with part.open("wb") as fh:
                for block in resp.iter_content(chunk_size=chunk_bytes):
                    if not block:
                        continue
                    fh.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)

        if total and part.stat().st_size != total:
            part.unlink(missing_ok=True)
            raise CdseError(
                f"{product_id}: transferred {done} bytes, expected {total}. Retry."
            )
        part.replace(dest)
        return dest


def summarise_products(products: list[dict]) -> list[dict]:
    """Reduce raw OData records to the fields that matter for triage."""
    out = []
    for p in products:
        attrs = {
            a.get("Name"): a.get("Value")
            for a in p.get("Attributes", [])
            if isinstance(a, dict)
        }
        out.append(
            {
                "id": p.get("Id"),
                "name": p.get("Name"),
                "start": p.get("ContentDate", {}).get("Start"),
                "size_gb": round(float(p.get("ContentLength", 0)) / 1e9, 3),
                "orbit_direction": attrs.get("orbitDirection"),
                "relative_orbit": attrs.get("relativeOrbitNumber"),
                "polarisation": attrs.get("polarisationChannels"),
                "online": p.get("Online"),
            }
        )
    return out
