# =============================================================================
#  Clear Claims Co. — Hail Verification Report
#  CORE LOGIC  (this is the "engine" of the notebook)
# =============================================================================
#
#  Plain-English summary of what lives in this file:
#    * Turn a street address into a latitude / longitude  (U.S. Census geocoder)
#    * Work out the correct block of time to look at, in UTC, for the local
#      "date of loss"                                            (time windowing)
#    * Download the matching NOAA MRMS MESH radar files from AWS         (S3 fetch)
#    * Read those GRIB2 radar files and pull out the hail-size grid    (GRIB parse)
#    * Look up the hail size AT the property, and the WORST hail size within a
#      small radius of it                                             (sampling)
#    * Draw the footprint map and build the branded PDF        (map + report)
#
#  Every value the report shows is read straight from NOAA's public radar data.
#  Nothing here invents, simulates, or hard-codes a hail result. If the data is
#  missing, the functions raise a clear, friendly error instead of guessing.
# =============================================================================

from __future__ import annotations

import os
import re
import gc
import gzip
import math
import shutil
import tempfile
import datetime as dt
from dataclasses import dataclass, field

import numpy as np


# -----------------------------------------------------------------------------
#  CONSTANTS  (facts about the data source — not things an operator changes)
# -----------------------------------------------------------------------------

# The AWS public MRMS archive (bucket s3://noaa-mrms-pds) became reliable with
# the MRMS v12 upgrade on 2020-10-14. Before that, we fall back to the Iowa
# Environmental Mesonet (IEM) MRMS archive, which reaches back to ~2014.
ARCHIVE_START = dt.date(2020, 10, 14)        # AWS Open Data MRMS
IEM_ARCHIVE_START = dt.date(2014, 7, 1)      # IEM MRMS archive (approximate floor)
IEM_BASE = "https://mtarchive.geol.iastate.edu"

# The exact public bucket + product folder we read.
S3_BUCKET = "noaa-mrms-pds"
MESH_PRODUCT = "MESH_Max_1440min_00.50"          # 24-hour MAXIMUM hail size, 0.50 km layer
S3_PRODUCT_PREFIX = f"{S3_BUCKET}/CONUS/{MESH_PRODUCT}"

# MRMS stores hail size in MILLIMETRES. Inches = mm / 25.4 .
MM_PER_INCH = 25.4

# The filename timestamp looks like:  ...00.50_20240603-050043.grib2.gz
_TS_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2(?:\.gz)?$")


# =============================================================================
#  1.  DATE VALIDATION + UTC TIME WINDOW
# =============================================================================

def validate_date_of_loss(date_of_loss: dt.date, today: dt.date | None = None) -> None:
    """Raise a friendly error if the date can't be served by the AWS archive.

    * Too early  -> before the 2020-10-14 archive start.
    * In future  -> obviously no radar exists yet.
    """
    today = today or dt.date.today()
    if date_of_loss < IEM_ARCHIVE_START:
        raise ValueError(
            f"Date of loss {date_of_loss:%Y-%m-%d} is before the available radar "
            f"archives (which reach back to about {IEM_ARCHIVE_START:%Y-%m-%d}). "
            f"This date can't be verified from radar."
        )
    if date_of_loss > today:
        raise ValueError(
            f"Date of loss {date_of_loss:%Y-%m-%d} is in the future — there is no "
            f"radar data for it yet."
        )


def local_day_utc_window(date_of_loss: dt.date, lat: float, lon: float):
    """Return (utc_start, utc_end, tz_name) covering the FULL local calendar day.

    Why this matters: MRMS files are timestamped in UTC, but a homeowner's
    "date of loss" is a *local* calendar day. A storm at 11 PM local in Texas is
    already the next day in UTC. We find the property's local time zone from its
    coordinates, then convert local-midnight-to-local-midnight into UTC so we
    capture the whole local day no matter the offset.
    """
    # timezonefinder works fully offline (no network needed).
    from timezonefinder import TimezoneFinder
    try:
        from zoneinfo import ZoneInfo            # Python 3.9+ standard library
    except ImportError:                          # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore

    tz_name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
    if tz_name is None:
        # Far offshore or unknown — fall back to UTC so we still produce a window.
        tz_name = "UTC"
    tz = ZoneInfo(tz_name)

    local_start = dt.datetime(date_of_loss.year, date_of_loss.month, date_of_loss.day,
                              0, 0, 0, tzinfo=tz)
    local_end = local_start + dt.timedelta(days=1)

    utc_start = local_start.astimezone(dt.timezone.utc)
    utc_end = local_end.astimezone(dt.timezone.utc)
    return utc_start, utc_end, tz_name


# =============================================================================
#  2.  GEOCODING  (address -> latitude / longitude)
# =============================================================================

def geocode_census(address: str, timeout: int = 30):
    """Geocode with the U.S. Census geocoder (authoritative US address ranges).

    Returns (lat, lon, matched_address, 'U.S. Census') or None. The Census file
    is excellent but lags reality — newer/rural addresses are often missing.
    """
    import requests
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    matches = resp.json().get("result", {}).get("addressMatches", [])
    if not matches:
        return None
    best = matches[0]
    c = best["coordinates"]
    return float(c["y"]), float(c["x"]), best.get("matchedAddress", address), "U.S. Census"


def geocode_nominatim(address: str, timeout: int = 30):
    """Geocode with OpenStreetMap Nominatim (broad coverage, incl. new homes).

    Returns (lat, lon, matched_address, 'OpenStreetMap') or None. Nominatim's
    usage policy requires a descriptive User-Agent and ≤1 request/second — both
    fine for this notebook's low volume.
    """
    import requests
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1, "addressdetails": 0}
    headers = {"User-Agent": "ClearClaimsHailReport/1.0 (ops@clearclaimsco.co)"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        return None
    top = data[0]
    return float(top["lat"]), float(top["lon"]), top.get("display_name", address), "OpenStreetMap"


def geocode_address(address: str, timeout: int = 30):
    """Try multiple geocoders in order of authority, return the first hit.

    1. U.S. Census  — official US address ranges.
    2. OpenStreetMap Nominatim — fills the gaps (new construction, rural, etc.).

    Returns (lat, lon, matched_address, provider) or None if all miss. A network
    error from one provider doesn't stop the others.
    """
    for fn in (geocode_census, geocode_nominatim):
        try:
            r = fn(address, timeout=timeout)
        except Exception as exc:
            print(f"[geocode] {fn.__name__} unavailable ({exc}); trying next…")
            r = None
        if r:
            return r
    return None


def resolve_location(address: str | None,
                     manual_lat: float | None,
                     manual_lon: float | None):
    """Decide the final lat/lon to use.

    Priority:
      1. If a manual lat/lon override is provided, use it (and trust it).
      2. Otherwise geocode the address.
      3. If geocoding finds nothing, raise a friendly error telling the operator
         to use the manual override.

    Returns dict: {lat, lon, label, source}
    """
    if manual_lat is not None and manual_lon is not None:
        return {"lat": float(manual_lat), "lon": float(manual_lon),
                "label": address or f"{manual_lat:.5f}, {manual_lon:.5f}",
                "source": "manual lat/long override"}

    if not address or not address.strip():
        raise ValueError("No address was provided and no manual lat/long override was set.")

    result = geocode_address(address)
    if result is None:
        raise ValueError(
            f"The address could not be matched by either geocoder (U.S. Census or "
            f"OpenStreetMap):\n    {address!r}\n"
            f"Tips: include city, state and ZIP; try the street spelled out (e.g. "
            f"'Road' not 'Rd'); or set MANUAL_LAT / MANUAL_LON in the Settings cell "
            f"to enter coordinates by hand (right-click the spot in Google Maps to copy "
            f"them — remember West longitude is negative)."
        )
    lat, lon, matched, provider = result
    return {"lat": lat, "lon": lon, "label": matched,
            "source": f"{provider} geocoder"}


# =============================================================================
#  3.  FIND + DOWNLOAD THE RIGHT MRMS FILES ON AWS S3
# =============================================================================

def _parse_ts_from_key(key: str):
    """Pull the UTC timestamp out of an MRMS filename. Returns datetime or None."""
    m = _TS_RE.search(key)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc)


def open_s3():
    """Open an ANONYMOUS connection to the public bucket (no AWS account needed)."""
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def _list_day_keys(fs, day: dt.date):
    """List every MESH file key in one UTC day folder (empty list if folder absent)."""
    folder = f"{S3_PRODUCT_PREFIX}/{day:%Y%m%d}"
    try:
        return [k for k in fs.ls(folder) if k.endswith(".grib2.gz")]
    except FileNotFoundError:
        return []


def select_files_for_window(fs, utc_start: dt.datetime, utc_end: dt.datetime,
                            max_files: int = 5, tail_buffer_hours: int = 3):
    """Choose which MESH files to read for the local day.

    Key idea: MESH_Max_1440min is a *running 24-hour maximum*. The file
    timestamped at the END of the local day (in UTC) already contains the
    largest hail over that entire day. We therefore prefer files at/just after
    `utc_end`, and take a few of them (cell-wise max later) for robustness.

    Fallbacks keep it working at the edges of the archive (e.g. 'today', or a
    day whose tail spills past now): if nothing exists after utc_end, we take the
    latest available files at or before utc_end instead.

    Returns a list of S3 keys (1..max_files of them), or [] if the day is empty.
    """
    # Candidate keys can live in the utc_start day folder and the utc_end day folder.
    candidate_days = sorted({utc_start.date(), utc_end.date(),
                             (utc_end + dt.timedelta(hours=tail_buffer_hours)).date()})
    keys = []
    for d in candidate_days:
        keys.extend(_list_day_keys(fs, d))
    if not keys:
        return []

    stamped = [(k, _parse_ts_from_key(k)) for k in keys]
    stamped = [(k, t) for k, t in stamped if t is not None]
    stamped.sort(key=lambda kt: kt[1])

    window_hi = utc_end + dt.timedelta(hours=tail_buffer_hours)
    # Files whose 24h-max window ends just after the local day ends:
    after = [kt for kt in stamped if utc_end <= kt[1] <= window_hi]

    if after:
        chosen = after
    else:
        # Nothing after the day ended (e.g. an in-progress 'today'): use the
        # latest files that fall within the local day itself.
        within = [kt for kt in stamped if utc_start <= kt[1] <= utc_end]
        chosen = within[-max_files:] if within else stamped[-max_files:]

    # Thin to at most `max_files`, evenly spaced (adjacent running-max files are
    # nearly identical, so a handful is plenty and keeps the run fast).
    if len(chosen) > max_files:
        idx = np.linspace(0, len(chosen) - 1, max_files).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return [k for k, _ in chosen]


def download_and_gunzip(fs, key: str, tmpdir: str) -> str:
    """Download one gzipped GRIB2 from S3 and decompress to a local .grib2 file.

    Returns the local path to the decompressed file.
    """
    base = os.path.basename(key)
    gz_path = os.path.join(tmpdir, base)
    grib_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path + ".grib2"
    fs.get(key, gz_path)
    with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return grib_path


def download_url_gunzip(url: str, tmpdir: str, timeout: int = 60) -> str:
    """Download a gzipped GRIB2 over HTTPS (used for the IEM archive) and unzip it."""
    import requests
    base = os.path.basename(url.split("?")[0])
    gz_path = os.path.join(tmpdir, base)
    grib_path = gz_path[:-3] if gz_path.endswith(".gz") else gz_path + ".grib2"
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    with gzip.open(gz_path, "rb") as fin, open(grib_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return grib_path


def parse_iem_listing(html: str, folder_url: str):
    """Pull MESH_Max_1440min .grib2.gz file URLs out of an IEM directory listing."""
    names = re.findall(r'href="(MESH_Max_1440min_00\.50_\d{8}-\d{6}\.grib2\.gz)"', html)
    base = folder_url if folder_url.endswith("/") else folder_url + "/"
    return [base + n for n in dict.fromkeys(names)]   # de-dup, keep order


def select_iem_files_for_window(utc_start, utc_end, max_files=4, tail_buffer_hours=3):
    """Find MRMS MESH files on the IEM archive covering the local day (full-day max).

    IEM mirrors NCEP MRMS at:
      {IEM_BASE}/YYYY/MM/DD/mrms/ncep/MESH_Max_1440min/MESH_Max_1440min_00.50_YYYYMMDD-HHMMSS.grib2.gz
    """
    import requests
    candidate_days = sorted({utc_start.date(), utc_end.date(),
                             (utc_end + dt.timedelta(hours=tail_buffer_hours)).date()})
    stamped = []
    for d in candidate_days:
        folder = f"{IEM_BASE}/{d:%Y/%m/%d}/mrms/ncep/MESH_Max_1440min/"
        try:
            html = requests.get(folder, timeout=30).text
        except Exception:
            continue
        for url in parse_iem_listing(html, folder):
            t = _parse_ts_from_key(url)
            if t is not None:
                stamped.append((url, t))
    if not stamped:
        return []
    stamped.sort(key=lambda kt: kt[1])
    window_hi = utc_end + dt.timedelta(hours=tail_buffer_hours)
    after = [kt for kt in stamped if utc_end <= kt[1] <= window_hi]
    chosen = after or [kt for kt in stamped if utc_start <= kt[1] <= utc_end][-max_files:] or stamped[-max_files:]
    if len(chosen) > max_files:
        idx = np.linspace(0, len(chosen) - 1, max_files).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx))]
    return [u for u, _ in chosen]


def fetch_mesh_paths(utc_start, utc_end, date_of_loss, tmpdir, max_files=5):
    """Get local GRIB2 paths for the day, trying AWS first then the IEM archive.

    Returns (paths, source_label, keys). Either source can be empty; the caller
    raises a friendly 'no data' error if paths is empty.
    """
    paths, source, keys = [], None, []

    # 1) AWS Open Data MRMS (2020-10-14 → present)
    if date_of_loss >= ARCHIVE_START:
        try:
            fs = open_s3()
            keys = select_files_for_window(fs, utc_start, utc_end, max_files)
            paths = [download_and_gunzip(fs, k, tmpdir) for k in keys]
            if paths:
                source = "NOAA MRMS — AWS Open Data (s3://noaa-mrms-pds)"
        except Exception:
            paths = []

    # 2) IEM MRMS archive fallback (≈2014 → present; covers older dates / AWS gaps)
    if not paths:
        try:
            urls = select_iem_files_for_window(utc_start, utc_end, max_files)
            paths = [download_url_gunzip(u, tmpdir) for u in urls]
            if paths:
                source = "NOAA MRMS — Iowa Environmental Mesonet archive"
                keys = urls
        except Exception:
            paths = []

    return paths, source, keys


# =============================================================================
#  4.  READ THE GRIB2 FILE  (pull out the hail-size grid)
# =============================================================================

def read_mesh_grib(path: str):
    """Read one MRMS MESH GRIB2 file into plain numpy arrays.

    Returns (lats_1d, lons_1d, mesh_mm_2d) where:
      * lats_1d  : 1-D latitudes  (north -> south, the MRMS order)
      * lons_1d  : 1-D longitudes converted to the -180..180 range
      * mesh_mm_2d: 2-D hail size in MILLIMETRES, shape (len(lats), len(lons))

    MRMS MESH often loads with the data variable named 'unknown' and longitudes
    in the 0..360 range — both are handled here.
    """
    import xarray as xr

    # indexpath='' stops cfgrib writing a sidecar .idx file (read-only dirs).
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})

    # Pick the hail-size variable: prefer 'unknown', else the first data var.
    var = "unknown" if "unknown" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]

    lats = np.asarray(ds["latitude"].values, dtype="float64")
    lons = np.asarray(ds["longitude"].values, dtype="float64")
    # float32 for the big grid keeps memory low (precise enough for mm values).
    mesh = np.asarray(da.values, dtype="float32")

    # Convert 0..360 longitudes to -180..180 so they match geocoder output.
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    # Some MRMS grids use a large negative/sentinel for "no coverage"; clamp to 0.
    mesh = np.where(np.isfinite(mesh), mesh, np.nan)
    mesh = np.where((mesh < 0) | (mesh > 1000), np.nan, mesh)

    ds.close()
    return lats, lons, mesh


def crop_to_bbox(lats, lons, mesh, lat, lon, pad_deg=0.30):
    """Crop the (possibly CONUS-wide) grid to a small box around the point.

    This keeps memory tiny when we max across several files. Returns cropped
    (lats, lons, mesh). pad_deg ~0.30 deg ≈ 20 miles, comfortably more than the
    sampling radius.
    """
    lat_mask = (lats >= lat - pad_deg) & (lats <= lat + pad_deg)
    lon_mask = (lons >= lon - pad_deg) & (lons <= lon + pad_deg)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            "The property location is outside this radar grid's coverage area. "
            "Check the coordinates / address."
        )
    li = np.where(lat_mask)[0]
    lj = np.where(lon_mask)[0]
    return (lats[li], lons[lj], mesh[np.ix_(li, lj)])


def max_mesh_over_files(grib_paths, lat, lon, pad_deg=0.30):
    """Read several GRIB2 files and return the CELL-WISE MAXIMUM over all of them,
    already cropped to the small box around the point.

    Processing one file at a time (and cropping immediately) keeps memory low.
    Returns (lats, lons, mesh_mm) for the cropped box.
    """
    acc_lats = acc_lons = acc = None
    for p in grib_paths:
        la, lo, me = read_mesh_grib(p)
        cla, clo, cme = crop_to_bbox(la, lo, me, lat, lon, pad_deg)
        del la, lo, me            # free the full CONUS grid immediately
        if acc is None:
            acc_lats, acc_lons, acc = cla, clo, cme
        else:
            acc = np.fmax(acc, cme)   # fmax ignores NaNs sensibly
            del cme
        gc.collect()              # keep peak memory low (free-tier friendly)
    if acc is None:
        raise ValueError("No radar files could be read for this date.")
    return acc_lats, acc_lons, acc


# =============================================================================
#  5.  SAMPLE THE GRID  (value at property + worst value within a radius)
# =============================================================================

def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/lon points (vectorised)."""
    R = 3958.7613  # Earth radius in miles
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def sample_point_and_radius(lats, lons, mesh_mm, lat, lon, radius_miles):
    """Return both readings the report needs:

      * point  : MESH at the single grid cell nearest the property
      * radius : the MAXIMUM MESH within `radius_miles` of the property
                 (accounts for ~1 km grid resolution + geocoding wobble)

    Returns a dict with mm and inch values plus where the radius-max occurred.
    """
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)

    # --- nearest cell (the property point) ---
    flat_idx = np.nanargmin(dist)
    pi, pj = np.unravel_index(flat_idx, dist.shape)
    point_mm = float(np.nan_to_num(mesh_mm[pi, pj], nan=0.0))

    # --- maximum within the radius ---
    in_radius = dist <= radius_miles
    if not in_radius.any():
        # Radius smaller than one grid cell: fall back to the nearest cell.
        in_radius = np.zeros_like(dist, dtype=bool)
        in_radius[pi, pj] = True

    vals = np.where(in_radius, mesh_mm, np.nan)
    if np.all(np.isnan(vals)):
        max_mm, mlat, mlon, ncells = 0.0, lat, lon, int(in_radius.sum())
    else:
        midx = np.nanargmax(vals)
        mi, mj = np.unravel_index(midx, vals.shape)
        max_mm = float(np.nan_to_num(mesh_mm[mi, mj], nan=0.0))
        mlat, mlon = float(lats[mi]), float(lons[mj])
        ncells = int(in_radius.sum())

    return {
        "point_mm": point_mm,
        "point_in": point_mm / MM_PER_INCH,
        "point_cell_lat": float(lats[pi]),
        "point_cell_lon": float(lons[pj]),
        "radius_max_mm": max_mm,
        "radius_max_in": max_mm / MM_PER_INCH,
        "radius_max_lat": mlat,
        "radius_max_lon": mlon,
        "radius_cells": ncells,
        "radius_miles": radius_miles,
    }


def sample_rings(lats, lons, mesh_mm, lat, lon, rings=(1, 3, 5)):
    """Value at the property (nearest cell) plus the MAX within each ring radius.

    This fills the report's four-row table: 'At Property', 'Within 1 mile',
    'Within 3 miles', 'Within 5 miles'. Returns:
        {'point': {'in','mm'}, 1: {'in','mm','lat','lon'}, 3:{...}, 5:{...}}
    """
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)

    pi, pj = np.unravel_index(np.nanargmin(dist), dist.shape)
    point_mm = float(np.nan_to_num(mesh_mm[pi, pj], nan=0.0))
    out = {"point": {"mm": point_mm, "in": point_mm / MM_PER_INCH,
                     "lat": float(lats[pi]), "lon": float(lons[pj])}}

    for r in rings:
        mask = dist <= r
        if not mask.any():
            mask[pi, pj] = True
        vals = np.where(mask, mesh_mm, np.nan)
        if np.all(np.isnan(vals)):
            out[r] = {"mm": 0.0, "in": 0.0, "lat": lat, "lon": lon}
        else:
            mi, mj = np.unravel_index(np.nanargmax(vals), vals.shape)
            mm = float(np.nan_to_num(mesh_mm[mi, mj], nan=0.0))
            out[r] = {"mm": mm, "in": mm / MM_PER_INCH,
                      "lat": float(lats[mi]), "lon": float(lons[mj])}
    return out


def decide_detection(sample: dict, threshold_in: float):
    """Hail 'detected' if the worst reading within the radius meets the threshold.

    Using the radius-max (not just the point) is the forensically conservative
    choice: it reflects the largest hail the radar saw essentially at the property.
    """
    worst_in = max(sample["radius_max_in"], sample["point_in"])
    return bool(worst_in >= threshold_in)


# =============================================================================
#  5b.  GROUND-TRUTH CORROBORATION  (independent observed hail reports)
# =============================================================================
#  Radar MESH is an estimate. Cross-checking it against actual storm reports
#  logged by NWS spotters / the public near the property on that date makes the
#  verification far stronger. Two free sources:
#    * NWS Local Storm Reports via the Iowa Environmental Mesonet (IEM) API
#    * SPC daily hail reports CSV
#  Both are best-effort: a network/format hiccup just yields zero reports.
# =============================================================================

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass_bearing(lat1, lon1, lat2, lon2) -> str:
    """16-point compass direction FROM point 1 TO point 2 (e.g. 'NE')."""
    import math
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
    brng = (math.degrees(math.atan2(y, x)) + 360) % 360
    return _COMPASS[int((brng + 11.25) % 360 / 22.5)]


def parse_iem_lsr_geojson(obj: dict, lat: float, lon: float, radius_miles: float):
    """Extract hail reports from an IEM LSR GeoJSON object, within radius_miles."""
    out = []
    for feat in (obj or {}).get("features", []):
        p = feat.get("properties", {}) or {}
        typ = (p.get("type") or "").upper()
        typetext = (p.get("typetext") or "").upper()
        if typ != "H" and "HAIL" not in typetext:
            continue
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None]
        rlon, rlat = coords[0], coords[1]
        if rlat is None or rlon is None:
            continue
        try:
            size_in = float(p.get("magnitude"))
        except (TypeError, ValueError):
            size_in = None
        d = float(haversine_miles(lat, lon, rlat, rlon))
        if d <= radius_miles:
            out.append({
                "source": "NWS LSR", "size_in": size_in,
                "lat": float(rlat), "lon": float(rlon), "dist_mi": d,
                "dir": compass_bearing(lat, lon, rlat, rlon),
                "time": p.get("valid", ""), "city": p.get("city", ""),
            })
    return out


def parse_spc_hail_csv(text: str, lat: float, lon: float, radius_miles: float):
    """Extract hail reports from an SPC daily hail CSV, within radius_miles.

    SPC columns: Time, Size (hundredths of an inch), Location, County, State,
    Lat, Lon, Comments.
    """
    import csv
    import io
    out = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            rlat = float(row.get("Lat")); rlon = float(row.get("Lon"))
            size_in = float(row.get("Size")) / 100.0
        except (TypeError, ValueError):
            continue
        d = float(haversine_miles(lat, lon, rlat, rlon))
        if d <= radius_miles:
            out.append({
                "source": "SPC", "size_in": size_in,
                "lat": rlat, "lon": rlon, "dist_mi": d,
                "dir": compass_bearing(lat, lon, rlat, rlon),
                "time": row.get("Time", ""), "city": row.get("Location", ""),
            })
    return out


def fetch_storm_reports(lat, lon, utc_start, utc_end, date_of_loss, radius_miles=12.0):
    """Best-effort: gather nearby observed hail reports from IEM + SPC. Never raises."""
    import requests
    reports = []

    # IEM Local Storm Reports (GeoJSON), filtered to a bbox + the local-day window.
    try:
        pad = 0.6  # ~40 miles, comfortably larger than the search radius
        params = {
            "sts": utc_start.strftime("%Y-%m-%dT%H:%MZ"),
            "ets": (utc_end + dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ"),
            "west": lon - pad, "east": lon + pad,
            "south": lat - pad, "north": lat + pad,
        }
        r = requests.get("https://mesonet.agron.iastate.edu/geojson/lsr.geojson",
                         params=params, timeout=30)
        reports += parse_iem_lsr_geojson(r.json(), lat, lon, radius_miles)
    except Exception:
        pass

    # SPC daily hail CSV (UTC convective day ~ matches our date of loss).
    try:
        url = f"https://www.spc.noaa.gov/climo/reports/{date_of_loss:%y%m%d}_rpts_hail.csv"
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and "Lat" in r.text[:200]:
            reports += parse_spc_hail_csv(r.text, lat, lon, radius_miles)
    except Exception:
        pass

    # De-duplicate near-identical reports (same rounded spot + size).
    seen, deduped = set(), []
    for rep in sorted(reports, key=lambda x: x["dist_mi"]):
        key = (round(rep["lat"], 2), round(rep["lon"], 2),
               round(rep["size_in"] or 0, 2))
        if key not in seen:
            seen.add(key); deduped.append(rep)
    return deduped


def assess_confidence(point_in, ring_max_in, reports, threshold_in, source=None):
    """Combine radar + ground reports into a stated confidence level.

    Heuristic (clearly labelled as such on the report). Returns
    {level, color, note}.
    """
    radar_max = max(point_in, ring_max_in)
    detected = point_in >= threshold_in
    n = len(reports)
    biggest = max([r["size_in"] for r in reports if r["size_in"]], default=0.0)

    if detected:
        if n >= 1:
            level = "High"
            note = (f"Radar-estimated hail is corroborated by {n} independent ground "
                    f"report(s) within the search area"
                    + (f" (largest {biggest:.2f}″)." if biggest else "."))
        elif radar_max >= threshold_in + 0.50:
            level = "Moderate"
            note = ("Radar estimate is well above the threshold, but no independent "
                    "ground report was logged nearby. Storm reports are sparse, so "
                    "this does not contradict the radar.")
        else:
            level = "Moderate"
            note = ("Radar estimate is near the threshold with no nearby ground report; "
                    "treat as a borderline result.")
    else:
        if n >= 1:
            level = "Low"
            note = (f"Radar did not meet the threshold at the property, yet {n} hail "
                    f"report(s) were logged nearby — verify exact timing and location.")
        else:
            level = "High"
            note = ("Radar shows no significant hail at the property and no ground "
                    "reports were logged nearby on this date.")

    color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[level]
    return {"level": level, "color": color, "note": note, "n_reports": n}


def corroboration_line(reports, radius_miles) -> str:
    """One-line human summary of the nearest ground reports (for the report)."""
    if not reports:
        return (f"No independent ground reports within {radius_miles:.0f} miles on this "
                f"date (radar-only estimate).")
    parts = []
    for r in reports[:3]:
        size = f"{r['size_in']:.2f}″" if r["size_in"] else "hail"
        parts.append(f"{size} — {r['dist_mi']:.1f} mi {r['dir']} ({r['source']})")
    extra = f" +{len(reports) - 3} more" if len(reports) > 3 else ""
    return "Nearby reports: " + "; ".join(parts) + extra + "."


# =============================================================================
#  6.  FOOTPRINT MAP  (matplotlib)
# =============================================================================

def make_footprint_map(lats, lons, mesh_mm, lat, lon, radius_miles, out_png,
                       brand=None, title="Estimated Hail Footprint"):
    """Draw the MESH field around the property with a marker + distance rings.

    Saves a PNG to `out_png`. Uses only matplotlib so it always works in Colab
    (no cartopy/contextily dependency required).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    from matplotlib.patches import Circle

    brand = brand or {}
    mesh_in = mesh_mm / MM_PER_INCH

    # Brand-flavoured size ramp: green (small) -> amber -> coral (giant).
    cmap = LinearSegmentedColormap.from_list(
        "hail", ["#28a678", "#7cc36a", "#e6a117", "#e07a2e", "#d94f3d"]).copy()
    cmap.set_bad(alpha=0.0)          # no-hail cells render fully transparent
    cmap.set_over("#a8332a")         # > top level
    levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    norm = BoundaryNorm(levels, cmap.N, extend="max")

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=160)
    ax.set_facecolor("#eef3f8")      # light map backdrop where there's no hail
    # Hide trivial/zero values so only real hail is coloured (not a green wash).
    mesh_plot = np.where((mesh_in < 0.25) | ~np.isfinite(mesh_in), np.nan, mesh_in)
    pcm = ax.pcolormesh(lons, lats, mesh_plot, cmap=cmap, norm=norm, shading="auto")

    # Property marker.
    ax.plot(lon, lat, marker="o", markersize=9, markerfacecolor="#06101f",
            markeredgecolor="white", markeredgewidth=1.6, zorder=6)

    # Distance rings at 1 / 3 / 5 miles (converted to degrees, lat-corrected).
    for r in (1, 3, 5):
        dlat = r / 69.0
        dlon = r / (69.0 * max(math.cos(math.radians(lat)), 1e-6))
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((lon, lat), 2 * dlon, 2 * dlat, fill=False,
                             edgecolor="#06101f", linestyle=":", linewidth=0.9,
                             alpha=0.55, zorder=5))
        ax.text(lon, lat + dlat, f"{r} mi", fontsize=7, color="#06101f",
                ha="center", va="bottom", alpha=0.7, zorder=5)

    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10, color="#06101f")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, extend="both")
    cbar.set_label("Estimated hail size (inches)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


# =============================================================================
#  7.  FONTS + PDF REPORT  (reportlab)
# =============================================================================

# Brand colours (also exported for the map / notebook).
BRAND = {
    "midnight": "#06101f",
    "slate":    "#0c1a30",
    "accent":   "#2b7de9",
    "bright":   "#4a9af5",
    "green":    "#28a678",
    "coral":    "#d94f3d",
    "ice":      "#b8cce0",
    "name":     "Clear Claims Co.",
    "tagline":  "Fairness in Every Claim",
    "contact":  "clearclaimsco.co",
}


def register_fonts(font_dir: str | None):
    """Register DM Serif Display (headings) + Outfit (body) with reportlab.

    Returns a dict naming the heading / body / body-bold fonts to use. If the
    TTFs aren't present or registration fails, falls back to Helvetica so the
    PDF still builds.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = {"head": "Helvetica-Bold", "body": "Helvetica", "body_bold": "Helvetica-Bold"}
    if not font_dir or not os.path.isdir(font_dir):
        return fonts

    def _try(name, filename):
        # Register one font file; return True only if it actually loads.
        # Exceptions (e.g. an unsupported variable font on older reportlab) are
        # caught here so one bad font never blocks the others.
        path = os.path.join(font_dir, filename)
        if not os.path.isfile(path):
            return False
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return True
        except Exception as exc:                   # pragma: no cover
            print(f"[fonts] could not load {filename} ({exc}); skipping it.")
            return False

    try:
        if _try("DMSerif", "DMSerifDisplay-Regular.ttf"):
            fonts["head"] = "DMSerif"
        # Body: accept either a static weight or the variable font (saved as
        # Outfit-Regular.ttf). Variable fonts register fine at their default weight.
        if _try("Outfit", "Outfit-Regular.ttf") or _try("Outfit", "Outfit[wght].ttf"):
            fonts["body"] = "Outfit"
        # Bold body: use a real bold/semibold weight if present, otherwise reuse
        # the regular Outfit so the typeface stays consistent (never Helvetica).
        if _try("Outfit-Bold", "Outfit-SemiBold.ttf") or _try("Outfit-Bold", "Outfit-Bold.ttf"):
            fonts["body_bold"] = "Outfit-Bold"
        elif fonts["body"] == "Outfit":
            fonts["body_bold"] = "Outfit"
    except Exception as exc:                       # pragma: no cover
        print(f"[fonts] registration failed ({exc}); using Helvetica fallback.")
        return {"head": "Helvetica-Bold", "body": "Helvetica", "body_bold": "Helvetica-Bold"}
    return fonts


def png_to_data_uri(png_path: str) -> str:
    """Turn a PNG file into a data: URI so it can be embedded straight into HTML."""
    import base64
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return "data:image/png;base64," + b64


# =============================================================================
#  7b.  EXACT-FORMAT REPORT  — the "Classic Forensic" HTML template + WeasyPrint
# =============================================================================
#  This reproduces the approved ClearClaims template pixel-for-pixel by building
#  the very same HTML/CSS and rendering it to PDF with WeasyPrint (no browser
#  needed). The matplotlib footprint map is embedded as a data: URI.
# =============================================================================

# Theme colours for the two states (mirrors the template's JS logic exactly).
_THEME_DETECTED = dict(main="#d94f3d", dark="#c0392b", deep="#a23a2c",
                       tint="#fdecea", tintBorder="#f3c4bd", rowTint="#fdf1ef")
_THEME_CLEAR = dict(main="#28a678", dark="#1b8a5f", deep="#176b4a",
                    tint="#e8f6f0", tintBorder="#bfe6d5", rowTint="#edf8f3")

# The shield logo (light version, for the dark header) — exact path data from brand.
_LOGO_SVG = (
    '<svg width="40" height="46" viewBox="0 0 200 230" fill="none">'
    '<path d="M100 10 L180 48 L180 52 C180 100, 175 140, 100 212 C25 140, 20 100, 20 52 L20 48 Z" fill="#f0f4f8"/>'
    '<path d="M100 55 L56 92 L66 92 L66 150 L134 150 L134 92 L144 92 Z" fill="#06101f"/>'
    '<rect x="124" y="64" width="13" height="30" rx="1.5" fill="#f0f4f8"/>'
    '<rect x="122" y="60" width="17" height="7" rx="1.5" fill="#f0f4f8"/>'
    '<rect x="127" y="68" width="7" height="22" rx="1" fill="#06101f"/>'
    '<rect x="86" y="104" width="28" height="28" rx="2" fill="#f0f4f8"/>'
    '<line x1="100" y1="104" x2="100" y2="132" stroke="#06101f" stroke-width="3"/>'
    '<line x1="86" y1="118" x2="114" y2="118" stroke="#06101f" stroke-width="3"/></svg>')

_ICON_TRIANGLE = ('<svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="#fff" '
                  'stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">'
                  '<path d="M10.3 3.9 2.4 18a1.9 1.9 0 0 0 1.7 2.9h15.8a1.9 1.9 0 0 0 1.7-2.9L13.7 3.9a1.9 1.9 0 0 0-3.4 0Z"/>'
                  '<path d="M12 9v4.5"/><path d="M12 17v.01"/></svg>')
_ICON_CHECK = ('<svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="#fff" '
               'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M20 6.5 9.2 17.3 4 12.1"/></svg>')


def _font_face_css(font_dir: str | None) -> str:
    """Build @font-face rules pointing at locally-downloaded brand TTFs.

    This makes the PDF use DM Serif Display + Outfit even if Google Fonts is
    slow/blocked. If the files aren't present, returns '' and the <link> to
    Google Fonts (also in the HTML) is relied upon instead.
    """
    if not font_dir or not os.path.isdir(font_dir):
        return ""
    import pathlib
    css = []
    dm = os.path.join(font_dir, "DMSerifDisplay-Regular.ttf")
    if os.path.isfile(dm):
        css.append("@font-face{font-family:'DM Serif Display';font-style:normal;"
                   f"font-weight:400;src:url('{pathlib.Path(dm).as_uri()}');}}")
    outfit = None
    for cand in ("Outfit-Regular.ttf", "Outfit[wght].ttf"):
        p = os.path.join(font_dir, cand)
        if os.path.isfile(p):
            outfit = p
            break
    if outfit:
        css.append("@font-face{font-family:'Outfit';font-style:normal;"
                   f"font-weight:300 700;src:url('{pathlib.Path(outfit).as_uri()}');}}")
    return "\n".join(css)


def build_report_html(data: dict, font_dir: str | None = None) -> str:
    """Build the complete, self-contained HTML for one report.

    `data` keys (all pre-formatted strings unless noted):
      reportId, dateGenerated, dateOfLoss, propertyAddress, claimRef,
      coordinates, contactUrl, contactCity, bandLabel, reportTitle,
      detected (bool), results {atProperty/mile1/mile3/mile5: {in,mm}},
      mapDataUri (str or ''), mapCaption, methodologyText, disclaimerText
    `font_dir` (optional): folder with the brand TTFs to embed via @font-face.
    """
    font_face = _font_face_css(font_dir)
    detected = bool(data["detected"])
    t = _THEME_DETECTED if detected else _THEME_CLEAR
    icon = _ICON_TRIANGLE if detected else _ICON_CHECK
    status_text = "Detected" if detected else "Not Detected"
    finding_verb = "was detected" if detected else "was not detected"
    threshold_word = "above" if detected else "below"
    threshold_label = f'{float(data.get("thresholdInches", 1.00)):.2f}″'

    res = data["results"]
    ap, m1, m3, m5 = res["atProperty"], res["mile1"], res["mile3"], res["mile5"]
    max_value = f'{ap["in"]}″ ({ap["mm"]} mm)'

    # Map area: real image if present, else the styled placeholder (template default).
    if data.get("mapDataUri"):
        map_block = (
            f'<div style="height:158px; background-color:#dbe7f2; '
            f'background-image:url(\'{data["mapDataUri"]}\'); background-size:cover; '
            f'background-position:center; background-repeat:no-repeat;"></div>')
    else:
        map_block = (
            '<div style="position:relative; height:158px; background:#dbe7f2; overflow:hidden;">'
            '<div style="position:absolute; inset:0; background:radial-gradient(circle at 50% 52%, '
            'rgba(217,79,61,.78) 0%, rgba(230,161,23,.62) 26%, rgba(40,166,120,.42) 50%, rgba(219,231,242,0) 72%);"></div>'
            '</div>')

    methodology = data.get("methodologyText",
        "Hail-size estimates are derived from NOAA's Multi-Radar Multi-Sensor (MRMS) "
        "Maximum Estimated Size of Hail (MESH) product — a radar algorithm that models "
        "in-storm hail growth from reflectivity and freezing-level data. Reported values "
        "represent the maximum estimated hail diameter within each radius during the "
        "storm's passage over the property on the date of loss.")
    disclaimer = data.get("disclaimerText",
        "This is a radar-derived estimate, not a guarantee of hail size or property damage, "
        "and is not a substitute for a physical inspection by a qualified professional. "
        "Clear Claims Co. makes no warranty and accepts no liability arising from use of this "
        "report. Source data is U.S. NOAA public-domain radar. Clear Claims Co. is an "
        "independent provider and is <strong style=\"color:#5a6b7e;\">not affiliated with "
        "Cotality or CoreLogic</strong>.")

    # Optional confidence + corroboration block (shown only if provided).
    conf_level = data.get("confidenceLevel")
    if conf_level:
        conf_color = data.get("confidenceColor", "#7d8ea1")
        conf_note = data.get("confidenceNote", "")
        corrob = data.get("corroborationLine", "")
        conf_block = f"""
    <div style="margin-top:10px; border:1px solid #e2e9f1; border-radius:4px; padding:10px 14px; display:flex; gap:14px; align-items:flex-start;">
      <div style="flex:none; background:{conf_color}; color:#fff; font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; padding:6px 11px; border-radius:3px; white-space:nowrap;">{conf_level} Confidence</div>
      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:3px;">Corroboration &amp; Confidence</div>
        <div style="font-size:10.5px; color:#46566a; line-height:1.5;">{conf_note}</div>
        <div style="font-size:10px; color:#5a6b7e; line-height:1.5; margin-top:4px;">{corrob}</div>
      </div>
    </div>"""
    else:
        conf_block = ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  {font_face}
  @page {{ size: 8.5in 11in; margin: 0; }}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; padding:0; background:#fff; }}
  body {{ font-family:'Outfit', sans-serif; }}
  /* Fixed one-page height + clip = the report can NEVER spill to a 2nd page,
     regardless of which fonts the renderer substitutes. */
  .page {{ width:816px; height:1056px; background:#ffffff; display:flex; flex-direction:column; overflow:hidden; }}
</style></head>
<body>
<div class="page">

  <!-- Header band -->
  <div style="background:#06101f; padding:26px 44px 22px; display:flex; align-items:flex-start; justify-content:space-between;">
    <div style="display:flex; align-items:center; gap:14px;">
      {_LOGO_SVG}
      <div>
        <div style="font-family:'DM Serif Display',serif; font-size:25px; line-height:1; color:#fff; white-space:nowrap;">Clear <span style="color:#4a9af5;">Claims</span> <span style="font-family:'Outfit'; font-size:12px; font-weight:500; color:#8aa0b8; letter-spacing:.03em;">Co.</span></div>
        <div style="font-family:'DM Serif Display',serif; font-style:italic; font-size:12px; color:#8aa0b8; margin-top:3px;">Fairness in every claim</div>
      </div>
    </div>
    <div style="text-align:right; font-size:11px; line-height:1.6; color:#9fb2c7; padding-top:3px;">
      <div style="color:#4a9af5; font-weight:600; letter-spacing:.04em;">{data["contactUrl"]}</div>
      <div>{data["contactCity"]}</div>
    </div>
  </div>
  <div style="background:#0c1a30; padding:11px 44px; border-top:1px solid #1c2c44; display:flex; align-items:center; justify-content:space-between;">
    <div style="font-family:'DM Serif Display',serif; font-size:20px; color:#fff; letter-spacing:.01em;">{data["reportTitle"]}</div>
    <div style="font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#6e84a0;">{data["bandLabel"]}</div>
  </div>

  <!-- Body -->
  <div style="padding:16px 44px 0; flex:1; display:flex; flex-direction:column;">

    <!-- Report details -->
    <div style="border:1px solid #d9e4f0; border-radius:3px; display:grid; grid-template-columns:1fr 1fr 1fr;">
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Report ID</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["reportId"]}</div></div>
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Date Generated</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["dateGenerated"]}</div></div>
      <div style="padding:9px 16px; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Date of Loss</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["dateOfLoss"]}</div></div>
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; grid-column:span 2;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Property Address</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["propertyAddress"]}</div></div>
      <div style="padding:9px 16px;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Claim / Reference</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["claimRef"]}</div></div>
      <div style="padding:9px 16px; border-top:1px solid #e7eef6; grid-column:span 3;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Coordinates</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px; font-variant-numeric:tabular-nums;">{data["coordinates"]}</div></div>
    </div>

    <!-- KEY FINDING callout -->
    <div style="margin-top:12px; background:{t['tint']}; border:1px solid {t['tintBorder']}; border-left:6px solid {t['main']}; border-radius:3px; padding:13px 18px; display:flex; align-items:center; gap:16px;">
      <div style="flex:none; width:52px; height:52px; border-radius:50%; background:{t['main']}; display:flex; align-items:center; justify-content:center;">{icon}</div>
      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.14em; text-transform:uppercase; color:{t['deep']}; font-weight:700;">Key Finding</div>
        <div style="font-family:'DM Serif Display',serif; font-size:24px; color:#06101f; line-height:1.08; margin-top:2px;">Hail of {threshold_label} or greater <span style="color:{t['dark']};">{finding_verb}</span> at this property.</div>
        <div style="font-size:11.5px; color:#54616f; margin-top:4px;">Maximum estimated hail at the property location on the date of loss was <strong style="color:#06101f;">{max_value}</strong> — {threshold_word} the {threshold_label} damage threshold.</div>
      </div>
      <div style="flex:none; background:{t['main']}; color:#fff; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; padding:8px 14px; border-radius:3px; text-align:center; white-space:nowrap;">{status_text}</div>
    </div>

    <!-- Two columns: table + map -->
    <div style="display:flex; gap:20px; margin-top:12px;">
      <div style="flex:none; width:300px;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:8px;">Estimated Maximum Hail Size</div>
        <table style="width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;">
          <thead><tr style="background:#0c1a30;">
            <th style="text-align:left; padding:8px 12px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">Location</th>
            <th style="text-align:right; padding:8px 10px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">in</th>
            <th style="text-align:right; padding:8px 12px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">mm</th>
          </tr></thead>
          <tbody>
            <tr style="background:{t['rowTint']};"><td style="padding:9px 12px; font-size:12.5px; color:#06101f; font-weight:700; border-bottom:1px solid #e7eef6;">At Property</td><td style="padding:9px 10px; text-align:right; font-size:13px; color:{t['dark']}; font-weight:700; border-bottom:1px solid #e7eef6;">{ap['in']}</td><td style="padding:9px 12px; text-align:right; font-size:12px; color:#5a6b7e; border-bottom:1px solid #e7eef6;">{ap['mm']}</td></tr>
            <tr><td style="padding:9px 12px; font-size:12px; color:#0c1a30; border-bottom:1px solid #eef3f8;">Within 1 mile</td><td style="padding:9px 10px; text-align:right; font-size:13px; color:#06101f; font-weight:600; border-bottom:1px solid #eef3f8;">{m1['in']}</td><td style="padding:9px 12px; text-align:right; font-size:12px; color:#5a6b7e; border-bottom:1px solid #eef3f8;">{m1['mm']}</td></tr>
            <tr style="background:#f7fafd;"><td style="padding:9px 12px; font-size:12px; color:#0c1a30; border-bottom:1px solid #eef3f8;">Within 3 miles</td><td style="padding:9px 10px; text-align:right; font-size:13px; color:#06101f; font-weight:600; border-bottom:1px solid #eef3f8;">{m3['in']}</td><td style="padding:9px 12px; text-align:right; font-size:12px; color:#5a6b7e; border-bottom:1px solid #eef3f8;">{m3['mm']}</td></tr>
            <tr><td style="padding:9px 12px; font-size:12px; color:#0c1a30;">Within 5 miles</td><td style="padding:9px 10px; text-align:right; font-size:13px; color:#06101f; font-weight:600;">{m5['in']}</td><td style="padding:9px 12px; text-align:right; font-size:12px; color:#5a6b7e;">{m5['mm']}</td></tr>
          </tbody>
        </table>
        <div style="font-size:9px; color:#90a0b2; margin-top:7px; line-height:1.4;">Values are peak radar-estimated diameters within each radius during storm passage.</div>
      </div>

      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:8px;">Hail Footprint</div>
        <div style="border:1px solid #c4d2e2; border-radius:3px; overflow:hidden;">
          {map_block}
          <div style="display:flex; align-items:center; gap:0; padding:7px 10px; border-top:1px solid #e7eef6; background:#fff;">
            <span style="font-size:8.5px; color:#7d8ea1; margin-right:8px; font-weight:600;">SIZE</span>
            <span style="flex:1; height:9px; border-radius:2px; background:linear-gradient(90deg,#28a678,#7cc36a,#e6a117,#e07a2e,#d94f3d);"></span>
            <span style="font-size:8.5px; color:#7d8ea1; margin-left:8px;">0.5″</span>
            <span style="font-size:8.5px; color:#7d8ea1; margin-left:6px;">2.5″+</span>
          </div>
        </div>
        <div style="font-size:9px; color:#90a0b2; margin-top:7px; line-height:1.4;">{data["mapCaption"]}</div>
      </div>
    </div>

    {conf_block}

    <!-- Methodology -->
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #e7eef6;">
      <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:5px;">Methodology</div>
      <div style="font-size:9.7px; color:#46566a; line-height:1.45;">{methodology}</div>
    </div>

    <!-- Disclaimer -->
    <div style="margin-top:10px; padding-top:0; margin-bottom:20px;">
      <div style="background:#f0f4f8; border-radius:3px; padding:11px 14px;">
        <div style="font-size:8px; letter-spacing:.12em; text-transform:uppercase; color:#8a99ab; font-weight:700; margin-bottom:4px;">Disclaimer</div>
        <div style="font-size:8.5px; color:#7a8a9c; line-height:1.5;">{disclaimer}</div>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div style="background:#06101f; padding:9px 44px; display:flex; align-items:center; justify-content:space-between; font-size:8.5px; color:#7d8ea1; letter-spacing:.04em;">
    <span>Report {data["reportId"]}</span>
    <span style="color:#4a9af5; font-weight:600; letter-spacing:.1em; text-transform:uppercase;">Confidential</span>
    <span>Page 1 of 1 · Generated {data["dateGenerated"]}</span>
  </div>
</div>
</body></html>"""


def render_pdf_weasyprint(html: str, out_pdf: str) -> str:
    """Render the report HTML to a PDF using WeasyPrint (no browser required)."""
    from weasyprint import HTML
    HTML(string=html, base_url=".").write_pdf(out_pdf)
    return out_pdf


@dataclass
class ReportInputs:
    """Everything the PDF needs, gathered in one tidy object."""
    report_id: str
    address_label: str
    lat: float
    lon: float
    location_source: str
    date_of_loss: dt.date
    tz_name: str
    generated: dt.datetime
    threshold_in: float
    radius_miles: int
    sample: dict
    detected: bool
    map_png: str
    files_used: list = field(default_factory=list)


def build_pdf(out_pdf: str, r: ReportInputs, fonts: dict, logo_path: str | None = None):
    """Render the one-page branded Hail Verification Report PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor

    W, H = letter
    c = canvas.Canvas(out_pdf, pagesize=letter)

    midnight = HexColor(BRAND["midnight"])
    slate = HexColor(BRAND["slate"])
    accent = HexColor(BRAND["accent"])
    bright = HexColor(BRAND["bright"])
    green = HexColor(BRAND["green"])
    coral = HexColor(BRAND["coral"])
    ice = HexColor(BRAND["ice"])
    ink = HexColor("#1f2c3d")
    grey = HexColor("#5a6b7e")

    result_color = coral if r.detected else green
    M = 0.6 * inch                      # page margin

    # ---------- HEADER BAND ----------
    band_h = 0.95 * inch
    c.setFillColor(midnight)
    c.rect(0, H - band_h, W, band_h, fill=1, stroke=0)

    logo_drawn = False
    if logo_path and os.path.isfile(logo_path):
        try:
            img = ImageReader(logo_path)
            iw, ih = img.getSize()
            target_h = 0.55 * inch
            target_w = target_h * iw / ih
            c.drawImage(img, M, H - band_h + (band_h - target_h) / 2,
                        width=target_w, height=target_h, mask="auto")
            text_x = M + target_w + 12
            logo_drawn = True
        except Exception:
            logo_drawn = False
    if not logo_drawn:
        text_x = M

    # Wordmark: "Clear" white + "Claims" bright-blue + " Co." muted (brand spec).
    from reportlab.pdfbase.pdfmetrics import stringWidth
    wm_y = H - 0.50 * inch
    c.setFont(fonts["head"], 22)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(text_x, wm_y, "Clear ")
    x2 = text_x + stringWidth("Clear ", fonts["head"], 22)
    c.setFillColor(bright)
    c.drawString(x2, wm_y, "Claims")
    x3 = x2 + stringWidth("Claims", fonts["head"], 22)
    c.setFillColor(ice)
    c.setFont(fonts["body"], 11)
    c.drawString(x3 + 4, wm_y, "Co.")
    c.setFillColor(ice)
    c.setFont(fonts["body"], 9.5)
    c.drawString(text_x, H - 0.68 * inch, BRAND["tagline"])

    c.setFillColor(bright)
    c.setFont(fonts["body_bold"], 9)
    c.drawRightString(W - M, H - 0.50 * inch, BRAND["contact"])
    c.setFillColor(ice)
    c.setFont(fonts["body"], 8.5)
    c.drawRightString(W - M, H - 0.66 * inch, "Radar-Based Hail Verification")

    # Title strip
    strip_y = H - band_h - 0.34 * inch
    c.setFillColor(slate)
    c.rect(0, strip_y, W, 0.34 * inch, fill=1, stroke=0)
    c.setFillColor(HexColor("#ffffff"))
    c.setFont(fonts["head"], 13)
    c.drawString(M, strip_y + 0.10 * inch, "Hail Verification Report")
    c.setFillColor(ice)
    c.setFont(fonts["body"], 8)
    c.drawRightString(W - M, strip_y + 0.11 * inch, f"Report {r.report_id}")

    y = strip_y - 0.30 * inch

    # ---------- METADATA BLOCK ----------
    def meta_row(label, value, yy):
        c.setFillColor(grey)
        c.setFont(fonts["body_bold"], 7.5)
        c.drawString(M, yy, label.upper())
        c.setFillColor(ink)
        c.setFont(fonts["body"], 10)
        c.drawString(M + 1.55 * inch, yy, value)

    coord_str = f"{r.lat:.5f} N, {abs(r.lon):.5f} {'W' if r.lon < 0 else 'E'}"
    meta_row("Property", r.address_label, y); y -= 0.205 * inch
    meta_row("Coordinates", f"{coord_str}   ({r.location_source})", y); y -= 0.205 * inch
    meta_row("Date of Loss", f"{r.date_of_loss:%B %d, %Y}  (local: {r.tz_name})", y); y -= 0.205 * inch
    meta_row("Report Generated", f"{r.generated:%B %d, %Y  %H:%M UTC}", y); y -= 0.28 * inch

    # ---------- RESULTS PANEL ----------
    panel_h = 1.18 * inch
    panel_y = y - panel_h
    c.setFillColor(HexColor("#f0f4f8"))
    c.roundRect(M, panel_y, W - 2 * M, panel_h, 6, fill=1, stroke=0)
    c.setFillColor(result_color)
    c.roundRect(M, panel_y, 0.12 * inch, panel_h, 3, fill=1, stroke=0)

    verdict = "HAIL DETECTED" if r.detected else "NO HAIL DETECTED"
    c.setFillColor(result_color)
    c.setFont(fonts["head"], 20)
    c.drawString(M + 0.30 * inch, panel_y + panel_h - 0.42 * inch, verdict)

    c.setFillColor(grey)
    c.setFont(fonts["body"], 9)
    thresh_txt = (f"Based on a detection threshold of {r.threshold_in:.2f}\" "
                  f"(MESH within {r.radius_miles} mile"
                  f"{'s' if r.radius_miles != 1 else ''} of the property).")
    c.drawString(M + 0.30 * inch, panel_y + panel_h - 0.62 * inch, thresh_txt)

    s = r.sample

    def big_stat(x, label, value_in, value_mm, color=ink):
        c.setFillColor(grey)
        c.setFont(fonts["body_bold"], 7.5)
        c.drawString(x, panel_y + 0.40 * inch, label.upper())
        c.setFillColor(color)
        c.setFont(fonts["head"], 17)
        c.drawString(x, panel_y + 0.14 * inch, f"{value_in:.2f}\"")
        c.setFillColor(grey)
        c.setFont(fonts["body"], 8)
        c.drawString(x + 0.78 * inch, panel_y + 0.18 * inch, f"({value_mm:.0f} mm)")

    big_stat(M + 0.30 * inch, "Max hail at property", s["point_in"], s["point_mm"],
             result_color)
    big_stat(M + 3.05 * inch, f"Max within {r.radius_miles} mi",
             s["radius_max_in"], s["radius_max_mm"], result_color)
    c.setFillColor(grey)
    c.setFont(fonts["body"], 7.5)
    c.drawString(M + 5.55 * inch, panel_y + 0.40 * inch, "THRESHOLD")
    c.setFillColor(ink)
    c.setFont(fonts["head"], 17)
    c.drawString(M + 5.55 * inch, panel_y + 0.14 * inch, f"{r.threshold_in:.2f}\"")

    y = panel_y - 0.22 * inch

    # ---------- MAP ----------
    if r.map_png and os.path.isfile(r.map_png):
        img = ImageReader(r.map_png)
        iw, ih = img.getSize()
        map_w = W - 2 * M
        map_h = map_w * ih / iw
        max_h = 3.0 * inch
        if map_h > max_h:
            map_h = max_h
            map_w = map_h * iw / ih
        c.drawImage(img, (W - map_w) / 2, y - map_h, width=map_w, height=map_h,
                    mask="auto")
        y = y - map_h - 0.12 * inch
    c.setFillColor(grey)
    c.setFont(fonts["body"], 7.5)
    c.drawCentredString(W / 2, y,
                        f"Estimated hail footprint — NOAA MRMS MESH, {r.date_of_loss:%B %d, %Y}. "
                        f"Marker = property; blue ring = {r.radius_miles}-mile sampling radius.")
    y -= 0.22 * inch

    # ---------- METHODOLOGY ----------
    def para(title, text, yy, size=8):
        c.setFillColor(slate)
        c.setFont(fonts["body_bold"], 8)
        c.drawString(M, yy, title.upper())
        yy -= 0.155 * inch
        c.setFillColor(grey)
        c.setFont(fonts["body"], size)
        # simple word-wrap
        from reportlab.pdfbase.pdfmetrics import stringWidth
        max_w = W - 2 * M
        words, line = text.split(), ""
        for w in words:
            test = (line + " " + w).strip()
            if stringWidth(test, fonts["body"], size) > max_w:
                c.drawString(M, yy, line); yy -= 0.135 * inch; line = w
            else:
                line = test
        if line:
            c.drawString(M, yy, line); yy -= 0.135 * inch
        return yy - 0.06 * inch

    method_txt = (
        "Hail sizes are derived from NOAA's Multi-Radar/Multi-Sensor (MRMS) Maximum "
        "Estimated Size of Hail (MESH) product — a radar algorithm that estimates the "
        "largest hail a storm likely produced, on a grid of roughly 1 km (~0.6 mile) "
        "cells. This report uses the 24-hour maximum (MESH_Max_1440min) for the local "
        f"date of loss. Because each grid cell covers ~1 km and an address pinpoint can "
        f"be slightly off, we report both the value at the nearest cell and the maximum "
        f"within {r.radius_miles} mile(s). MESH is a radar ESTIMATE, not a measurement.")
    y = para("Methodology", method_txt, y)

    src_txt = ("Data source: NOAA Multi-Radar/Multi-Sensor System (MRMS), MESH product, "
               "obtained from the NOAA Open Data Dissemination program on AWS "
               "(s3://noaa-mrms-pds). NOAA data are public domain. This product is not "
               "endorsed by and does not imply affiliation with NOAA.")
    y = para("Data Source & Attribution", src_txt, y)

    disc_txt = (
        "DISCLAIMER: This is a radar-derived ESTIMATE provided for informational "
        "purposes only. It is NOT a physical inspection, NOT a guarantee that hail "
        "damage did or did not occur, and NOT a substitute for an on-site assessment "
        "by a qualified adjuster or inspector. Clear Claims Co. makes no warranty and "
        "accepts no liability arising from use of this report.")
    y = para("Liability Disclaimer", disc_txt, y)

    # ---------- FOOTER ----------
    c.setFillColor(midnight)
    c.rect(0, 0, W, 0.32 * inch, fill=1, stroke=0)
    c.setFillColor(ice)
    c.setFont(fonts["body"], 7.5)
    c.drawString(M, 0.12 * inch, f"Report {r.report_id}")
    c.setFillColor(bright)
    c.drawCentredString(W / 2, 0.12 * inch, "CONFIDENTIAL")
    c.setFillColor(ice)
    c.drawRightString(W - M, 0.12 * inch, f"Generated {r.generated:%Y-%m-%d}  ·  Page 1 of 1")

    c.showPage()
    c.save()
    return out_pdf
