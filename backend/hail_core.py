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


def sample_rings(lats, lons, mesh_mm, lat, lon, rings=(0.5, 1, 3, 5)):
    """Value at the property (nearest cell) plus the MAX within each ring radius.

    CRITICAL (defect D1): a cell with NO RADAR COVERAGE is NaN, and NaN is NOT
    zero hail. This function therefore returns ``None`` for mm/in whenever the
    footprint contains no valid radar data at all, and reports how many cells
    were valid. Callers must render "coverage unavailable", never "0.00 in".

    Returns {'point': {...}, 0.5: {...}, 1: {...}, 3: {...}, 5: {...}} where each
    entry is {'mm', 'in', 'lat', 'lon', 'valid_cells', 'total_cells'} and mm/in
    are None when no valid cell was found.
    """
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)
    finite = np.isfinite(mesh_mm)

    pi, pj = np.unravel_index(np.nanargmin(dist), dist.shape)
    p_valid = bool(finite[pi, pj])
    p_mm = float(mesh_mm[pi, pj]) if p_valid else None
    out = {"point": {"mm": p_mm,
                     "in": (p_mm / MM_PER_INCH) if p_mm is not None else None,
                     "lat": float(lats[pi]), "lon": float(lons[pj]),
                     "valid_cells": int(p_valid), "total_cells": 1}}

    for r in rings:
        mask = dist <= r
        if not mask.any():
            mask = np.zeros_like(dist, dtype=bool)
            mask[pi, pj] = True
        valid = mask & finite
        n_total, n_valid = int(mask.sum()), int(valid.sum())
        if n_valid == 0:
            # No radar coverage anywhere in this footprint. NOT zero hail.
            out[r] = {"mm": None, "in": None, "lat": lat, "lon": lon,
                      "valid_cells": 0, "total_cells": n_total}
            continue
        vals = np.where(valid, mesh_mm, np.nan)
        mi, mj = np.unravel_index(np.nanargmax(vals), vals.shape)
        mm = float(mesh_mm[mi, mj])
        out[r] = {"mm": mm, "in": mm / MM_PER_INCH,
                  "lat": float(lats[mi]), "lon": float(lons[mj]),
                  "valid_cells": n_valid, "total_cells": n_total}
    return out


# -----------------------------------------------------------------------------
#  RADAR COVERAGE STATE  (defect D1 — "no data" must never look like "no hail")
# -----------------------------------------------------------------------------

#  Fraction of cells inside COVERAGE_RADIUS_MI that must carry valid radar data
#  before we are willing to state a hail size at all. Tunable via env.
COVERAGE_RADIUS_MI = 5.0
COVERAGE_OK_FRAC = 0.80      # >= this -> "ok"
COVERAGE_MIN_FRAC = 0.50     # >= this -> "partial";  below -> "none"


def assess_coverage(lats, lons, mesh_mm, lat, lon, radius_miles=COVERAGE_RADIUS_MI):
    """Classify radar coverage around the property as 'ok' | 'partial' | 'none'.

    MRMS encodes no-coverage / missing with large negative sentinels, which
    read_mesh_grib turns into NaN. A location in a radar gap or behind terrain
    blockage (very relevant in the Black Hills) therefore has few or no valid
    cells — and MUST NOT be reported as 0.00 in / NOT DETECTED.

    Returns {state, valid_frac, valid_cells, total_cells, radius_miles}.
    """
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)
    mask = dist <= radius_miles
    if not mask.any():
        pi, pj = np.unravel_index(np.nanargmin(dist), dist.shape)
        mask = np.zeros_like(dist, dtype=bool)
        mask[pi, pj] = True
    total = int(mask.sum())
    valid = int((mask & np.isfinite(mesh_mm)).sum())
    frac = (valid / total) if total else 0.0
    if frac >= COVERAGE_OK_FRAC:
        state = "ok"
    elif frac >= COVERAGE_MIN_FRAC:
        state = "partial"
    else:
        state = "none"
    return {"state": state, "valid_frac": frac, "valid_cells": valid,
            "total_cells": total, "radius_miles": radius_miles}


# -----------------------------------------------------------------------------
#  HAIL CLASSIFICATION  (T0-2 / T0-3 / T0-4)
# -----------------------------------------------------------------------------
#  Two independent judgements, deliberately kept apart:
#
#    * detected  -> a BUSINESS badge: did the radar peak meet the client's
#                   damage threshold (default 0.75 in, overridable per request)?
#    * likelihood-> a SCIENCE statement anchored to published verification, not
#                   to the client's threshold. Anchors used:
#                     0.50 in  practical floor; below this MESH cannot
#                              discriminate hail from ordinary convection
#                              (Witt SHI is ~6 at 0.25 in — that is "there was
#                              a storm", not "there was hail").
#                     1.14 in  = 29 mm, the best operational MESH proxy for a
#                              1.00 in ground report (Wendt & Jirak 2021).
#                     2.00 in  significant-hail territory.
#
#  Thresholds drive badges. Thresholds NEVER hide a value (defect T0-2).
# -----------------------------------------------------------------------------

MESH_DISCRIMINATION_FLOOR_IN = 0.50
MESH_SEVERE_PROXY_IN = 1.14      # 29 mm — Wendt & Jirak (2021)
SIGNIFICANT_HAIL_IN = 2.00

MESH_DISCLOSURE = (
    "MESH is NOAA&rsquo;s radar-derived 75th-percentile maximum estimated hail size, "
    "not a stone measured at this address.")


def classify_hail(peak_in, cell_in, threshold_in, coverage_state="ok"):
    """Turn the sampled numbers into the report's verdict, badge and theme.

    `peak_in` is the peak within 1/2 mile; `cell_in` is the nearest grid cell.
    Either may be None (no valid radar data). Returns a dict consumed directly
    by the PDF template.
    """
    thr = f"{threshold_in:.2f}\u2033"

    if coverage_state == "none" or peak_in is None:
        return {
            "band": "no_coverage", "theme": "unknown", "detected": None,
            "badge": "Coverage Unavailable",
            "verdict": "Radar coverage was unavailable at this location on this date.",
            "detail": ("No hail size can be stated. This is an absence of data, "
                       "not evidence that hail did not occur."),
            "likelihood": "Cannot be determined",
        }

    detected = bool(peak_in >= threshold_in)

    if peak_in <= 0.0:
        band, theme = "none", "clear"
        badge = "None Detected"
        verdict = "Radar shows no hail signature at this property on this date."
        detail = ("A radar-based negative is weaker evidence than a positive. "
                  "Small or brief hail can fall below what the radar resolves.")
        likelihood = "No indication"
    elif peak_in < MESH_DISCRIMINATION_FLOOR_IN:
        band, theme = "trace", "caution"
        badge = "Trace / Indeterminate"
        verdict = "Damaging hail is UNLIKELY to have occurred at this property."
        detail = ("The radar value is below the level at which MESH can "
                  "distinguish hail from ordinary convection. The figure is "
                  "reported for completeness, not as verified hail.")
        likelihood = "Unlikely"
    elif peak_in < 0.75:
        band, theme = "indicated", "caution"
        badge = "Hail Indicated"
        verdict = ("Small hail is INDICATED near this property, below the "
                   "0.75\u2033 threshold commonly used for roof damage.")
        detail = ("Hail at this size is generally not associated with functional "
                  "damage to conventional roofing, but is not zero.")
        likelihood = "Possible \u2014 sub-threshold"
    elif peak_in < MESH_SEVERE_PROXY_IN:
        band, theme = "threshold", "detected"
        badge = "At or Above Threshold"
        verdict = (f"Hail of {thr} or greater is PROBABLE within \u00bd mile of "
                   f"this property.")
        detail = ("Occurrence of 1.00\u2033 (NWS severe) hail is UNCERTAIN at this "
                  "radar value \u2014 published verification puts the best match to a "
                  "1.00\u2033 ground report at MESH 1.14\u2033.")
        likelihood = "Probable"
    elif peak_in < SIGNIFICANT_HAIL_IN:
        band, theme = "severe", "detected"
        badge = "Severe Hail Likely"
        verdict = ("Hail of 1.00\u2033 or greater is LIKELY to have occurred within "
                   "\u00bd mile of this property.")
        detail = ("This radar value is at or above 1.14\u2033, the best operational "
                  "MESH match to a verified 1.00\u2033 ground report "
                  "(Wendt &amp; Jirak 2021).")
        likelihood = "Likely"
    else:
        band, theme = "significant", "detected"
        badge = "Significant Hail Likely"
        verdict = ("Hail of 1.00\u2033 or greater is LIKELY, and radar indicates "
                   "significant hail within \u00bd mile of this property.")
        detail = ("Radar values in this range are associated with hail capable of "
                  "damaging most conventional roofing materials.")
        likelihood = "Likely \u2014 significant"

    return {"band": band, "theme": theme, "detected": detected, "badge": badge,
            "verdict": verdict, "detail": detail, "likelihood": likelihood}


def stable_report_id(label: str, date_of_loss, prefix: str = "CC") -> str:
    """Deterministic report ID (defect D4).

    The old implementation used Python's hash(), which is randomised per process
    by PYTHONHASHSEED — so the same address and date produced a DIFFERENT id
    after every deploy, and 5 decimal digits collide at roughly 370 reports.
    A SHA-1 digest is stable forever and collides far later.
    """
    import hashlib
    seed = f"{label}|{date_of_loss}".encode("utf-8")
    return f"{prefix}-{date_of_loss:%Y}-{hashlib.sha1(seed).hexdigest()[:8].upper()}"


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


def assess_confidence(point_in, ring_max_in, reports, threshold_in, source=None,
                      coverage_state="ok"):
    """Combine radar + ground reports into a stated confidence level.

    Returns {level, color, note, n_reports} — or level=None when radar coverage
    was insufficient, in which case the report shows NO confidence chip at all
    (defect D1 + D7: we must never print a confident verdict over missing data).

    Note: the full confidence matrix additionally weights radar range/beam
    height. That arrives with the radar-quality score; until then a clean
    negative is capped at Moderate rather than High.
    """
    if coverage_state == "none" or point_in is None:
        return {"level": None, "color": None, "n_reports": len(reports),
                "note": ("Radar coverage was insufficient at this location on this "
                         "date, so no confidence level is stated. This is an absence "
                         "of data, not evidence that hail did not occur.")}

    radar_max = max(point_in, ring_max_in if ring_max_in is not None else point_in)
    detected = point_in >= threshold_in
    n = len(reports)
    biggest = max([r["size_in"] for r in reports if r["size_in"]], default=0.0)
    partial = (coverage_state == "partial")

    if detected:
        if n >= 1:
            level = "High"
            note = (f"Radar-estimated hail is corroborated by {n} independent ground "
                    f"report(s) within the search area"
                    + (f" (largest {biggest:.2f}\u2033)." if biggest else "."))
        elif radar_max >= threshold_in + 0.50:
            level = "Moderate"
            note = ("Radar estimate is well above the threshold, but no independent "
                    "ground report was logged nearby. Storm reports are sparse in "
                    "rural areas, so this does not contradict the radar.")
        else:
            level = "Moderate"
            note = ("Radar estimate is near the threshold with no nearby ground "
                    "report; treat as a borderline result.")
    else:
        if n >= 1:
            level = "Low"
            note = (f"Radar did not meet the threshold at the property, yet {n} hail "
                    f"report(s) were logged nearby \u2014 verify exact timing and location.")
        else:
            # A radar-based NEGATIVE is inherently weaker than a positive: brief or
            # small hail can fall below what the radar resolves, and ground reports
            # are sparse where nobody lives. Capped at Moderate (was: High).
            level = "Moderate"
            note = ("Radar shows no significant hail at the property and no ground "
                    "reports were logged nearby. A radar-based negative is weaker "
                    "evidence than a positive \u2014 brief or small hail can fall below "
                    "what the radar resolves.")

    if partial:
        level = {"High": "Moderate", "Moderate": "Low", "Low": "Low"}[level]
        note += (" Radar coverage at this location was only partial, so confidence "
                 "has been reduced accordingly.")

    color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[level]
    return {"level": level, "color": color, "note": note, "n_reports": n}


def corroboration_line(reports, radius_miles, coverage_state="ok") -> str:
    """One-line human summary of the nearest ground reports (for the report)."""
    if not reports:
        if coverage_state == "none":
            return (f"No independent ground reports within {radius_miles:.0f} miles on "
                    f"this date either. Ground reports are sparse in rural areas and "
                    f"their absence is not evidence that hail did not occur.")
        return (f"No independent ground reports within {radius_miles:.0f} miles on this "
                f"date (radar-only estimate). Ground reports are sparse in rural areas; "
                f"their absence does not contradict the radar.")
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
#  7.  ASSET HELPERS FOR THE PDF
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
_THEME_CAUTION = dict(main="#e6a117", dark="#c98c10", deep="#a5720c",
                      tint="#fdf6e7", tintBorder="#f2e0b4", rowTint="#fdf9ef")
_THEME_UNKNOWN = dict(main="#5a6b7e", dark="#4a5d76", deep="#3b4c62",
                      tint="#eef2f7", tintBorder="#d5dfea", rowTint="#f4f7fa")
_THEMES = {"detected": _THEME_DETECTED, "clear": _THEME_CLEAR,
           "caution": _THEME_CAUTION, "unknown": _THEME_UNKNOWN}

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
_ICON_INFO = ('<svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="#fff" '
              'stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">'
              '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.5v.01"/></svg>')
_ICON_QUESTION = ('<svg width="27" height="27" viewBox="0 0 24 24" fill="none" stroke="#fff" '
                  'stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">'
                  '<circle cx="12" cy="12" r="9"/>'
                  '<path d="M9.4 9.2a2.7 2.7 0 0 1 5.2.9c0 1.8-2.6 2.7-2.6 2.7"/>'
                  '<path d="M12 17.2v.01"/></svg>')
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


# --------------------------------------------------------------------------- #
#  Revised report template (design refresh, 2026-06).  Uses str.format — every
#  literal CSS brace is doubled; single-brace tokens are the fill-ins. Fed by
#  the existing pipeline `data` dict (keys mapped in build_report_html below).
# --------------------------------------------------------------------------- #
_HAIL_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
<style>
  {font_face}
  @page {{ size: 8.5in 11in; margin: 0; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ font-family: 'Outfit', Helvetica, Arial, sans-serif; color: #152742; }}
  /* Fixed one-page height + clip = the report can never spill to a 2nd page. */
  .page {{ width: 8.5in; height: 11in; overflow: hidden; display: block; position: relative; }}

  .hdr {{ background: #06101f; padding: 11px 40px; display: flex; align-items: center; justify-content: space-between; }}
  .brand {{ display: flex; align-items: center; gap: 14px; }}
  .brand svg.logo {{ width: 42px; height: 49px; display: block; }}
  .wm {{ font-family: 'DM Serif Display', serif; font-size: 25px; line-height: 1; color: #f0f4f8; white-space: nowrap; }}
  .wm .b {{ color: #4a9af5; }}
  .wm .co {{ font-size: 15px; color: #8fa3b8; margin-left: 3px; }}
  .tag {{ font-family: 'DM Serif Display', serif; font-style: italic; font-size: 13px; color: #8fa3b8; margin-top: 5px; }}
  .hdr-right {{ text-align: right; }}
  .hdr-right .site {{ font-size: 13px; font-weight: 600; color: #4a9af5; }}
  .hdr-right .loc {{ font-size: 13px; color: #b8cce0; margin-top: 4px; }}

  .titleband {{ background: #0b1626; padding: 9px 40px; display: flex; align-items: baseline; justify-content: space-between; }}
  .titleband h1 {{ font-family: 'DM Serif Display', serif; font-weight: 400; font-size: 27px; color: #f0f4f8; letter-spacing: .2px; }}
  .titleband .kick {{ font-size: 12px; font-weight: 500; letter-spacing: .28em; text-transform: uppercase; color: #5a6b7e; }}

  .body {{ display: block; padding: 6px 40px 4px; }}

  .meta {{ display: grid; grid-template-columns: 1fr 1fr 1fr; border: 1px solid #dde6f0; border-radius: 9px; overflow: hidden; }}
  .meta .cell {{ padding: 4px 16px; border-right: 1px solid #e7eef6; border-bottom: 1px solid #e7eef6; min-width: 0; }}
  .meta .cell.c3 {{ border-right: none; }}
  .meta .cell.span2 {{ grid-column: span 2; }}
  .meta .cell.row-last {{ border-bottom: none; }}
  .meta .lbl {{ font-size: 9px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #5a6b7e; }}
  .meta .val {{ font-size: 14.5px; font-weight: 600; color: #152742; margin-top: 3px; }}
    .val {{ overflow-wrap: anywhere; }}

  .keyfind {{ margin-top: 6px; display: table; width: 100%; box-sizing: border-box;
    background: {kf_bg}; border: 1px solid {kf_bd}; border-left: 5px solid {kf_accent};
    border-radius: 10px; padding: 10px 18px; }}
  .kf-cell {{ display: table-cell; vertical-align: middle; }}
  .kf-icon-cell {{ width: 58px; min-width: 58px; }}
  .kf-icon {{ width: 52px; height: 52px; border-radius: 50%; background: {kf_accent};
    display: flex; align-items: center; justify-content: center; }}
  .kf-icon svg {{ width: 30px; height: 30px; display: block; }}
  .kf-main {{ padding: 0 18px; }}
  .kf-lbl {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {kf_accent}; }}
  .keyfind h2 {{ font-family: 'DM Serif Display', serif; font-weight: 400; font-size: 19.5px; line-height: 1.1; color: #06101f; margin-top: 4px; }}
  .keyfind h2 .fig {{ color: {kf_accent}; }}
  .kf-sub {{ font-size: 12.5px; color: #4a5d76; line-height: 1.4; margin-top: 7px; }}
  .kf-sub b {{ color: #152742; font-weight: 600; }}
  .kf-note {{ font-size: 10.5px; color: #6b7d94; line-height: 1.38; margin-top: 5px; }}
  .kf-badge-cell {{ white-space: nowrap; }}
  .kf-badge {{ display: inline-block; background: {kf_accent}; color: #fff; font-size: 13px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; padding: 12px 18px; border-radius: 8px; white-space: nowrap; }}

  .seclbl {{ font-size: 11px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #152742; }}

  .duo {{ margin-top: 6px; display: grid; grid-template-columns: 0.92fr 1.18fr; gap: 26px; align-items: start; }}

  table {{ width: 100%; border-collapse: collapse; margin-top: 7px; border-radius: 8px; overflow: hidden; }}
  thead th {{ background: #0e2138; color: #f0f4f8; text-align: left; font-size: 9.5px; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase; padding: 7px 14px; }}
  thead th.num {{ text-align: right; }}
  tbody td {{ padding: 4px 14px; font-size: 12.5px; color: #152742; border-bottom: 1px solid #e7eef6; }}
  tbody td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tbody tr:nth-child(even) {{ background: #f4f8fc; }}
  tbody tr.hot {{ background: {kf_bg}; }}
  tbody tr.hot td {{ font-weight: 700; }}
  tbody tr.hot td:first-child {{ box-shadow: inset 3px 0 0 {kf_accent}; }}
  tbody tr.hot td.num {{ color: {kf_accent}; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .cap {{ font-size: 9px; color: #8a99ab; line-height: 1.4; margin-top: 6px; }}

  .map-wrap img {{ width: 100%; height: 163px; object-fit: contain; object-position: center; display: block;
    border: 1px solid #dde6f0; border-radius: 8px; background: #ffffff; }}

  .conf {{ margin-top: 6px; border: 1px solid #dde6f0; border-radius: 10px; padding: 7px 18px;
    display: flex; align-items: center; gap: 20px; }}
  .conf .chip {{ flex: 0 0 auto; color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; padding: 11px 16px; border-radius: 8px; }}
  .conf .conf-body .seclbl {{ margin-bottom: 6px; }}
  .conf .conf-body p {{ font-size: 11.5px; color: #4a5d76; line-height: 1.42; }}
  .conf .conf-body p + p {{ margin-top: 5px; }}

  .method {{ margin-top: 5px; }}
  .method p {{ font-size: 11px; color: #4a5d76; line-height: 1.42; margin-top: 4px; }}
  .method p.mesh-disc {{ color: #152742; font-weight: 600; }}

  .disc {{ margin-top: 5px; background: #f0f4f8; border-radius: 8px; padding: 6px 16px; margin-bottom: 44px; }}
  .disc .dl {{ font-size: 9px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #8a99ab; margin-bottom: 5px; }}
  .disc p {{ font-size: 9px; color: #8a99ab; line-height: 1.42; }}
  .disc b, .disc strong {{ color: #5a6b7e; }}

  .spacer {{ display: none; }}

  .foot {{ background: #06101f; padding: 13px 40px; display: flex; align-items: center; justify-content: space-between; position: absolute; left: 0; right: 0; bottom: 0; }}
  .foot span {{ font-size: 10px; color: #8fa3b8; letter-spacing: .03em; }}
  .foot .conf-tag {{ color: #4a9af5; font-weight: 700; letter-spacing: .18em; }}
</style>
</head>
<body>
  <div class="page">

    <div class="hdr">
      <div class="brand">
        {logo}
        <div>
          <div class="wm">Clear <span class="b">Claims</span> <span class="co">Co.</span></div>
          <div class="tag">Fairness in every claim</div>
        </div>
      </div>
      <div class="hdr-right">
        <div class="site">{contact_url}</div>
        <div class="loc">{contact_city}</div>
      </div>
    </div>

    <div class="titleband">
      <h1>{report_title}</h1>
      <div class="kick">{band_label}</div>
    </div>

    <div class="body">

      <div class="meta">
        <div class="cell"><div class="lbl">Report ID</div><div class="val">{report_id}</div></div>
        <div class="cell"><div class="lbl">Date Generated</div><div class="val">{report_date}</div></div>
        <div class="cell c3"><div class="lbl">Date of Loss</div><div class="val">{date_of_loss}</div></div>
        <div class="cell span2"><div class="lbl">Property Address</div><div class="val">{address}</div></div>
        <div class="cell c3"><div class="lbl">Claim / Reference</div><div class="val">{claim_ref}</div></div>
        <div class="cell row-last"><div class="lbl">Coordinates</div><div class="val">{coords}</div></div>
        <div class="cell span2 c3 row-last"></div>
      </div>

      <div class="keyfind">
        <div class="kf-cell kf-icon-cell"><div class="kf-icon">{icon}</div></div>
        <div class="kf-cell kf-main">
          <div class="kf-lbl">Key Finding</div>
          <h2>{verdict}</h2>
          <div class="kf-sub">{kf_readings}</div>
          <div class="kf-note">{kf_detail}</div>
        </div>
        <div class="kf-cell kf-badge-cell"><div class="kf-badge">{badge}</div></div>
      </div>

      <div class="duo">
        <div class="size-wrap">
          <div class="seclbl">Radar-Estimated Hail Size</div>
          <table>
            <thead><tr><th>Measurement</th><th class="num">In</th><th class="num">MM</th></tr></thead>
            <tbody>{est_rows}</tbody>
          </table>
          <div class="cap">{table_caption}</div>
        </div>
        <div class="map-wrap">
          <div class="seclbl">Hail Footprint</div>
          <div style="margin-top:9px;"><img src="{footprint_src}" alt="Estimated hail footprint" /></div>
          <div class="cap">{footprint_caption}</div>
        </div>
      </div>

      <div class="conf">
        {chip_html}
        <div class="conf-body">
          <div class="seclbl">Corroboration &amp; Confidence</div>
          <p>{corrob}</p>
          {nearby_html}
        </div>
      </div>

      <div class="method">
        <div class="seclbl">Methodology</div>
        <p class="mesh-disc">{mesh_disclosure}</p>
        <p>{methodology}</p>
      </div>

      <div class="disc">
        <div class="dl">Disclaimer</div>
        <p>{disclaimer}</p>
      </div>

      <div class="spacer"></div>
    </div>

    <div class="foot">
      <span>Report {report_id} &middot; {version_line}</span>
      <span class="conf-tag">CONFIDENTIAL</span>
      <span>Page 1 of 1 &middot; Generated {generated_utc}</span>
    </div>

  </div>
</body>
</html>"""


def clip_text(text, limit):
    """Cap pathological input lengths so the one-page layout can never break.
    Normal addresses/claim refs are far below these caps; only torture-length
    strings lose a few trailing characters to an ellipsis."""
    t = str(text or "")
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "\u2026"


def soft_wrap_html(text, limit=58):
    """Break long text into <=limit-char lines joined with <br>.

    WeasyPrint 69's CSS grid sizes columns by their content, so one long
    unwrapped line (e.g. a very long property address) pushes sibling cells
    right off the page — CSS min-width/overflow-wrap does not save it. Breaking
    the line server-side keeps every renderer honest. Splits at spaces; hard-
    splits any single token longer than the limit.
    """
    words = str(text or "").split()
    lines, cur = [], ""
    for w in words:
        while len(w) > limit:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(w[:limit])
            w = w[limit:]
        if cur and len(cur) + 1 + len(w) > limit:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def build_report_html(data: dict, font_dir: str | None = None) -> str:
    """Build the complete, self-contained HTML for one hail report.

    Expected `data` keys:
      reportId, dateGenerated, generatedUtc, dateOfLoss, propertyAddress,
      claimRef, coordinates, contactUrl, contactCity, bandLabel, reportTitle,
      thresholdInches, versionLine,
      classification  -> the dict returned by classify_hail()
      coverage        -> the dict returned by assess_coverage()
      results         -> {cell, half, mile1, mile3, mile5}; each {in, mm} where
                         the values are floats OR None (None = no radar data)
      mapDataUri, mapCaption, methodologyText, disclaimerText,
      confidenceLevel, confidenceColor, confidenceNote, corroborationLine
    """
    font_face = _font_face_css(font_dir)

    cls = data.get("classification") or {}
    cov = data.get("coverage") or {}
    theme_key = cls.get("theme", "unknown")
    t = _THEMES.get(theme_key, _THEME_UNKNOWN)
    icon = {"detected": _ICON_TRIANGLE, "clear": _ICON_CHECK,
            "caution": _ICON_INFO, "unknown": _ICON_QUESTION}.get(theme_key, _ICON_QUESTION)

    threshold_in = float(data.get("thresholdInches", 0.75))
    res = data["results"]

    def num(v, dec=2):
        """Format a reading, or an em-dash when there is no radar data."""
        return f"{v:.{dec}f}" if v is not None else "\u2014"

    def phrase(d):
        if d.get("in") is None:
            return "no radar data"
        return f'{d["in"]:.2f}\u2033\u00a0({d["mm"]:.0f}\u00a0mm)'

    cell, half = res["cell"], res["half"]

    # ---- Key Finding readings line ---------------------------------------
    if cls.get("band") == "no_coverage":
        kf_readings = (f'Valid radar cells within {cov.get("radius_miles", 5):.0f} miles: '
                       f'<b>{cov.get("valid_cells", 0)} of {cov.get("total_cells", 0)}</b> '
                       f'({cov.get("valid_frac", 0.0) * 100:.0f}%).')
    else:
        kf_readings = (f'Peak within &frac12; mile: <b>{phrase(half)}</b> &nbsp;&middot;&nbsp; '
                       f'value at nearest grid cell: <b>{phrase(cell)}</b> &nbsp;&middot;&nbsp; '
                       f'client damage threshold: <b>{threshold_in:.2f}\u2033</b>.')
    kf_detail = cls.get("detail", "")

    # ---- Size table ------------------------------------------------------
    _rows = [("Value at nearest grid cell", cell, False),
             ("Peak within \u00bd mile", half, True),
             ("Peak within 1 mile", res["mile1"], False),
             ("Peak within 3 miles", res["mile3"], False),
             ("Peak within 5 miles", res["mile5"], False)]
    est_rows = "".join(
        '<tr class="{c}"><td>{l}</td><td class="num">{i}</td><td class="num">{m}</td></tr>'.format(
            c="hot" if hot else "", l=label,
            i=num(d.get("in")), m=num(d.get("mm"), 0))
        for label, d, hot in _rows)

    table_caption = data.get("tableCaption") or (
        "Each row is the PEAK radar-estimated diameter within that radius, not an "
        "average and not a measurement at the address. The MRMS grid is &asymp;1 km, "
        "so &lsquo;nearest grid cell&rsquo; already covers roughly a city block. "
        "&mdash; means no valid radar data in that footprint.")

    # ---- Confidence chip: suppressed entirely when coverage is unusable ---
    _conf_level = data.get("confidenceLevel") or ""
    if cls.get("band") == "no_coverage" or not _conf_level:
        chip_html = ""
        corrob = data.get("confidenceNote") or (
            "No confidence level is stated because radar coverage was insufficient "
            "at this location on this date.")
    else:
        chip_html = ('<div class="chip" style="background:{bg};">{txt}</div>'.format(
            bg=data.get("confidenceColor", "#5a6b7e"),
            txt=f"{_conf_level} Confidence"))
        corrob = data.get("confidenceNote") or ""
    _nearby = data.get("corroborationLine") or ""
    nearby_html = f"<p>{_nearby}</p>" if _nearby else ""

    methodology = data.get("methodologyText",
        "Hail-size estimates are derived from NOAA&rsquo;s Multi-Radar Multi-Sensor (MRMS) "
        "Maximum Estimated Size of Hail (MESH) product &mdash; a single-polarisation radar "
        "algorithm that infers in-storm hail growth from reflectivity above a modelled "
        "freezing level. This report reads the 24-hour maximum field (MESH_Max_1440min) "
        "for the local date of loss, converted from millimetres at 25.4 mm per inch.")
    disclaimer = data.get("disclaimerText",
        "This is a radar-derived estimate, not a guarantee of hail size or property damage, "
        "and is not a substitute for a physical inspection by a qualified professional. "
        "Clear Claims Co. makes no warranty and accepts no liability arising from use of this "
        "report. Source data is U.S. NOAA public-domain radar. Clear Claims Co. is an "
        "independent provider and is <strong style=\"color:#5a6b7e;\">not affiliated with "
        "Cotality or CoreLogic</strong>.")

    return _HAIL_REPORT_TEMPLATE.format(
        font_face=font_face,
        logo=_LOGO_SVG,
        contact_url=data.get("contactUrl", "clearclaimsco.co"),
        contact_city=data.get("contactCity", "support@clearclaimsco.co"),
        report_title=data.get("reportTitle", "Radar-Based Hail Estimate Report"),
        band_label=data.get("bandLabel", "Weather Analysis"),
        report_id=data["reportId"],
        report_date=data["dateGenerated"],
        generated_utc=data.get("generatedUtc", data["dateGenerated"]),
        version_line=data.get("versionLine", "methodology v2"),
        date_of_loss=data["dateOfLoss"],
        address=soft_wrap_html(clip_text(data["propertyAddress"], 110)),
        claim_ref=clip_text(data["claimRef"], 28),
        coords=data["coordinates"],
        kf_accent=t["main"], kf_bg=t["tint"], kf_bd=t["tintBorder"],
        icon=icon,
        verdict=cls.get("verdict", ""),
        kf_readings=kf_readings,
        kf_detail=kf_detail,
        badge=cls.get("badge", ""),
        est_rows=est_rows,
        table_caption=table_caption,
        footprint_src=data.get("mapDataUri", ""),
        footprint_caption=data.get(
            "mapCaption", "Estimated hail footprint &mdash; NOAA MRMS MESH."),
        chip_html=chip_html,
        corrob=corrob,
        nearby_html=nearby_html,
        mesh_disclosure=MESH_DISCLOSURE,
        methodology=methodology,
        disclaimer=disclaimer,
    )


def render_pdf_weasyprint(html: str, out_pdf: str) -> str:
    """Render the report HTML to a PDF using WeasyPrint (no browser required)."""
    from weasyprint import HTML
    HTML(string=html, base_url=".").write_pdf(out_pdf)
    return out_pdf
