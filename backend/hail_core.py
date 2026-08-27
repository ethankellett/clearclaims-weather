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
#  The old value here was 2014-07-01, which was simply wrong and made
#  validate_date_of_loss accept dates we cannot serve — the request then died
#  deep in the pipeline with a vague "try a nearby date". Checked against the
#  IEM mirror on 2026-08-27: 2018 and 2019 hold NO MESH product at all (only
#  precipitation), and although early/mid-2020 does carry MESH_Max_1440min it
#  is roughly TWO files for a whole day, which cannot support a defensible
#  24-hour maximum. So the honest floor is the AWS start date, and a request
#  before it is now refused up front with a clear reason.
IEM_ARCHIVE_START = ARCHIVE_START
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

#  Each peril's data reaches back a different distance (F3, 2026-08-27):
#    hail — MRMS MESH archive begins 2020-10-14 (verified: 2018/2019 hold no
#           MESH at all, and early 2020 has ~2 files/day, unusable).
#    wind — ASOS/AWOS station records via IEM reach back decades; LSR/SPC
#           corroboration is thin before the mid-2000s. Floor: 2005-01-01.
#    snow — SNODAS daily grids begin 2003-09-30. Floor: 2003-10-01.
PERIL_FLOORS = {
    "hail": (dt.date(2020, 10, 14), "national hail-radar (MRMS MESH) archive"),
    "wind": (dt.date(2005, 1, 1), "wind station and storm-report archive"),
    "snow": (dt.date(2003, 10, 1), "national snow analysis (SNODAS)"),
}


def validate_date_of_loss(date_of_loss: dt.date, today: dt.date | None = None,
                          peril: str = "hail") -> None:
    """Raise a friendly error if the date can't be served by the AWS archive.

    * Too early  -> before the 2020-10-14 archive start.
    * In future  -> obviously no radar exists yet.
    """
    today = today or dt.date.today()
    floor, label = PERIL_FLOORS.get(peril, PERIL_FLOORS["hail"])
    if date_of_loss < floor:
        raise ValueError(
            f"Date of loss {date_of_loss:%Y-%m-%d} is before the {label} begins "
            f"({floor:%B %d, %Y}). This date cannot be verified \u2014 the data does "
            f"not exist, rather than being temporarily unavailable."
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

#  How exact is the pin? A rooftop match and a town centroid are both "a
#  lat/lon", but only one of them justifies quoting a half-mile peak as though
#  it were the property. The report prints this so the reader can judge.
GEOCODE_PRECISION = {
    "rooftop": "Matched to a specific address point.",
    "interpolated": "Interpolated along an address range \u2014 typically accurate to "
                    "within a few hundred feet.",
    "street": "Matched to a street or road centreline, not a building.",
    "area": "Matched only to a town, ZIP or area centroid \u2014 the pin may be a "
            "long way from the actual property.",
    "manual": "Coordinates supplied directly.",
    "unknown": "Match precision not reported by the geocoder.",
}


def classify_nominatim_precision(top: dict) -> str:
    """Map a Nominatim result's class/type onto our precision buckets."""
    cls = (top.get("class") or "").lower()
    typ = (top.get("type") or "").lower()
    # Order matters: a highway can carry type "residential", which would
    # otherwise fall through to the rooftop branch below.
    if cls == "highway":
        return "street"
    if cls == "place" and typ in ("house", "building", "address"):
        return "rooftop"
    if cls == "building" or typ in ("house", "address", "yes"):
        return "rooftop"
    if cls == "place" and typ in ("city", "town", "village", "hamlet",
                                  "suburb", "postcode", "county", "state"):
        return "area"
    if cls == "boundary":
        return "area"
    return "unknown"


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
    side = ((best.get("tigerLine") or {}).get("side") or "").strip()
    # The Census "onelineaddress" locator interpolates along TIGER address
    # ranges; it does not return rooftop points.
    precision = "interpolated" if side else "unknown"
    return (float(c["y"]), float(c["x"]), best.get("matchedAddress", address),
            "U.S. Census", precision)


def geocode_nominatim(address: str, timeout: int = 30):
    """Geocode with OpenStreetMap Nominatim (broad coverage, incl. new homes).

    Returns (lat, lon, matched_address, 'OpenStreetMap') or None. Nominatim's
    usage policy requires a descriptive User-Agent and ≤1 request/second — both
    fine for this notebook's low volume.
    """
    import requests
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {"User-Agent": "ClearClaimsHailReport/1.0 (ops@clearclaimsco.co)"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or not data:
        return None
    top = data[0]
    return (float(top["lat"]), float(top["lon"]), top.get("display_name", address),
            "OpenStreetMap", classify_nominatim_precision(top))


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
                "source": "manual lat/long override",
                "precision": "manual",
                "precision_note": GEOCODE_PRECISION["manual"]}

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
    lat, lon, matched, provider, precision = result
    return {"lat": lat, "lon": lon, "label": matched,
            "source": f"{provider} geocoder",
            "precision": precision,
            "precision_note": GEOCODE_PRECISION.get(precision,
                                                    GEOCODE_PRECISION["unknown"])}


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
    """Pull the 24-hour-max MESH .grib2.gz URLs out of an IEM directory listing.

    IEM changed its filename prefix at some point: modern days are named
    MESH_Max_1440min_00.50_..., older ones MRMS_Max_1440min_00.50_.... The
    original pattern only accepted the first, so the whole IEM fallback matched
    ZERO files on older dates and silently reported "no radar files found".
    Verified 2026-08-27: 2023-07-11 has 48 MESH_-prefixed files; 2020-06-20 has
    2 MRMS_-prefixed ones.
    """
    names = re.findall(
        r'href="((?:MESH|MRMS)_Max_1440min_00\.50_\d{8}-\d{6}\.grib2\.gz)"', html)
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

def read_grib_field(path: str):
    """Read one MRMS GRIB2 file into plain numpy arrays (lats_1d, lons_1d, 2-D).

    Shared by the MESH and RQI readers. Handles the two MRMS quirks: the data
    variable is often named 'unknown', and longitudes come in the 0..360 range.
    Sentinel values (MRMS uses large negatives for missing/no-coverage) become
    NaN. What a NaN MEANS depends on the product and is the caller's business:
    for MESH it means "no hail diagnosed here"; for RQI it means "no radar
    quality value here", which really is an absence of coverage.
    """
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    var = "unknown" if "unknown" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var]

    lats = np.asarray(ds["latitude"].values, dtype="float64")
    lons = np.asarray(ds["longitude"].values, dtype="float64")
    arr = np.asarray(da.values, dtype="float32")
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.where((arr < 0) | (arr > 1000), np.nan, arr)
    ds.close()
    return lats, lons, arr


def read_mesh_grib(path: str):
    """MESH grid in MILLIMETRES (lats north->south, lons -180..180)."""
    return read_grib_field(path)


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

    IMPORTANT — what a NaN cell means here (corrected 2026-08-27 after the first
    live run against real MRMS data):

    MESH is a SPARSE field. The algorithm writes a value only where it diagnosed
    hail; everywhere else carries a missing sentinel that read_mesh_grib turns
    into NaN. So on an ordinary quiet day almost the whole grid is NaN. NaN
    therefore means "no hail signal in this cell", and reads as 0.00 in.

    It does NOT mean "the radar cannot see here". Nothing in the MESH field
    distinguishes a genuine radar gap from a hail-free cell — that needs the
    separate MRMS RadarQualityIndex product, which is wired up in PR 2. Until
    then the report must not claim to have verified coverage either way.

    `no_data` is True only when the ENTIRE cropped grid was unreadable, which is
    a real failure rather than a quiet day.
    """
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)
    finite = np.isfinite(mesh_mm)

    pi, pj = np.unravel_index(np.nanargmin(dist), dist.shape)
    p_mm = float(mesh_mm[pi, pj]) if finite[pi, pj] else 0.0
    out = {"point": {"mm": p_mm, "in": p_mm / MM_PER_INCH,
                     "lat": float(lats[pi]), "lon": float(lons[pj]),
                     "valid_cells": int(finite[pi, pj]), "total_cells": 1}}

    for r in rings:
        mask = dist <= r
        if not mask.any():
            mask = np.zeros_like(dist, dtype=bool)
            mask[pi, pj] = True
        valid = mask & finite
        n_total, n_valid = int(mask.sum()), int(valid.sum())
        if n_valid == 0:
            # No hail diagnosed anywhere in this footprint -> 0.00 in.
            out[r] = {"mm": 0.0, "in": 0.0, "lat": lat, "lon": lon,
                      "valid_cells": 0, "total_cells": n_total}
            continue
        vals = np.where(valid, mesh_mm, np.nan)
        mi, mj = np.unravel_index(np.nanargmax(vals), vals.shape)
        mm = float(mesh_mm[mi, mj])
        out[r] = {"mm": mm, "in": mm / MM_PER_INCH,
                  "lat": float(lats[mi]), "lon": float(lons[mj]),
                  "valid_cells": n_valid, "total_cells": n_total}
    return out


# =============================================================================
#  RADAR QUALITY INDEX (RQI)  — the real answer to "could the radar see here?"
# =============================================================================
#  Verified live on 2026-08-27 against s3://noaa-mrms-pds:
#      CONUS/RadarQualityIndex_00.00/YYYYMMDD/
#      MRMS_RadarQualityIndex_00.00_YYYYMMDD-HHMMSS.grib2.gz
#  2-minute cadence, ~650 KB gzipped, history back through at least 2023.
#
#  RQI is NOAA's own 0..1 index built from static terrain-blockage maps and
#  beam height relative to the freezing level. It is the field the MESH grid
#  cannot provide: MESH is sparse and says nothing where no hail fell, whereas
#  RQI is defined everywhere the radar network reaches. This is what lets the
#  report tell a genuine radar gap (Black Hills terrain blockage, say) apart
#  from a quiet day.
#
#  Honest caveat printed on the report: RQI is designed as a QPE (rainfall)
#  quality index. We use it as a coverage / blockage indicator, which is what
#  its blockage term measures, not as a hail-specific quality score.
# =============================================================================

RQI_PRODUCT = "RadarQualityIndex_00.00"
RQI_S3_PREFIX = f"{S3_BUCKET}/CONUS/{RQI_PRODUCT}"


def select_rqi_keys(fs, utc_start, utc_end, max_files=2):
    """Pick a few RQI files spread across the local day (UTC window).

    We deliberately sample rather than read all 720 daily files. The blockage
    term of RQI barely moves through a day, so a handful answers the coverage
    question; taking the MAX across samples asks "at its best, could the radar
    see this point today?", which is the fair test for a coverage gap.
    """
    keys = []
    for d in sorted({utc_start.date(), utc_end.date()}):
        folder = f"{RQI_S3_PREFIX}/{d:%Y%m%d}"
        try:
            keys.extend([k for k in fs.ls(folder) if k.endswith(".grib2.gz")])
        except FileNotFoundError:
            continue
    stamped = [(k, _parse_ts_from_key(k)) for k in keys]
    stamped = [(k, t) for k, t in stamped if t is not None and utc_start <= t <= utc_end]
    if not stamped:
        return []
    stamped.sort(key=lambda kt: kt[1])
    if len(stamped) > max_files:
        idx = np.linspace(0, len(stamped) - 1, max_files).round().astype(int)
        stamped = [stamped[i] for i in sorted(set(idx))]
    return [k for k, _ in stamped]


def fetch_rqi_at_point(utc_start, utc_end, lat, lon, tmpdir, max_files=None,
                       pad_deg=0.30):
    """Best-effort MAX RQI at/near the property across the local day.

    Returns {'value', 'n_files', 'source'} with value in 0..1, or value=None if
    RQI could not be obtained (pre-2020 dates served from the IEM mirror, a
    network hiccup, anything). A None here must degrade to "not assessed" on
    the report — never to a confident claim in either direction.
    """
    if max_files is None:
        try:
            max_files = max(1, int(os.environ.get("RQI_MAX_FILES", "2")))
        except ValueError:
            max_files = 2

    best = None
    n = 0
    try:
        fs = open_s3()
        keys = select_rqi_keys(fs, utc_start, utc_end, max_files)
        for k in keys:
            try:
                p = download_and_gunzip(fs, k, tmpdir)
                la, lo, arr = read_grib_field(p)
                cla, clo, carr = crop_to_bbox(la, lo, arr, lat, lon, pad_deg)
                del la, lo, arr
                LON, LAT = np.meshgrid(clo, cla)
                dist = haversine_miles(lat, lon, LAT, LON)
                pi, pj = np.unravel_index(np.nanargmin(dist), dist.shape)
                v = carr[pi, pj]
                if np.isfinite(v):
                    best = float(v) if best is None else max(best, float(v))
                n += 1
                del carr, cla, clo
                try:
                    os.remove(p)
                except OSError:
                    pass
                gc.collect()
            except Exception:
                continue
    except Exception:
        pass

    return {"value": best, "n_files": n,
            "source": (f"NOAA MRMS {RQI_PRODUCT} (AWS Open Data)" if n else None)}


#  Grades. RQI runs 0 (useless) to 1 (clean view). The cut points below are
#  ours, not NOAA's, and are deliberately conservative: we would rather say
#  "quality fair, treat with care" than imply a clean look the radar never had.
RQI_GRADES = (
    (0.80, "Excellent", "Clean radar view of this location."),
    (0.50, "Good", "Usable radar view of this location."),
    (0.20, "Fair", "Degraded radar view \u2014 partial beam blockage or long range."),
    (0.01, "Poor", "Severely degraded radar view \u2014 terrain blockage or extreme range."),
)


def grade_rqi(value):
    """Turn an RQI value into {grade, note, value}. None -> 'Not assessed'."""
    if value is None:
        return {"grade": "Not assessed", "value": None,
                "note": ("Radar coverage quality could not be retrieved for this date, "
                         "so it is not stated.")}
    for cut, grade, note in RQI_GRADES:
        if value >= cut:
            return {"grade": grade, "value": value, "note": note}
    return {"grade": "No coverage", "value": value,
            "note": ("NOAA's radar quality index is zero at this location on this date "
                     "\u2014 the radar network could not see this point.")}


# -----------------------------------------------------------------------------
#  RADAR COVERAGE STATE  (defect D1 — "no data" must never look like "no hail")
# -----------------------------------------------------------------------------

#  Fraction of cells inside COVERAGE_RADIUS_MI that must carry valid radar data
#  before we are willing to state a hail size at all. Tunable via env.
COVERAGE_RADIUS_MI = 5.0
COVERAGE_OK_FRAC = 0.80      # >= this -> "ok"
COVERAGE_MIN_FRAC = 0.50     # >= this -> "partial";  below -> "none"


def assess_coverage(rqi_grade: dict | None = None, hail_cells: int = 0,
                    total_cells: int = 0, radius_miles=COVERAGE_RADIUS_MI):
    """Decide the radar-coverage state from NOAA's RQI, not from the MESH field.

    Why not from MESH: MESH is sparse and writes nothing where no hail fell, so
    counting valid MESH cells measures "did hail happen nearby" — a different
    question entirely. That mistake shipped briefly on 2026-08-27 and made every
    quiet day read as a coverage failure. RQI is defined everywhere the radar
    network reaches, so it can actually answer this.

    States:
      'none'    - RQI is zero here: the radar network cannot see this point.
      'partial' - RQI is Poor: severe blockage or extreme range.
      'ok'      - RQI is Fair or better.
      'unknown' - RQI unavailable (e.g. a pre-2020 date served from the IEM
                  mirror). The report says coverage is unverified rather than
                  guessing.

    `hail_cells` / `total_cells` are carried for diagnostics only and must never
    drive this decision again.
    """
    g = (rqi_grade or {}).get("grade", "Not assessed")
    state = {"No coverage": "none", "Poor": "partial", "Fair": "ok",
             "Good": "ok", "Excellent": "ok"}.get(g, "unknown")
    return {"state": state, "hail_cells": hail_cells, "total_cells": total_cells,
            "radius_miles": radius_miles,
            "quality_grade": g,
            "quality_value": (rqi_grade or {}).get("value"),
            "quality_note": (rqi_grade or {}).get("note", ""),
            "quality_source": (rqi_grade or {}).get("source"),
            # legacy keys kept so nothing KeyErrors
            "valid_cells": hail_cells,
            "valid_frac": (hail_cells / total_cells) if total_cells else 0.0}


def count_hail_cells(lats, lons, mesh_mm, lat, lon, radius_miles=COVERAGE_RADIUS_MI):
    """Diagnostics only: how many cells within the radius carried a hail signal."""
    LON, LAT = np.meshgrid(lons, lats)
    dist = haversine_miles(lat, lon, LAT, LON)
    mask = dist <= radius_miles
    if not mask.any():
        return 0, 0
    return int((mask & np.isfinite(mesh_mm)).sum()), int(mask.sum())


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

    coverage_caveat = (
        " Radar coverage quality could not be retrieved for this date, so a "
        "coverage gap cannot be fully ruled out.")

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
        verdict = "Radar shows no hail signature at or near this property on this date."
        detail = ("A radar-based negative is weaker evidence than a positive. "
                  "Small or brief hail can fall below what the radar resolves."
                  + (coverage_caveat if coverage_state == "unknown" else "")
                  + (" NOAA's radar quality index shows only partial coverage at "
                     "this location, which weakens this negative further."
                     if coverage_state == "partial" else ""))
        likelihood = "No indication"
    elif peak_in < MESH_DISCRIMINATION_FLOOR_IN:
        band, theme = "trace", "caution"
        badge = "Trace / Indeterminate"
        verdict = "Damaging hail is UNLIKELY to have occurred near this property."
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


def spc_row_time_utc(hhmm: str, convective_day):
    """Turn an SPC 'Time' cell (HHMM, UTC) into a datetime on the right day.

    SPC convective day D covers 12Z on D through 12Z on D+1, so an hour before
    12 belongs to the NEXT calendar day. Returns None if unparseable.
    """
    try:
        hhmm = (hhmm or "").strip()
        if len(hhmm) != 4 or not hhmm.isdigit():
            return None
        hh, mm = int(hhmm[:2]), int(hhmm[2:])
        day = convective_day + (dt.timedelta(days=1) if hh < 12 else dt.timedelta(0))
        return dt.datetime(day.year, day.month, day.day, hh, mm,
                           tzinfo=dt.timezone.utc)
    except Exception:
        return None


def parse_spc_hail_csv(text: str, lat: float, lon: float, radius_miles: float,
                       convective_day=None, utc_start=None, utc_end=None):
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
        if d > radius_miles:
            continue
        # Keep only reports that actually fall inside the property's local day.
        t = spc_row_time_utc(row.get("Time", ""), convective_day) if convective_day else None
        if t is not None and utc_start is not None and utc_end is not None:
            if not (utc_start <= t <= utc_end):
                continue
        out.append({
            "source": "SPC", "size_in": size_in,
            "lat": rlat, "lon": rlon, "dist_mi": d,
            "dir": compass_bearing(lat, lon, rlat, rlon),
            "time": (t.strftime("%Y-%m-%dT%H:%MZ") if t else row.get("Time", "")),
            "city": row.get("Location", ""),
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

    # SPC daily hail CSV. DEFECT D2: an SPC "daily" file runs 12Z -> 12Z, NOT
    # local midnight to midnight. A 2 a.m. local event on the date of loss lives
    # in the PREVIOUS day's file. Fetch both days and let the de-dup sort it out.
    for d in (date_of_loss - dt.timedelta(days=1), date_of_loss):
        try:
            url = f"https://www.spc.noaa.gov/climo/reports/{d:%y%m%d}_rpts_hail.csv"
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "Lat" in r.text[:200]:
                reports += parse_spc_hail_csv(r.text, lat, lon, radius_miles,
                                              convective_day=d,
                                              utc_start=utc_start, utc_end=utc_end)
        except Exception:
            continue

    # De-duplicate near-identical reports (same rounded spot + size).
    seen, deduped = set(), []
    for rep in sorted(reports, key=lambda x: x["dist_mi"]):
        key = (round(rep["lat"], 2), round(rep["lon"], 2),
               round(rep["size_in"] or 0, 2))
        if key not in seen:
            seen.add(key); deduped.append(rep)
    return deduped


#  Confidence matrix (defect D7, completed in PR 2 now that radar quality is a
#  real measured value rather than a guess).
#
#  Two asymmetries are deliberate and both are defensible in a deposition:
#    * A radar POSITIVE corroborated by ground reports is the strongest result
#      the system can produce.
#    * A radar NEGATIVE is inherently weaker than a positive, because brief or
#      small hail can fall below what the radar resolves. A negative can only
#      reach High when the radar demonstrably had a clean view (RQI Excellent).
_QUALITY_RANK = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1,
                 "No coverage": 0, "Not assessed": None}


def assess_confidence(point_in, ring_max_in, reports, threshold_in, source=None,
                      coverage_state="unknown", quality_grade="Not assessed",
                      n_warnings=0):
    """Combine radar reading, radar QUALITY and ground reports into a level.

    Returns {level, color, note, n_reports}. level is None when coverage is
    missing entirely — in that case the report shows no confidence chip at all,
    because a confident verdict over absent data is exactly the failure this
    whole rebuild exists to remove.
    """
    if coverage_state == "none" or point_in is None:
        return {"level": None, "color": None, "n_reports": len(reports),
                "note": ("NOAA's radar quality index shows no usable radar coverage "
                         "at this location on this date, so no confidence level is "
                         "stated. This is an absence of data, not evidence that hail "
                         "did not occur.")}

    rank = _QUALITY_RANK.get(quality_grade)
    radar_max = max(point_in, ring_max_in if ring_max_in is not None else point_in)
    detected = point_in >= threshold_in
    n = len(reports)
    biggest = max([r["size_in"] for r in reports if r["size_in"]], default=0.0)
    qual_txt = (f" Radar quality at this location was {quality_grade.lower()}."
                if rank is not None else
                " Radar coverage quality could not be retrieved for this date.")

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
                    f"report(s) were logged nearby \u2014 verify exact timing and "
                    f"location.")
        elif rank is not None and rank >= 4:
            # Only a demonstrably clean radar view earns a confident negative.
            level = "High"
            note = ("Radar shows no significant hail at the property, no ground "
                    "reports were logged nearby, and NOAA's radar quality index "
                    "confirms a clean view of this location.")
        else:
            level = "Moderate"
            note = ("Radar shows no significant hail at the property and no ground "
                    "reports were logged nearby. A radar-based negative is weaker "
                    "evidence than a positive \u2014 brief or small hail can fall "
                    "below what the radar resolves.")

    # Quality adjustments apply to conclusions that REST on the radar reading.
    # A positive corroborated by independent ground observers does not: those
    # reports are evidence in their own right, and the radar's view of the
    # property is no longer what the finding depends on. Exempting that case
    # keeps the strongest result the system can produce from being watered down
    # by a missing quality value.
    corroborated_positive = detected and n >= 1
    if corroborated_positive:
        if rank is not None and rank <= 1:
            level = "Moderate"
            note += (" NOAA's radar quality index shows severely degraded radar "
                     "coverage here, so this rests mainly on the ground reports.")
        color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[level]
        return {"level": level, "color": color, "note": note, "n_reports": n}

    # Degraded radar quality pulls confidence down whatever the reading says.
    if rank is not None and rank <= 1:
        level = {"High": "Low", "Moderate": "Low", "Low": "Low"}[level]
        note += (" NOAA's radar quality index shows severely degraded coverage here, "
                 "so confidence has been reduced.")
    elif rank == 2:
        level = {"High": "Moderate", "Moderate": "Moderate", "Low": "Low"}[level]
        note += " Radar quality at this location was only fair."
    elif rank is None:
        level = {"High": "Moderate", "Moderate": "Moderate", "Low": "Low"}[level]
        note += qual_txt

    # F8 (found on Ethan's first real report): a severe thunderstorm or tornado
    # warning was in force at this property on the date of loss. That is a human
    # forecaster's real-time judgement that this county was in danger. It does
    # not put hail at the address \u2014 but a NEGATIVE should not claim High
    # confidence while page 2 of the same report shows a WARNED badge.
    if not detected and n_warnings and level == "High":
        level = "Moderate"
        note += (f" Note: {n_warnings} severe weather warning(s) covered this "
                 f"property on the date of loss, so severe weather was in the "
                 f"area even though radar shows no hail at this location.")

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
  .meta .val {{ font-size: 13.5px; font-weight: 600; color: #152742; margin-top: 3px; }}
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
  .kf-note {{ font-size: 10px; color: #6b7d94; line-height: 1.34; margin-top: 4px; }}
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
  .method p {{ font-size: 10.5px; color: #4a5d76; line-height: 1.38; margin-top: 3px; }}
  .method p.mesh-disc {{ color: #152742; font-weight: 600; }}

  .disc {{ margin-top: 5px; background: #f0f4f8; border-radius: 8px; padding: 6px 16px; margin-bottom: 44px; }}
  .disc .dl {{ font-size: 9px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #8a99ab; margin-bottom: 5px; }}
  .disc p {{ font-size: 8.5px; color: #8a99ab; line-height: 1.38; }}
  .disc b, .disc strong {{ color: #5a6b7e; }}

  .spacer {{ display: none; }}

  .foot {{ background: #06101f; padding: 13px 40px; display: flex; align-items: center; justify-content: space-between; position: absolute; left: 0; right: 0; bottom: 0; }}
  .foot span {{ font-size: 10px; color: #8fa3b8; letter-spacing: .03em; }}
  .foot .conf-tag {{ color: #4a9af5; font-weight: 700; letter-spacing: .18em; }}

  /* ---- page 2: storm history & context ---- */
  .ctx {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 9px; }}
  .ctx .card {{ border: 1px solid #dde6f0; border-radius: 9px; padding: 11px 16px; }}
  .ctx .card .lbl {{ font-size: 9px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #5a6b7e; }}
  .ctx .card .big {{ font-family: 'DM Serif Display', serif; font-size: 21px; color: #06101f; margin-top: 5px; line-height: 1.1; }}
  .ctx .card .sub {{ font-size: 11px; color: #4a5d76; line-height: 1.45; margin-top: 5px; }}
  .pill {{ display: inline-block; color: #fff; font-size: 10px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 6px; }}

  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 9px; }}
  .stats .s {{ background: #f4f8fc; border-radius: 8px; padding: 9px 12px; }}
  .stats .s .k {{ font-size: 8.5px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: #5a6b7e; line-height: 1.3; }}
  .stats .s .v {{ font-family: 'DM Serif Display', serif; font-size: 19px; color: #06101f; margin-top: 4px; line-height: 1; }}
  .stats .s .u {{ font-size: 10px; color: #8a99ab; margin-top: 3px; }}

  .note {{ font-size: 9.5px; color: #8a99ab; line-height: 1.45; margin-top: 8px; }}
  .note b {{ color: #5a6b7e; }}
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
        <div class="cell row-last"><div class="lbl">Radar Coverage Quality</div><div class="val">{radar_quality}</div></div>
        <div class="cell c3 row-last"><div class="lbl">Address Match</div><div class="val">{geocode_quality}</div></div>
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
      <span>{page_label} &middot; Generated {generated_utc}</span>
    </div>

  </div>
{page2}
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



#  Cap on printed history rows; the rest are summarised as "+N more" so the
#  supplement can never run to a third page.
MAX_PRINTED_EVENTS = 12


def build_context_page(data: dict) -> str:
    """Build the optional page 2: same-day context + 24-month hail history.

    Returns "" when no context was gathered, so a report can still be a clean
    single page if the lookups are switched off or all fail.
    """
    ctx = data.get("context") or {}
    if not ctx:
        return ""

    prior = ctx.get("prior_hail") or {}
    warn = ctx.get("warnings") or {}
    wind = ctx.get("wind") or {}
    summ = prior.get("summary") or {}

    # Times on this card are shown in the PROPERTY's local time (F6): a bare
    # "02:42 UTC" reads like the middle of the night when it was actually
    # 8:42 PM the previous evening locally.
    tz_name = data.get("tzName") or "UTC"
    try:
        from zoneinfo import ZoneInfo
        _tz = ZoneInfo(tz_name)
    except Exception:
        _tz = dt.timezone.utc

    def _local(t):
        try:
            lt = t.astimezone(_tz)
            return f'{lt:%b %d, %I:%M %p} {lt.tzname() or ""}'.replace(" 0", " ")
        except Exception:
            return f"{t:%H:%M} UTC"

    # ---- same-day: NWS warnings -----------------------------------------
    if not warn.get("ok"):
        w_pill = '<span class="pill" style="background:#5a6b7e;">Unavailable</span>'
        w_big = "Not retrieved"
        w_sub = "The NWS warning record could not be retrieved for this date."
    elif warn.get("warned"):
        names = []
        for x in warn["warnings"][:3]:
            names.append(f'{x["name"]} &middot; issued {_local(x["issued"])}')
        extra = (f' &nbsp;+{len(warn["warnings"]) - 3} more'
                 if len(warn["warnings"]) > 3 else "")
        w_pill = '<span class="pill" style="background:#d94f3d;">Warned</span>'
        w_big = f'{len(warn["warnings"])} severe warning(s)'
        w_sub = "; ".join(names) + extra + (
            f' &mdash; issued by NWS {warn["warnings"][0]["wfo"]}.')
    else:
        n_watch = len(warn.get("watches") or [])
        w_pill = '<span class="pill" style="background:#28a678;">Not warned</span>'
        w_big = "No severe warning"
        w_sub = ("No severe thunderstorm or tornado warning covered this property "
                 "on the date of loss."
                 + (f" A watch was in effect ({n_watch})." if n_watch else ""))

    # ---- same-day: measured wind ----------------------------------------
    if not wind.get("ok") or wind.get("peak_mph") is None:
        wind_big = "No measurement"
        wind_sub = ("No nearby ASOS/AWOS station reported a gust for this date. "
                    "Downbursts are small and routinely miss every station in a county.")
    else:
        st = wind.get("station") or "nearest station"
        dm = wind.get("dist_mi")
        wind_big = f'{wind["peak_mph"]:.0f} mph'
        wind_sub = (f'Peak measured gust at {st}'
                    + (f', {dm:.1f} mi away' if dm is not None else "")
                    + ". This is an instrument reading, not a radar estimate \u2014 but a "
                      "downburst can miss the station entirely.")

    # ---- prior hail summary stats ---------------------------------------
    inner = summ.get("inner_mi", 5)
    outer = summ.get("outer_mi", 10)
    if not prior.get("ok"):
        stats_html = ('<div class="note">The prior-hail record could not be retrieved '
                      'for this property. This section is unavailable, which is not the '
                      'same as a clean history.</div>')
        rows_html = ""
        cap = ""
    else:
        largest = summ.get("largest_in")
        ds = summ.get("days_since_severe")
        # Ordered to answer the question people actually ask first: when did it
        # last hail near here, and how close.
        _la_date = summ.get("last_any_date")
        _la_dist, _la_dir = summ.get("last_any_dist_mi"), summ.get("last_any_dir")
        _ls_date = summ.get("last_severe_date")
        stats = [
            ("Most recent hail<br>within %g mi" % inner,
             (_la_date or "None"),
             (f'{summ["last_any_size_in"]:.2f}\u2033 &middot; {_la_dist:.1f} mi {_la_dir}'
              if _la_date and _la_dist is not None else f"in {summ.get('months', 24)} months")),
            ("Most recent<br>&ge;1.00\u2033 within %g mi" % inner,
             (_ls_date or "None"),
             (f"{ds} days before loss" if ds is not None else "none in window")),
            ("Largest report<br>within %g mi" % inner,
             f'{largest:.2f}\u2033' if largest else "None",
             (summ.get("largest_date") or "in window") if largest else f"in {summ.get('months', 24)} months"),
            ("Reports &ge; %.2f\u2033 within<br>%g mi / %g mi" % (summ.get("min_size_in", 0.75), inner, outer),
             f'{summ.get("count_inner", 0)} / {summ.get("count_outer", 0)}', "events"),
        ]
        stats_html = '<div class="stats">' + "".join(
            f'<div class="s"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="u">{u}</div></div>' for k, v, u in stats) + '</div>'

        # Sort MOST RECENT FIRST for display. The underlying list is ordered
        # largest-first (that drives the "largest" stat), but a size-ordered
        # table buries the thing a reader wants: when it last hailed here. On a
        # big hail day dozens of spotters call in, so a size sort can fill the
        # whole table with one date and hide everything since.
        evs = sorted(prior.get("events") or [],
                     key=lambda e: (e.get("date", ""), -e.get("dist_mi", 0)),
                     reverse=True)
        shown = evs[:MAX_PRINTED_EVENTS]
        if shown:
            body = "".join(
                '<tr><td>{d}</td><td class="num">{s:.2f}</td><td class="num">{km:.1f} mi {dir}</td>'
                '<td>{c}</td></tr>'.format(
                    d=e["date"], s=e["size_in"], km=e["dist_mi"], dir=e["dir"],
                    c=clip_text(e.get("city") or e.get("county") or "", 30))
                for e in shown)
            # Keep the overflow line OUTSIDE the table: as a colspan row inside
            # <tbody> WeasyPrint floated it past the caption instead of keeping
            # it with its rows.
            more = (f'<div class="note" style="margin-top:5px;">'
                    f'+{len(evs) - len(shown)} further report(s) in this window are '
                    f'not listed.</div>'
                    if len(evs) > len(shown) else "")
            rows_html = (
                '<table><thead><tr><th>Date</th><th class="num">Size (in)</th>'
                '<th class="num">From this property</th>'
                '<th>How the NWS logged the location</th>'
                '</tr></thead><tbody>' + body + '</tbody></table>' + more
                + '<div class="note" style="margin-top:5px;">Newest first. '
                  '<b>From this property</b> is measured from the address on page 1. '
                  'The final column is the National Weather Service&rsquo;s own wording for '
                  'where the spotter was, given relative to a nearby landmark &mdash; not '
                  'relative to this property. A single storm often produces many reports '
                  'on the same date from different spotters.</div>')
        else:
            rows_html = ('<div class="note">No official hail report of '
                         f'{summ.get("min_size_in", 0.75):.2f}\u2033 or greater was logged '
                         f'within {outer:g} miles in this window.</div>')

        below = summ.get("n_below_cutoff", 0)
        cap = ('<div class="note"><b>This is history, not evidence of damage to this '
               'roof.</b> Prior hail nearby does not establish that this property was '
               'struck, and does not establish that any damage predates the loss. '
               'Reports are what a person observed and called in: nobody logs hail where '
               'nobody lives, so a quiet record in open country is not proof that nothing '
               'fell. Source: NWS Local Storm Reports via Iowa Environmental Mesonet'
               + (f'; {below} further report(s) below the '
                  f'{summ.get("min_size_in", 0.75):.2f}\u2033 display cutoff are not shown'
                  if below else "") + '.</div>')

    win = prior.get("window") or {}
    win_txt = (f'{win["start"]:%b %Y} \u2013 {win["end"]:%b %Y}'
               if win.get("start") else "previous 24 months")

    return f'''
    <div class="page">
      <div class="hdr">
        <div class="brand">
          {_LOGO_SVG}
          <div>
            <div class="wm">Clear <span class="b">Claims</span> <span class="co">Co.</span></div>
            <div class="tag">Fairness in every claim</div>
          </div>
        </div>
        <div class="hdr-right">
          <div class="site">{data.get("contactUrl", "clearclaimsco.co")}</div>
          <div class="loc">Report {data["reportId"]}</div>
        </div>
      </div>

      <div class="titleband">
        <h1>Storm History &amp; Context</h1>
        <div class="kick">Supplement</div>
      </div>

      <div class="body">
        <div class="seclbl">Date of Loss &mdash; Independent Context</div>
        <div class="ctx">
          <div class="card">
            <div class="lbl">NWS Warnings at This Property</div>
            <div class="big">{w_big}</div>
            <div style="margin-top:7px;">{w_pill}</div>
            <div class="sub">{w_sub}</div>
          </div>
          <div class="card">
            <div class="lbl">Peak Measured Wind Gust</div>
            <div class="big">{wind_big}</div>
            <div class="sub">{wind_sub}</div>
          </div>
        </div>

        <div style="margin-top:14px;" class="seclbl">
          Prior Hail Near This Property &mdash; {win_txt}
        </div>
        {stats_html}
        {rows_html}
        {cap}
      </div>

      <div class="foot">
        <span>Report {data["reportId"]} &middot; {data.get("versionLine", "")}</span>
        <span class="conf-tag">CONFIDENTIAL</span>
        <span>Page 2 of 2 &middot; Generated {data.get("generatedUtc", "")}</span>
      </div>
    </div>'''


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

    no_cov = (cls.get("band") == "no_coverage")

    def num(v, dec=2):
        """Format a reading \u2014 an em-dash when there is no radar data.

        F2: when NOAA's radar quality index says the radar could not see this
        point, the sampled values are meaningless and the verdict refuses to
        state a size. The table must refuse too: printing 0.00 next to
        "no size can be stated" is a contradiction on one page.
        """
        if no_cov or v is None:
            return "\u2014"
        return f"{v:.{dec}f}"

    def phrase(d):
        if d.get("in") is None:
            return "no radar data"
        return f'{d["in"]:.2f}\u2033\u00a0({d["mm"]:.0f}\u00a0mm)'

    cell, half = res["cell"], res["half"]

    # ---- Key Finding readings line ---------------------------------------
    if cls.get("band") == "no_coverage":
        kf_readings = ("The radar grid for this date could not be read at this "
                       "location, so no hail size is stated.")
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

    if no_cov:
        table_caption = data.get("tableCaption") or (
            "No values are stated because NOAA&rsquo;s radar quality index shows the "
            "radar network could not see this location on this date. An absence of "
            "data is not an absence of hail.")
    else:
        table_caption = data.get("tableCaption") or (
            "Each row is the PEAK radar-estimated diameter within that radius \u2014 not an "
            "average, and not a measurement at the address. The MRMS grid is &asymp;1 km, so "
            "&lsquo;nearest grid cell&rsquo; already covers roughly a city block. A larger "
            "radius can only ever report the same value or a bigger one.")

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

    _method_default = (
        "Hail-size estimates are derived from NOAA&rsquo;s Multi-Radar Multi-Sensor (MRMS) "
        "Maximum Estimated Size of Hail (MESH) product &mdash; a single-polarisation radar "
        "algorithm that infers in-storm hail growth from reflectivity above a modelled "
        "freezing level. This report reads the 24-hour maximum field (MESH_Max_1440min) "
        "covering the property&rsquo;s local calendar day. It is a rolling 24-hour "
        "maximum sampled up to three hours past local midnight, so hail in the early "
        "hours of the next morning can contribute. Radar coverage quality is NOAA&rsquo;s "
        "Radar Quality Index (0&ndash;1, from terrain-blockage maps and beam height) &mdash; "
        "an indicator of whether the radar could see this point, not a hail-specific score.")
    methodology = data.get("methodologyText", _method_default)
    disclaimer = data.get("disclaimerText",
        "This is a radar-derived estimate, not a guarantee of hail size or property damage, "
        "and is not a substitute for a physical inspection by a qualified professional. "
        "Clear Claims Co. makes no warranty and accepts no liability arising from use of this "
        "report. Source data is U.S. NOAA public-domain radar. Clear Claims Co. is an "
        "independent provider and is <strong style=\"color:#5a6b7e;\">not affiliated with "
        "Cotality or CoreLogic</strong>.")

    page2 = build_context_page(data)
    return _HAIL_REPORT_TEMPLATE.format(
        page2=page2,
        page_label=("Page 1 of 2" if page2 else "Page 1 of 1"),
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
        radar_quality=data.get("radarQuality", "Not assessed"),
        geocode_quality=data.get("geocodeQuality", "Unknown"),
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
