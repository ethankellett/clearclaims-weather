# =============================================================================
#  ClearClaims — SNOW verification engine
#  Reports, at/near the property on the date of loss:
#    * Snow depth          (SNODAS gridded, NOAA NOHRSC)
#    * Snow-water-equivalent (SWE) → ROOF LOAD in psf  (collapse-relevant)
#    * New snowfall         (nearest COOP/CoCoRaHS station, direct measurement)
#  Detection: snow depth >= threshold (default 6 in) = "significant accumulation".
#
#  Reuses hail_core (geocode, UTC window, distance, fonts, render) and
#  peril_report (shared one-page template).
#
#  SNODAS is a 30-arc-second CONUS binary grid. The parser/indexing/sampling and
#  the roof-load math + rendering are unit-tested against a synthetic file in the
#  same format. The live download (NOHRSC) + station fetch run on the open
#  internet — confirm them on the first Colab run.
# =============================================================================

from __future__ import annotations
import datetime as dt
import gzip
import io
import math
import tarfile
import numpy as np

import hail_core as hc
import peril_report

MM_PER_IN = 25.4
KGM2_TO_PSF = 0.2048161             # 1 kg/m² (=1 mm SWE) of load in lb/ft²
DEFAULT_DEPTH_THRESHOLD_IN = 6.0

# SNODAS masked-CONUS georeference (cell CENTERS, NW-origin row order).
SNODAS = {
    "ncols": 6935, "nrows": 3351,
    "xll": -124.729583333,    # west edge center
    "yll": 24.949583333,      # south edge center
    "cell": 0.008333333333,   # ~30 arc-seconds
    "nodata": -9999,
}
# SNODAS product codes within the daily tar (filename contains these):
SNODAS_DEPTH = "1036"   # snow depth (mm, scale 1000 → integer mm)
SNODAS_SWE = "1034"     # snow water equivalent (mm)


# ---- binary grid parsing + indexing --------------------------------------
def read_snodas_grid(raw: bytes, nrows=None, ncols=None) -> np.ndarray:
    """Parse a SNODAS .dat byte string → 2-D int16 array (big-endian, NW origin).

    Values are millimetres; nodata = -9999.
    """
    nrows = nrows or SNODAS["nrows"]
    ncols = ncols or SNODAS["ncols"]
    arr = np.frombuffer(raw, dtype=">i2")
    if arr.size != nrows * ncols:
        raise ValueError(f"SNODAS grid size {arr.size} != {nrows*ncols} "
                         f"(expected {nrows}x{ncols}).")
    return arr.reshape(nrows, ncols)


def snodas_rowcol(lat, lon, geo=SNODAS):
    """Map lat/lon → (row, col) in a SNODAS grid (row 0 = north)."""
    yur = geo["yll"] + (geo["nrows"] - 1) * geo["cell"]   # north edge center
    col = int(round((lon - geo["xll"]) / geo["cell"]))
    row = int(round((yur - lat) / geo["cell"]))
    return row, col


def _value_mm(arr, row, col, geo=SNODAS):
    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
        v = int(arr[row, col])
        return np.nan if v == geo["nodata"] else float(v)
    return np.nan


def sample_snodas(arr, lat, lon, geo=SNODAS, radius_miles=1.0, agg="point"):
    """Value (mm) at the property cell, or the MAX within radius_miles."""
    r0, c0 = snodas_rowcol(lat, lon, geo)
    if agg == "point":
        return _value_mm(arr, r0, c0, geo)
    # max within radius: cells span ~ radius/ (miles per cell)
    miles_per_deg = 69.0
    cell_mi = geo["cell"] * miles_per_deg
    rad_cells = max(1, int(round(radius_miles / max(cell_mi, 1e-6))))
    vals = []
    for dr in range(-rad_cells, rad_cells + 1):
        for dc in range(-rad_cells, rad_cells + 1):
            v = _value_mm(arr, r0 + dr, c0 + dc, geo)
            if not np.isnan(v):
                vals.append(v)
    return max(vals) if vals else np.nan


def swe_to_load_psf(swe_mm: float) -> float:
    """Snow-water-equivalent (mm) → roof load (lb/ft²). 1 mm SWE = 1 kg/m²."""
    if swe_mm is None or np.isnan(swe_mm):
        return 0.0
    return round(swe_mm * KGM2_TO_PSF, 1)


# ---- SNODAS fetch (best-effort; live only) --------------------------------
def fetch_snodas_product(date_of_loss, product_code, tmpdir, timeout=120):
    """Download the SNODAS daily tar from NOHRSC, extract one product's .dat.

    Returns a 2-D int16 array, or raises on failure. (Live network — Colab/Render.)
    """
    import os
    import requests
    mon = f"{date_of_loss:%m_%b}"
    base = (f"https://noaadata.apps.nsidc.org/NOAA/G02158/masked/"
            f"{date_of_loss:%Y}/{mon}/SNODAS_{date_of_loss:%Y%m%d}.tar")
    r = requests.get(base, timeout=timeout)
    r.raise_for_status()
    tf = tarfile.open(fileobj=io.BytesIO(r.content))
    member = None
    for m in tf.getmembers():
        if product_code in m.name and m.name.endswith(".dat.gz"):
            member = m
            break
    if member is None:
        raise ValueError(f"Product {product_code} not found in SNODAS tar.")
    raw = gzip.decompress(tf.extractfile(member).read())
    return read_snodas_grid(raw)


# ---- station snowfall (corroboration) -------------------------------------
def parse_iem_daily_snow(obj: dict, lat, lon, radius_miles):
    """Parse IEM daily-climate GeoJSON for station snowfall (inches) near point."""
    out = []
    for feat in (obj or {}).get("features", []):
        p = feat.get("properties", {}) or {}
        snow = p.get("snow")
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        slon, slat = coords[0], coords[1]
        if slat is None or snow in (None, "", "M"):
            continue
        try:
            snow_in = float(snow)
        except (TypeError, ValueError):
            continue
        d = float(hc.haversine_miles(lat, lon, slat, slon))
        if d <= radius_miles:
            out.append({"source": "COOP/CoCoRaHS", "snow_in": snow_in,
                        "lat": float(slat), "lon": float(slon), "dist_mi": d,
                        "dir": hc.compass_bearing(lat, lon, slat, slon),
                        "name": p.get("name", p.get("station", ""))})
    out.sort(key=lambda s: s["dist_mi"])
    return out


def fetch_station_snowfall(lat, lon, date_of_loss, radius_miles=25.0, timeout=30):
    """Best-effort nearby station 24-h snowfall from IEM daily data. Never raises."""
    import requests
    try:
        pad = 0.5
        params = {"date": f"{date_of_loss:%Y-%m-%d}",
                  "west": lon - pad, "east": lon + pad, "south": lat - pad, "north": lat + pad}
        r = requests.get("https://mesonet.agron.iastate.edu/geojson/climodat_dvp.geojson",
                         params=params, timeout=timeout)
        return parse_iem_daily_snow(r.json(), lat, lon, radius_miles)
    except Exception:
        return []


# ---- footprint map (gridded snow depth) -----------------------------------
def make_snow_map(arr_depth_mm, lat, lon, geo, out_png, pad_deg=0.45, brand=None):
    """Crop SNODAS depth around the point and draw a snow-depth footprint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

    r0, c0 = snodas_rowcol(lat, lon, geo)
    dcell = max(1, int(round(pad_deg / geo["cell"])))
    r1, r2 = max(0, r0 - dcell), min(arr_depth_mm.shape[0], r0 + dcell)
    c1, c2 = max(0, c0 - dcell), min(arr_depth_mm.shape[1], c0 + dcell)
    sub = arr_depth_mm[r1:r2, c1:c2].astype("float32")
    sub = np.where(sub == geo["nodata"], np.nan, sub) / MM_PER_IN   # inches
    yur = geo["yll"] + (geo["nrows"] - 1) * geo["cell"]
    lats = yur - np.arange(r1, r2) * geo["cell"]
    lons = geo["xll"] + np.arange(c1, c2) * geo["cell"]

    cmap = LinearSegmentedColormap.from_list("snow", ["#dbe7f2", "#9ec6e6", "#5a9bd4",
                                                      "#3b6fb0", "#2b4a8a", "#5a3b8a"]).copy()
    cmap.set_bad(alpha=0.0)
    levels = [0.5, 1, 2, 4, 6, 9, 12, 18, 24]
    norm = BoundaryNorm(levels, cmap.N, extend="max")
    plot = np.where(sub < 0.5, np.nan, sub)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=160)
    ax.set_facecolor("#f3f7fb")
    pcm = ax.pcolormesh(lons, lats, plot, cmap=cmap, norm=norm, shading="auto")
    ax.plot(lon, lat, marker="o", markersize=10, markerfacecolor="#06101f",
            markeredgecolor="white", markeredgewidth=1.6, zorder=6)
    ax.set_xlim(lons.min(), lons.max()); ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Snow depth footprint", fontsize=9, color="#06101f")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, extend="max")
    cb.set_label("Snow depth (inches)", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return out_png


# ---- confidence -----------------------------------------------------------
def assess_snow_confidence(depth_in, station_reports, threshold_in):
    detected = depth_in is not None and depth_in >= threshold_in
    n = len(station_reports)
    if detected:
        if n >= 1:
            lvl, note = "High", (f"Gridded snow depth is corroborated by {n} nearby "
                                 f"station snowfall report(s).")
        else:
            lvl, note = "Moderate", ("Gridded snow depth meets the threshold, but no "
                                     "nearby station report was available to corroborate.")
    else:
        if n >= 1 and any(s["snow_in"] >= threshold_in for s in station_reports):
            lvl, note = "Low", ("Gridded depth at the property is below threshold, but a "
                                "nearby station reported significant snowfall — verify location.")
        else:
            lvl, note = "High", ("Gridded snow depth is below the threshold and nearby "
                                 "stations agree.")
    color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[lvl]
    return {"level": lvl, "color": color, "note": note, "n_reports": n}


# ---- assemble report ------------------------------------------------------
def build_snow_report_data(*, report_id, address_label, lat, lon, date_of_loss,
                           contact_url, contact_city, claim_ref, threshold_in,
                           depth_mm, swe_mm, station_reports, map_data_uri, generated=None):
    generated = generated or dt.datetime.now(dt.timezone.utc)
    depth_in = (depth_mm / MM_PER_IN) if depth_mm and not np.isnan(depth_mm) else 0.0
    swe_in = (swe_mm / MM_PER_IN) if swe_mm and not np.isnan(swe_mm) else 0.0
    load_psf = swe_to_load_psf(swe_mm)
    detected = depth_in >= threshold_in
    conf = assess_snow_confidence(depth_in, station_reports, threshold_in)

    ns = "N" if lat >= 0 else "S"; ew = "E" if lon >= 0 else "W"
    coord = f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    thr = f'{threshold_in:.0f}″'

    nearest_snow = station_reports[0]["snow_in"] if station_reports else None
    rows = [
        {"label": "Snow depth (at property)", "c1": f"{depth_in:.1f}", "c2": f"{depth_mm:.0f}" if depth_mm and not np.isnan(depth_mm) else "—", "highlight": True},
        {"label": "Snow-water-equiv (SWE)", "c1": f"{swe_in:.2f}", "c2": f"{swe_mm:.0f}" if swe_mm and not np.isnan(swe_mm) else "—"},
        {"label": "Roof load (from SWE)", "c1": f"{load_psf:.1f}", "c2": "psf"},
        {"label": "Nearest station snowfall",
         "c1": f"{nearest_snow:.1f}" if nearest_snow is not None else "—",
         "c2": f"{station_reports[0]['dist_mi']:.1f} mi" if station_reports else "—"},
    ]

    dk = (hc._THEME_DETECTED if detected else hc._THEME_CLEAR)["dark"]
    finding = (f'Significant snow accumulation (≥ {thr}) <span style="color:{dk};">'
               f'{"was present" if detected else "was not present"}</span> at this property.')
    sub = (f'Estimated snow depth at the property on the date of loss was '
           f'<strong style="color:#06101f;">{depth_in:.1f}″</strong> '
           f'(roof load ≈ <strong style="color:#06101f;">{load_psf:.0f} psf</strong>) — '
           f'{"at/above" if detected else "below"} the {thr} significance threshold.')

    return {
        "reportId": report_id, "dateGenerated": f"{generated:%B %d, %Y}",
        "dateOfLoss": f"{date_of_loss:%B %d, %Y}", "propertyAddress": address_label,
        "claimRef": claim_ref or "—", "coordinates": coord,
        "contactUrl": contact_url, "contactCity": contact_city,
        "bandLabel": "Snow Analysis", "reportTitle": "Snow Verification Report",
        "flag": detected, "statusText": "Significant" if detected else "Below Threshold",
        "findingHtml": finding, "findingSubHtml": sub,
        "resultsTitle": "Snow Accumulation & Load",
        "colHeaders": {"label": "Measure", "c1": "in", "c2": "mm / dist"},
        "rows": rows,
        "resultsFootnote": "Depth & SWE from SNODAS grid; snowfall from nearest station.",
        "mapTitle": "Snow Depth Footprint", "mapDataUri": map_data_uri,
        "mapCaption": f"Estimated snow depth — NOAA SNODAS, {date_of_loss:%B %d, %Y}.",
        "legendGradient": "linear-gradient(90deg,#dbe7f2,#9ec6e6,#5a9bd4,#3b6fb0,#5a3b8a)",
        "legendLeft": "0.5", "legendRight": "24+ in", "legendLabel": "DEPTH",
        "confidenceLevel": conf["level"], "confidenceColor": conf["color"],
        "confidenceNote": conf["note"], "corroborationLine": _snow_corrob_line(station_reports),
        "methodologyText": (
            "Snow depth and snow-water-equivalent (SWE) are sampled from NOAA's SNODAS "
            "model (National Operational Hydrologic Remote Sensing Center), a ~1 km daily "
            "snow analysis. Roof load is derived from SWE (1 mm SWE ≈ 0.205 lb/ft²). New "
            "snowfall is the nearest COOP/CoCoRaHS station's measured 24-hour total. SNODAS "
            "is a model ESTIMATE; station snowfall is a direct measurement that may be miles "
            "from the property."),
        "disclaimerText": (
            "This is an estimate for informational purposes only. It is NOT a physical "
            "inspection, NOT a structural load determination, and not a substitute for an "
            "on-site assessment by a qualified professional. Roof-load figures are derived "
            "estimates and must not be used as engineering values. ClearClaims Co. makes no "
            "warranty and accepts no liability arising from use of this report. Source data "
            "is U.S. NOAA public-domain. ClearClaims Co. is an independent provider and is "
            "<strong style=\"color:#5a6b7e;\">not affiliated with Cotality or CoreLogic</strong>."),
        "_detected": detected, "_depth_in": round(depth_in, 1),
        "_load_psf": load_psf, "_confidence": conf,
    }


def _snow_corrob_line(reports):
    if not reports:
        return "No nearby station snowfall reports were available for this date."
    parts = [f"{r['snow_in']:.1f}″ — {r['dist_mi']:.1f} mi {r['dir']} ({r['source']})"
             for r in reports[:3]]
    extra = f" +{len(reports) - 3} more" if len(reports) > 3 else ""
    return "Nearby snowfall: " + "; ".join(parts) + extra + "."


def render(data: dict, out_pdf: str, font_dir: str | None = None) -> str:
    html = peril_report.build_report_html_generic(data, font_dir=font_dir)
    return hc.render_pdf_weasyprint(html, out_pdf)
