# =============================================================================
#  ClearClaims — WIND verification engine
#  Headline = peak wind GUST at/near the property on the date of loss, from:
#    * measured gusts at the nearest official stations (IEM ASOS archive)
#    * NWS Local Storm Reports + SPC wind reports (corroboration)
#  Detection: peak gust >= threshold (default 58 mph = NWS severe criterion).
#
#  Reuses hail_core for geocoding, the UTC window, distance/bearing, fonts and
#  PDF rendering, and peril_report for the shared one-page template.
#
#  NOTE: the live data feeds (Census reverse-geocode, IEM ASOS, IEM/SPC reports)
#  run on the open internet (Colab/Render). The PARSING + sampling + rendering
#  here are unit-tested against real-format samples; confirm the live fetch on
#  your first Colab run.
# =============================================================================

from __future__ import annotations
import datetime as dt
import math
import numpy as np

import hail_core as hc
import peril_report

KT_TO_MPH = 1.15078
DEFAULT_THRESHOLD_MPH = 58.0     # NWS severe thunderstorm wind criterion (50 kt)


# ---- nearest official stations -------------------------------------------
def census_state(lat: float, lon: float, timeout: int = 30):
    """Reverse-geocode lat/lon → 2-letter US state via the Census geographies API."""
    import requests
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {"x": lon, "y": lat, "benchmark": "Public_AR_Current",
              "vintage": "Current_Current", "format": "json", "layers": "States"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    states = r.json().get("result", {}).get("geographies", {}).get("States", [])
    if states:
        return states[0].get("STUSAB") or states[0].get("STATE")
    return None


def parse_station_geojson(obj: dict, lat: float, lon: float, n=3):
    """From an IEM network GeoJSON, return the n nearest stations to the point."""
    out = []
    for feat in (obj or {}).get("features", []):
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        slon, slat = coords[0], coords[1]
        if slat is None:
            continue
        sid = feat.get("id") or (feat.get("properties") or {}).get("sid")
        d = float(hc.haversine_miles(lat, lon, slat, slon))
        out.append({"id": sid, "lat": float(slat), "lon": float(slon), "dist_mi": d,
                    "name": (feat.get("properties") or {}).get("sname", sid)})
    out.sort(key=lambda s: s["dist_mi"])
    return out[:n]


def fetch_nearest_stations(lat, lon, n=3, timeout=30):
    """Find the nearest ASOS stations to the point (best-effort). Returns list."""
    import requests
    st = census_state(lat, lon, timeout=timeout)
    if not st:
        return []
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{st}_ASOS.geojson"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return parse_station_geojson(r.json(), lat, lon, n=n)


def parse_asos_gust_csv(text: str):
    """Max gust (mph) from an IEM ASOS CSV. Gust column is in knots → convert.

    Handles IEM's '#'-comment header lines and a 'gust' (knots) column; ignores
    'M'/missing values.
    """
    import csv
    import io
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if not lines:
        return None
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    gust_col = None
    for name in (reader.fieldnames or []):
        if name and "gust" in name.lower():
            gust_col = name
            break
    if not gust_col:
        return None
    peak_kt = None
    for row in reader:
        v = (row.get(gust_col) or "").strip()
        if v in ("", "M", "None", "null"):
            continue
        try:
            kt = float(v)
        except ValueError:
            continue
        if peak_kt is None or kt > peak_kt:
            peak_kt = kt
    return round(peak_kt * KT_TO_MPH, 1) if peak_kt is not None else None


def fetch_station_peak_gust(station_id, utc_start, utc_end, timeout=45):
    """Peak gust (mph) at one ASOS station over the window, via the IEM ASOS service."""
    import requests
    params = {
        "station": station_id, "data": "gust", "tz": "UTC",
        "year1": utc_start.year, "month1": utc_start.month, "day1": utc_start.day,
        "year2": utc_end.year, "month2": utc_end.month, "day2": utc_end.day,
        "format": "onlycomma", "missing": "M", "latlon": "no",
    }
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                     params=params, timeout=timeout)
    r.raise_for_status()
    return parse_asos_gust_csv(r.text)


def gather_station_gusts(lat, lon, utc_start, utc_end, n=3):
    """Best-effort: peak gust at the nearest n stations. Returns list of dicts."""
    out = []
    try:
        stations = fetch_nearest_stations(lat, lon, n=n)
    except Exception:
        stations = []
    for s in stations:
        try:
            g = fetch_station_peak_gust(s["id"], utc_start, utc_end)
        except Exception:
            g = None
        if g is not None:
            out.append({**s, "gust_mph": g,
                        "dir": hc.compass_bearing(lat, lon, s["lat"], s["lon"])})
    out.sort(key=lambda x: x["gust_mph"], reverse=True)
    return out


# ---- wind storm reports (NWS LSR + SPC) ----------------------------------
def parse_lsr_wind(obj: dict, lat, lon, radius_miles):
    """NWS LSR wind reports (gust 'G' and damage 'D') within radius. Speed in mph."""
    out = []
    for feat in (obj or {}).get("features", []):
        p = feat.get("properties", {}) or {}
        typ = (p.get("type") or "").upper()
        tt = (p.get("typetext") or "").upper()
        is_wind = typ in ("G", "D") or "WND" in tt or "WIND" in tt
        if not is_wind:
            continue
        coords = (feat.get("geometry") or {}).get("coordinates") or [None, None]
        rlon, rlat = coords[0], coords[1]
        if rlat is None:
            continue
        try:
            spd = float(p.get("magnitude"))
            # LSR wind magnitude can be mph (measured) or knots; IEM reports mph.
        except (TypeError, ValueError):
            spd = None
        d = float(hc.haversine_miles(lat, lon, rlat, rlon))
        if d <= radius_miles:
            out.append({"source": "NWS LSR", "speed_mph": spd, "lat": float(rlat),
                        "lon": float(rlon), "dist_mi": d,
                        "dir": hc.compass_bearing(lat, lon, rlat, rlon),
                        "time": p.get("valid", ""), "kind": tt or "WIND"})
    return out


def parse_spc_wind_csv(text: str, lat, lon, radius_miles):
    """SPC daily wind CSV: Time,Speed,Location,County,State,Lat,Lon,Comments.
    Speed is mph (measured) or 'UNK' (estimated/ gust from damage)."""
    import csv
    import io
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            rlat = float(row.get("Lat")); rlon = float(row.get("Lon"))
        except (TypeError, ValueError):
            continue
        spd_raw = (row.get("Speed") or "").strip()
        try:
            spd = float(spd_raw)
        except ValueError:
            spd = None     # 'UNK' = estimated/damage report
        d = float(hc.haversine_miles(lat, lon, rlat, rlon))
        if d <= radius_miles:
            out.append({"source": "SPC", "speed_mph": spd, "lat": rlat, "lon": rlon,
                        "dist_mi": d, "dir": hc.compass_bearing(lat, lon, rlat, rlon),
                        "time": row.get("Time", ""), "kind": row.get("Location", "")})
    return out


def fetch_wind_reports(lat, lon, utc_start, utc_end, date_of_loss, radius_miles=15.0):
    """Best-effort nearby wind reports from IEM LSR + SPC. Never raises."""
    import requests
    reports = []
    try:
        pad = 0.6
        params = {"sts": utc_start.strftime("%Y-%m-%dT%H:%MZ"),
                  "ets": (utc_end + dt.timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ"),
                  "west": lon - pad, "east": lon + pad, "south": lat - pad, "north": lat + pad}
        r = requests.get("https://mesonet.agron.iastate.edu/geojson/lsr.geojson",
                         params=params, timeout=30)
        reports += parse_lsr_wind(r.json(), lat, lon, radius_miles)
    except Exception:
        pass
    try:
        url = f"https://www.spc.noaa.gov/climo/reports/{date_of_loss:%y%m%d}_rpts_wind.csv"
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and "Lat" in r.text[:200]:
            reports += parse_spc_wind_csv(r.text, lat, lon, radius_miles)
    except Exception:
        pass
    reports.sort(key=lambda x: x["dist_mi"])
    return reports


# ---- confidence + map -----------------------------------------------------
def assess_wind_confidence(peak_mph, n_stations, reports, threshold_mph):
    detected = peak_mph is not None and peak_mph >= threshold_mph
    n = len(reports)
    has_measured = peak_mph is not None
    if detected:
        if n >= 1:
            lvl, note = "High", (f"Peak gust of {peak_mph:.0f} mph is corroborated by "
                                 f"{n} nearby wind report(s).")
        elif has_measured:
            lvl, note = "Moderate", (f"A measured gust of {peak_mph:.0f} mph meets the "
                                     f"threshold, but no nearby storm report was logged.")
        else:
            lvl, note = "Low", "Threshold met only by an estimated report; no measured gust."
    else:
        if n >= 1 and not has_measured:
            lvl, note = "Low", (f"No measured gust met the threshold, but {n} wind "
                                f"report(s) were logged nearby — verify station coverage.")
        elif has_measured:
            lvl, note = "High", (f"Nearest measured peak gust was {peak_mph:.0f} mph, "
                                 f"below the {threshold_mph:.0f} mph threshold.")
        else:
            lvl, note = "Low", "No measured station gust and no reports were available."
    color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[lvl]
    return {"level": lvl, "color": color, "note": note, "n_reports": n}


def make_wind_map(lat, lon, stations, reports, out_png, brand=None):
    """Locator map: property + nearest stations (with gust) + wind reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts_lat = [lat] + [s["lat"] for s in stations] + [r["lat"] for r in reports]
    pts_lon = [lon] + [s["lon"] for s in stations] + [r["lon"] for r in reports]
    pad = max(0.15, (max(pts_lat) - min(pts_lat)) * 0.25, (max(pts_lon) - min(pts_lon)) * 0.25) if len(pts_lat) > 1 else 0.3

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.set_facecolor("#eef3f8")
    # wind reports
    for r in reports:
        ax.plot(r["lon"], r["lat"], marker="^", markersize=8, color="#e6a117",
                markeredgecolor="#7a5b10", zorder=4)
    # stations
    for s in stations:
        ax.plot(s["lon"], s["lat"], marker="s", markersize=8, color="#2b7de9",
                markeredgecolor="#13407a", zorder=5)
        ax.annotate(f"{s.get('gust_mph',''):.0f} mph" if s.get("gust_mph") else "",
                    (s["lon"], s["lat"]), fontsize=7, color="#13407a",
                    xytext=(4, 4), textcoords="offset points")
    # property
    ax.plot(lon, lat, marker="o", markersize=11, markerfacecolor="#06101f",
            markeredgecolor="white", markeredgewidth=1.6, zorder=6)

    ax.set_xlim(min(pts_lon) - pad, max(pts_lon) + pad)
    ax.set_ylim(min(pts_lat) - pad, max(pts_lat) + pad)
    ax.set_xlabel("Longitude", fontsize=8); ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Property · ◻ stations · ▲ wind reports", fontsize=9, color="#06101f")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return out_png


# ---- assemble the report data --------------------------------------------
def build_wind_report_data(*, report_id, address_label, lat, lon, date_of_loss,
                           contact_url, contact_city, claim_ref, threshold_mph,
                           station_gusts, reports, map_data_uri, generated=None):
    generated = generated or dt.datetime.now(dt.timezone.utc)
    measured = [s["gust_mph"] for s in station_gusts if s.get("gust_mph") is not None]
    report_spds = [r["speed_mph"] for r in reports if r.get("speed_mph") is not None]
    peak = max(measured + report_spds) if (measured or report_spds) else None
    detected = peak is not None and peak >= threshold_mph
    conf = assess_wind_confidence(max(measured) if measured else None,
                                  len(station_gusts), reports, threshold_mph)

    ns = "N" if lat >= 0 else "S"; ew = "E" if lon >= 0 else "W"
    coord = f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    thr = f"{threshold_mph:.0f} mph"
    peak_txt = f"{peak:.0f} mph" if peak is not None else "no measurement"

    rows = []
    if station_gusts:
        s0 = station_gusts[0]
        rows.append({"label": f"Nearest station ({s0.get('name', s0['id'])})",
                     "c1": f"{s0['gust_mph']:.0f}", "c2": f"{s0['dist_mi']:.1f} mi",
                     "highlight": True})
        for s in station_gusts[1:3]:
            rows.append({"label": f"Station {s.get('id','')}",
                         "c1": f"{s['gust_mph']:.0f}", "c2": f"{s['dist_mi']:.1f} mi"})
    if reports:
        big = max(reports, key=lambda r: (r.get("speed_mph") or 0))
        rows.append({"label": "Peak nearby report",
                     "c1": f"{big['speed_mph']:.0f}" if big.get("speed_mph") else "est.",
                     "c2": f"{big['dist_mi']:.1f} mi"})
    if not rows:
        rows.append({"label": "No station/report data", "c1": "—", "c2": "—"})

    finding = (f'Damaging wind (≥ {thr}) <span style="color:{(hc._THEME_DETECTED if detected else hc._THEME_CLEAR)["dark"]};">'
               f'{"was detected" if detected else "was not detected"}</span> at this property.')
    sub = (f'Peak estimated wind gust at/near the property on the date of loss was '
           f'<strong style="color:#06101f;">{peak_txt}</strong> — '
           f'{"above" if detected else "below"} the {thr} damaging-wind threshold.')

    return {
        "reportId": report_id, "dateGenerated": f"{generated:%B %d, %Y}",
        "dateOfLoss": f"{date_of_loss:%B %d, %Y}", "propertyAddress": address_label,
        "claimRef": claim_ref or "—", "coordinates": coord,
        "contactUrl": contact_url, "contactCity": contact_city,
        "bandLabel": "Wind Analysis", "reportTitle": "Wind Verification Report",
        "flag": detected, "statusText": "Detected" if detected else "Not Detected",
        "findingHtml": finding, "findingSubHtml": sub,
        "resultsTitle": "Peak Wind Gust", "colHeaders": {"label": "Source", "c1": "mph", "c2": "distance"},
        "rows": rows,
        "resultsFootnote": "Measured gusts from official ASOS stations; reports from NWS/SPC.",
        "mapTitle": "Wind Observations", "mapDataUri": map_data_uri,
        "mapCaption": f"Nearest stations and wind reports — {date_of_loss:%B %d, %Y}.",
        "legendGradient": "linear-gradient(90deg,#28a678,#7cc36a,#e6a117,#e07a2e,#d94f3d)",
        "legendLeft": "40", "legendRight": "90+ mph", "legendLabel": "GUST",
        "confidenceLevel": conf["level"], "confidenceColor": conf["color"],
        "confidenceNote": conf["note"],
        "corroborationLine": _wind_corrob_line(reports),
        "methodologyText": (
            "Peak wind gusts are the highest measured gusts at the nearest official "
            "ASOS/AWOS stations over the local date of loss (NOAA, via the Iowa "
            "Environmental Mesonet), cross-checked against NWS Local Storm Reports and "
            "SPC wind reports. Station gusts are direct measurements; the nearest "
            "station may be several miles from the property, so nearby reports are "
            "included for spatial context."),
        "disclaimerText": (
            "This is an estimate based on the nearest available measurements and reports, "
            "for informational purposes only. It is NOT a physical inspection, NOT a "
            "guarantee of wind damage, and not a substitute for an on-site assessment. "
            "ClearClaims Co. makes no warranty and accepts no liability arising from use "
            "of this report. Source data is U.S. NOAA public-domain observations. "
            "ClearClaims Co. is an independent provider and is <strong style=\"color:#5a6b7e;\">"
            "not affiliated with Cotality or CoreLogic</strong>."),
        "_detected": detected, "_peak_mph": peak, "_confidence": conf,
    }


def _wind_corrob_line(reports):
    if not reports:
        return "No NWS/SPC wind reports within the search radius on this date."
    parts = []
    for r in reports[:3]:
        spd = f"{r['speed_mph']:.0f} mph" if r.get("speed_mph") else "damage (est.)"
        parts.append(f"{spd} — {r['dist_mi']:.1f} mi {r['dir']} ({r['source']})")
    extra = f" +{len(reports) - 3} more" if len(reports) > 3 else ""
    return "Nearby reports: " + "; ".join(parts) + extra + "."


def render(data: dict, out_pdf: str, font_dir: str | None = None) -> str:
    html = peril_report.build_report_html_generic(data, font_dir=font_dir)
    return hc.render_pdf_weasyprint(html, out_pdf)
