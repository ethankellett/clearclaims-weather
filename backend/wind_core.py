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


def parse_asos_gust_csv(text: str, utc_start=None, utc_end=None):
    """Max gust (mph) from an IEM ASOS CSV. Gust column is in knots → convert.

    Handles IEM's '#'-comment header lines and a 'gust' (knots) column; ignores
    'M'/missing values. When `utc_start`/`utc_end` are given and the CSV has a
    'valid' timestamp column, only readings inside [utc_start, utc_end) count —
    this pins the peak gust to the exact local calendar day of loss instead of
    whole UTC days (which could pick up a gust from the previous local evening
    or miss the evening of the loss date).
    """
    import csv
    import io
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if not lines:
        return None
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    gust_col = None
    valid_col = None
    for name in (reader.fieldnames or []):
        low = (name or "").strip().lower()
        if gust_col is None and "gust" in low:
            gust_col = name
        if valid_col is None and low == "valid":
            valid_col = name
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
        if valid_col is not None and utc_start is not None and utc_end is not None:
            ts_raw = (row.get(valid_col) or "").strip()
            ts = None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    ts = dt.datetime.strptime(ts_raw, fmt).replace(tzinfo=dt.timezone.utc)
                    break
                except ValueError:
                    pass
            # Unparseable timestamp: keep the reading (fail open, never lose data).
            if ts is not None and not (utc_start <= ts < utc_end):
                continue
        if peak_kt is None or kt > peak_kt:
            peak_kt = kt
    return round(peak_kt * KT_TO_MPH, 1) if peak_kt is not None else None


def fetch_station_peak_gust(station_id, utc_start, utc_end, timeout=45):
    """Peak gust (mph) at one ASOS station over the window, via the IEM ASOS service."""
    import requests
    # Request one extra day on each side of the UTC window (IEM's end date is
    # treated as exclusive) and let the parser filter readings to the exact
    # [utc_start, utc_end) window using each row's 'valid' timestamp.
    end_day = (utc_end + dt.timedelta(days=1)).date()
    params = {
        "station": station_id, "data": "gust", "tz": "UTC",
        "year1": utc_start.year, "month1": utc_start.month, "day1": utc_start.day,
        "year2": end_day.year, "month2": end_day.month, "day2": end_day.day,
        "format": "onlycomma", "missing": "M", "latlon": "no",
    }
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
                     params=params, timeout=timeout)
    r.raise_for_status()
    return parse_asos_gust_csv(r.text, utc_start, utc_end)


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
#  How far away is the nearest instrument? This is the whole ballgame for wind.
#  A thunderstorm downburst is typically one to two miles across. An ASOS station
#  twenty miles away that recorded nothing is very weak evidence that nothing
#  happened at the property — but the old code called that High confidence.
WIND_DISTANCE_GRADES = (
    (5.0, "Excellent", "An official station sits within 5 miles of the property."),
    (12.0, "Good", "The nearest official station is within 12 miles."),
    (25.0, "Fair", "The nearest official station is 12\u201325 miles away \u2014 a "
                   "localised downburst could pass between stations unrecorded."),
)


def grade_wind_distance(dist_mi):
    """Grade how well the station network covers this property."""
    if dist_mi is None:
        return {"grade": "No measurement", "dist_mi": None,
                "note": ("No official station reported a gust for this date, so there "
                         "is no measurement to reason from.")}
    for cut, grade, note in WIND_DISTANCE_GRADES:
        if dist_mi <= cut:
            return {"grade": grade, "dist_mi": dist_mi, "note": note}
    return {"grade": "Poor", "dist_mi": dist_mi,
            "note": (f"The nearest official station is {dist_mi:.0f} miles away. A "
                     f"downburst is typically 1\u20132 miles wide and can easily miss "
                     f"every station in a county.")}


_WIND_RANK = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1, "No measurement": 0}


def assess_wind_confidence(peak_mph, n_stations, reports, threshold_mph,
                           nearest_dist_mi=None):
    """Confidence in the wind verdict, weighted by how close the instrument was.

    The asymmetry is deliberate and mirrors the hail report: a measured gust over
    the threshold is strong evidence FOR damaging wind, but a measured gust under
    the threshold twenty miles away is weak evidence AGAINST it. The previous
    version returned High for any measured negative regardless of distance.
    """
    detected = peak_mph is not None and peak_mph >= threshold_mph
    n = len(reports)
    has_measured = peak_mph is not None
    q = grade_wind_distance(nearest_dist_mi)
    rank = _WIND_RANK.get(q["grade"], 0)

    if detected:
        if n >= 1:
            lvl = "High"
            note = (f"A measured gust of {peak_mph:.0f} mph meets the "
                    f"{threshold_mph:.0f} mph threshold and is corroborated by {n} "
                    f"independent wind report(s) nearby.")
        else:
            lvl = "Moderate"
            note = (f"A measured gust of {peak_mph:.0f} mph meets the "
                    f"{threshold_mph:.0f} mph threshold, but no independent wind "
                    f"report was logged nearby.")
    elif not has_measured:
        lvl = "Low"
        note = ("No official station reported a gust for this date. This report can "
                "neither confirm nor rule out damaging wind at the property."
                + (f" {n} nearby wind report(s) were logged." if n else ""))
    elif n >= 1:
        lvl = "Low"
        note = (f"No measured gust met the threshold, yet {n} wind report(s) were "
                f"logged nearby \u2014 verify timing and station coverage.")
    elif rank >= 4:
        lvl = "High"
        note = (f"The nearest official station, within {nearest_dist_mi:.0f} miles, "
                f"measured a peak of {peak_mph:.0f} mph \u2014 below the "
                f"{threshold_mph:.0f} mph threshold \u2014 and no wind reports were "
                f"logged nearby.")
    else:
        # The old code said High here regardless of distance. It does not follow.
        lvl = "Moderate" if rank >= 3 else "Low"
        note = (f"The nearest official station measured a peak of {peak_mph:.0f} mph, "
                f"below the {threshold_mph:.0f} mph threshold. {q['note']} A wind "
                f"negative is weaker evidence than a positive.")

    color = {"High": "#28a678", "Moderate": "#e6a117", "Low": "#d94f3d"}[lvl]
    return {"level": lvl, "color": color, "note": note, "n_reports": n,
            "quality": q}


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
    # Keep MEASURED and REPORTED separate. The old code took max() of both and
    # called the result "peak estimated wind gust at/near the property", so a
    # spotter's estimate fifteen miles away could read as the property's gust.
    meas = [s for s in station_gusts if s.get("gust_mph") is not None]
    measured_peak = max((s["gust_mph"] for s in meas), default=None)
    nearest = min(meas, key=lambda s: s.get("dist_mi", 1e9)) if meas else None
    nearest_dist = nearest.get("dist_mi") if nearest else None

    rep_speeds = [r["speed_mph"] for r in reports if r.get("speed_mph") is not None]
    reported_peak = max(rep_speeds) if rep_speeds else None

    peak = max([v for v in (measured_peak, reported_peak) if v is not None],
               default=None)
    detected = peak is not None and peak >= threshold_mph
    conf = assess_wind_confidence(measured_peak, len(station_gusts), reports,
                                  threshold_mph, nearest_dist_mi=nearest_dist)
    quality = conf["quality"]

    ns = "N" if lat >= 0 else "S"; ew = "E" if lon >= 0 else "W"
    coord = f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    thr = f"{threshold_mph:.0f} mph"
    peak_txt = f"{peak:.0f} mph" if peak is not None else "no measurement"

    # Measured and reported are visually separated so nobody can read a spotter's
    # estimate fifteen miles away as this property's gust.
    rows = []
    if nearest is not None:
        rows.append({"label": f"Measured — {nearest.get('name', nearest.get('id',''))}",
                     "c1": f"{nearest['gust_mph']:.0f}",
                     "c2": f"{nearest['dist_mi']:.1f} mi", "highlight": True})
    for st in [x for x in meas if x is not nearest][:2]:
        rows.append({"label": f"Measured — {st.get('id','')}",
                     "c1": f"{st['gust_mph']:.0f}", "c2": f"{st['dist_mi']:.1f} mi"})
    if not meas:
        rows.append({"label": "Measured — none available", "c1": "—", "c2": "—"})
    if reports:
        big = max(reports, key=lambda r: (r.get("speed_mph") or 0))
        rows.append({"label": "Reported nearby (not measured here)",
                     "c1": f"{big['speed_mph']:.0f}" if big.get("speed_mph") else "est.",
                     "c2": f"{big['dist_mi']:.1f} mi"})

    _dk = (hc._THEME_DETECTED if detected else hc._THEME_CLEAR)["dark"]
    if measured_peak is None and reported_peak is None:
        finding = (f'No wind measurement was available <span style="color:{_dk};">'
                   f'near this property</span> on this date.')
        sub = ('No official station reported a gust and no wind reports were logged '
               'nearby. This report can neither confirm nor rule out damaging wind '
               'at the property.')
    else:
        finding = (f'Damaging wind (≥ {thr}) <span style="color:{_dk};">'
                   f'{"was recorded" if detected else "was not recorded"}</span> '
                   f'near this property.')
        bits = []
        if measured_peak is not None:
            bits.append(
                f'Peak <b>measured</b> gust <strong style="color:#06101f;">'
                f'{measured_peak:.0f} mph</strong> at '
                f'{nearest.get("id", "the nearest station")}'
                + (f', {nearest_dist:.1f} mi away' if nearest_dist is not None else ''))
        else:
            bits.append('No official station measured a gust for this date')
        if reported_peak is not None:
            bits.append(f'peak <b>reported</b> gust {reported_peak:.0f} mph nearby')
        sub = ("; ".join(bits) +
               f'. Threshold is {thr}. Measured gusts are instrument readings at the '
               f'station, not at the address.')

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
        "resultsFootnote": ("Measured rows are instrument readings AT THE STATION, not at "
                            "the address. Reported rows are spotter or storm reports nearby "
                            "and may be estimates. " + quality["note"]),
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
            "Clear Claims Co. makes no warranty and accepts no liability arising from use "
            "of this report. Source data is U.S. NOAA public-domain observations. "
            "Clear Claims Co. is an independent provider and is <strong style=\"color:#5a6b7e;\">"
            "not affiliated with Cotality or CoreLogic</strong>."),
        "_detected": detected, "_peak_mph": peak, "_confidence": conf,
        "_measured_peak_mph": measured_peak, "_reported_peak_mph": reported_peak,
        "_nearest_dist_mi": nearest_dist, "_quality": quality,
        "measurementQuality": (
            f'{quality["grade"]}'
            + (f' ({nearest_dist:.1f} mi)' if nearest_dist is not None else '')),
    }


def _wind_corrob_line(reports):
    if not reports:
        return "No NWS/SPC wind reports within the search radius on this date."
    parts = []
    for r in reports[:3]:
        spd = f"{r['speed_mph']:.0f} mph" if r.get("speed_mph") else "damage (est.)"
        # Defensive: a malformed upstream record must not 500 an entire report.
        parts.append(f"{spd} — {r.get('dist_mi', 0):.1f} mi "
                     f"{r.get('dir', '')} ({r.get('source', 'report')})".replace("  ", " "))
    extra = f" +{len(reports) - 3} more" if len(reports) > 3 else ""
    return "Nearby reports: " + "; ".join(parts) + extra + "."


def render(data: dict, out_pdf: str, font_dir: str | None = None) -> str:
    html = peril_report.build_report_html_generic(data, font_dir=font_dir)
    return hc.render_pdf_weasyprint(html, out_pdf)
