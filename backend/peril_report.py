# =============================================================================
#  ClearClaims — GENERIC one-page report template (shared by wind & snow)
#  Same look as the hail "Classic Forensic" template, but every peril-specific
#  bit (title, wording, table columns/rows, legend, methodology) is a parameter.
#  Renders to PDF with WeasyPrint. One-page hard cap (can't spill to page 2).
# =============================================================================

import hail_core as hc   # reuse logo, icons, fonts, render, brand colours


def _rows_html(rows, accent_dark):
    """Build the results-table body rows. Each row: {label,c1,c2,highlight,tint}."""
    out = []
    for i, r in enumerate(rows):
        bg = r.get("tint", "#f7fafd" if i % 2 else "#ffffff")
        if r.get("highlight"):
            bg = r.get("rowTint", "#fdf1ef")
        c1_color = accent_dark if r.get("highlight") else "#06101f"
        weight = "700" if r.get("highlight") else "600"
        label_w = "700" if r.get("highlight") else "400"
        out.append(
            f'<tr style="background:{bg};"><td style="padding:8px 12px; font-size:12px; '
            f'color:#0c1a30; font-weight:{label_w}; border-bottom:1px solid #eef3f8;">{r["label"]}</td>'
            f'<td style="padding:8px 10px; text-align:right; font-size:13px; color:{c1_color}; '
            f'font-weight:{weight}; border-bottom:1px solid #eef3f8;">{r["c1"]}</td>'
            f'<td style="padding:8px 12px; text-align:right; font-size:12px; color:#5a6b7e; '
            f'border-bottom:1px solid #eef3f8;">{r["c2"]}</td></tr>')
    return "\n".join(out)


def build_report_html_generic(data: dict, font_dir: str | None = None) -> str:
    """Render a peril report to self-contained HTML. See wind_core/snow_core for
    the data dict they pass. Keys mirror the hail template plus:
      resultsTitle, colHeaders {c1,c2}, rows[], resultsFootnote,
      mapTitle, legendGradient, legendLeft, legendRight, legendLabel,
      findingHtml, findingSubHtml, statusText, flag (bool: alert vs clear).
    """
    flag = bool(data["flag"])
    t = hc._THEME_DETECTED if flag else hc._THEME_CLEAR
    icon = hc._ICON_TRIANGLE if flag else hc._ICON_CHECK
    font_face = hc._font_face_css(font_dir)

    # map block (real image, else neutral placeholder)
    if data.get("mapDataUri"):
        map_block = (f'<div style="height:158px; background-color:#eef3f8; '
                     f'background-image:url(\'{data["mapDataUri"]}\'); background-size:cover; '
                     f'background-position:center; background-repeat:no-repeat;"></div>')
    else:
        map_block = '<div style="height:158px; background:#eef3f8;"></div>'

    conf_level = data.get("confidenceLevel")
    conf_block = ""
    if conf_level:
        conf_block = f"""
    <div style="margin-top:10px; border:1px solid #e2e9f1; border-radius:4px; padding:10px 14px; display:flex; gap:14px; align-items:flex-start;">
      <div style="flex:none; background:{data.get('confidenceColor', '#7d8ea1')}; color:#fff; font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; padding:6px 11px; border-radius:3px; white-space:nowrap;">{conf_level} Confidence</div>
      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:3px;">Corroboration &amp; Confidence</div>
        <div style="font-size:10.5px; color:#46566a; line-height:1.5;">{data.get('confidenceNote', '')}</div>
        <div style="font-size:10px; color:#5a6b7e; line-height:1.5; margin-top:4px;">{data.get('corroborationLine', '')}</div>
      </div>
    </div>"""

    ch = data["colHeaders"]
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
  .page {{ width:816px; height:1056px; background:#ffffff; display:flex; flex-direction:column; overflow:hidden; }}
</style></head>
<body>
<div class="page">

  <div style="background:#06101f; padding:26px 44px 22px; display:flex; align-items:flex-start; justify-content:space-between;">
    <div style="display:flex; align-items:center; gap:14px;">
      {hc._LOGO_SVG}
      <div>
        <div style="font-family:'DM Serif Display',serif; font-size:25px; line-height:1; color:#fff; white-space:nowrap;">Clear<span style="color:#4a9af5;">Claims</span> <span style="font-family:'Outfit'; font-size:12px; font-weight:500; color:#8aa0b8; letter-spacing:.03em;">Co.</span></div>
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

  <div style="padding:16px 44px 0; flex:1; display:flex; flex-direction:column;">

    <div style="border:1px solid #d9e4f0; border-radius:3px; display:grid; grid-template-columns:1fr 1fr 1fr;">
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Report ID</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["reportId"]}</div></div>
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Date Generated</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["dateGenerated"]}</div></div>
      <div style="padding:9px 16px; border-bottom:1px solid #e7eef6;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Date of Loss</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["dateOfLoss"]}</div></div>
      <div style="padding:9px 16px; border-right:1px solid #e7eef6; grid-column:span 2;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Property Address</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["propertyAddress"]}</div></div>
      <div style="padding:9px 16px;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Claim / Reference</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px;">{data["claimRef"]}</div></div>
      <div style="padding:9px 16px; border-top:1px solid #e7eef6; grid-column:span 3;"><div style="font-size:8.5px; letter-spacing:.1em; text-transform:uppercase; color:#7d8ea1; font-weight:600;">Coordinates</div><div style="font-size:13px; color:#0c1a30; font-weight:600; margin-top:2px; font-variant-numeric:tabular-nums;">{data["coordinates"]}</div></div>
    </div>

    <div style="margin-top:12px; background:{t['tint']}; border:1px solid {t['tintBorder']}; border-left:6px solid {t['main']}; border-radius:3px; padding:13px 18px; display:flex; align-items:center; gap:16px;">
      <div style="flex:none; width:52px; height:52px; border-radius:50%; background:{t['main']}; display:flex; align-items:center; justify-content:center;">{icon}</div>
      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.14em; text-transform:uppercase; color:{t['deep']}; font-weight:700;">Key Finding</div>
        <div style="font-family:'DM Serif Display',serif; font-size:22px; color:#06101f; line-height:1.08; margin-top:2px;">{data["findingHtml"]}</div>
        <div style="font-size:11.5px; color:#54616f; margin-top:4px;">{data["findingSubHtml"]}</div>
      </div>
      <div style="flex:none; background:{t['main']}; color:#fff; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; padding:8px 14px; border-radius:3px; text-align:center; white-space:nowrap;">{data["statusText"]}</div>
    </div>

    <div style="display:flex; gap:20px; margin-top:12px;">
      <div style="flex:none; width:300px;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:8px;">{data["resultsTitle"]}</div>
        <table style="width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums;">
          <thead><tr style="background:#0c1a30;">
            <th style="text-align:left; padding:8px 12px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">{ch.get('label','Location')}</th>
            <th style="text-align:right; padding:8px 10px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">{ch['c1']}</th>
            <th style="text-align:right; padding:8px 12px; font-size:9px; letter-spacing:.06em; text-transform:uppercase; color:#9fb2c7; font-weight:600;">{ch['c2']}</th>
          </tr></thead>
          <tbody>
            {_rows_html(data["rows"], t['dark'])}
          </tbody>
        </table>
        <div style="font-size:9px; color:#90a0b2; margin-top:7px; line-height:1.4;">{data.get("resultsFootnote", "")}</div>
      </div>

      <div style="flex:1;">
        <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:8px;">{data["mapTitle"]}</div>
        <div style="border:1px solid #c4d2e2; border-radius:3px; overflow:hidden;">
          {map_block}
          <div style="display:flex; align-items:center; gap:0; padding:7px 10px; border-top:1px solid #e7eef6; background:#fff;">
            <span style="font-size:8.5px; color:#7d8ea1; margin-right:8px; font-weight:600;">{data.get("legendLabel","SCALE")}</span>
            <span style="flex:1; height:9px; border-radius:2px; background:{data["legendGradient"]};"></span>
            <span style="font-size:8.5px; color:#7d8ea1; margin-left:8px;">{data["legendLeft"]}</span>
            <span style="font-size:8.5px; color:#7d8ea1; margin-left:6px;">{data["legendRight"]}</span>
          </div>
        </div>
        <div style="font-size:9px; color:#90a0b2; margin-top:7px; line-height:1.4;">{data["mapCaption"]}</div>
      </div>
    </div>

    {conf_block}

    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #e7eef6;">
      <div style="font-size:9.5px; letter-spacing:.12em; text-transform:uppercase; color:#0c1a30; font-weight:700; margin-bottom:5px;">Methodology</div>
      <div style="font-size:9.7px; color:#46566a; line-height:1.45;">{data["methodologyText"]}</div>
    </div>

    <div style="margin-top:10px; padding-top:0; margin-bottom:20px;">
      <div style="background:#f0f4f8; border-radius:3px; padding:11px 14px;">
        <div style="font-size:8px; letter-spacing:.12em; text-transform:uppercase; color:#8a99ab; font-weight:700; margin-bottom:4px;">Disclaimer</div>
        <div style="font-size:8.5px; color:#7a8a9c; line-height:1.5;">{data["disclaimerText"]}</div>
      </div>
    </div>
  </div>

  <div style="background:#06101f; padding:9px 44px; display:flex; align-items:center; justify-content:space-between; font-size:8.5px; color:#7d8ea1; letter-spacing:.04em;">
    <span>Report {data["reportId"]}</span>
    <span style="color:#4a9af5; font-weight:600; letter-spacing:.1em; text-transform:uppercase;">Confidential</span>
    <span>Page 1 of 1 · Generated {data["dateGenerated"]}</span>
  </div>
</div>
</body></html>"""
