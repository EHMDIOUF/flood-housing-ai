"""
Flood-Exposed Housing Detection System: Streamlit dashboard (new design).
Run from the project root:   streamlit run src/dashboard/app.py
Reads data/processed/dashboard_kpis.json, *_feature_stack.tif, *_flood_prone_mask.tif,
*_exposure_map.tif and data/raw/<region>/<region>_ghsl_<year>.tif (same files as before).
"""
import inspect
import json
from contextlib import contextmanager
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from folium.plugins import DualMap, Fullscreen, MeasureControl, MiniMap, MousePosition
from PIL import Image
from rasterio.warp import Resampling, reproject, transform_bounds
from streamlit_folium import st_folium

st.set_page_config(page_title="Flood-Exposed Housing Detection | KDU FYP", page_icon="🌊", layout="wide")

PROC, RAW = Path("data/processed"), Path("data/raw")
REGIONS = {"region_a_kelani": "Lower Kelani Basin (Sri Lanka)", "region_b_dakar": "Dakar Northern Suburbs (Senegal)"}
SHORT = {"region_a_kelani": "Kelani", "region_b_dakar": "Dakar"}
PROJECT_TITLE = "Flood-Exposed Housing Detection System"
AUTHOR, SUPERVISOR = "El Hadji Malick Diouf", "Dr. BVKI Vidanage"
INSTITUTION = "Faculty of Computing, General Sir John Kotelawala Defence University"
PAGES = ["1. Executive Summary", "2. Map Viewer", "3. Time Comparison", "4. Built-up Growth (GHSL)", "5. Regional Comparison",
         "6. Risk Assessment", "7. Transferability", "8. Data Quality & Model Performance", "9. Methodology & Sources"]
SUBS = ["Key figures and study-area snapshots", "Interactive GIS workspace: toggle layers, switch basemaps, measure and inspect",
        "Swipe through built-up change between two GHSL epochs", "GHSL built-up surface trends and flood-zone encroachment",
        "How the two study regions compare", "HAND risk tiers: High < 2 m, Medium 2–5 m, Low > 5 m",
        "Does a classifier trained in one region generalise to the other?", "Classifier evaluation and resolved data issues",
        "Pipeline, datasets and limitations"]
ICONS = ['<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
         '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>', '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
         '<path d="m3 17 6-6 4 4 8-9"/><path d="M15 6h6v6"/>', '<path d="M5 21V10M12 21V4M19 21v-8"/>', '<path d="M12 3 2.5 20h19L12 3z"/><path d="M12 10v4M12 17.2v.1"/>',
         '<path d="M4 8h15l-3.5-3.5M20 16H5l3.5 3.5"/>', '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>', '<path d="M12 3 2 8l10 5 10-5-10-5z"/><path d="m2 13 10 5 10-5"/>']
LOGO = ('<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 11l9-7 9 7"/><path d="M5 10v5M19 10v5"/><path d="M2 18c2 0 2-1.5 4-1.5s2 1.5 4 1.5 2-1.5 4-1.5 2 1.5 4 1.5 2-1.5 4-1.5"/>'
        '<path d="M2 22c2 0 2-1.5 4-1.5s2 1.5 4 1.5 2-1.5 4-1.5 2 1.5 4 1.5 2-1.5 4-1.5"/></svg>')

DARK = dict(bg="#0b1218", panel="#111c25", panel2="#16242f", line="#223442", tx="#e6f1f6", mut="#8aa1af", ac="#22d3ee", ac2="#f5a524", bad="#ff6b5e", ok="#4ade80")
LIGHT = dict(bg="#eef4f7", panel="#ffffff", panel2="#f2f7fa", line="#d5e2ea", tx="#0e2230", mut="#5b7586", ac="#0b8fa8", ac2="#b86e00", bad="#c9382b", ok="#1f8f4e")
WIDE = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
IMG_WIDE = {"width": "stretch"} if "width" in inspect.signature(st.image).parameters else {"use_container_width": True}

# ------------------------------------------------------------------ SIDEBAR + THEME
with st.sidebar:
    st.markdown(f'<div class="brand"><span class="logo">{LOGO}</span>Flood-Exposed Housing</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    light = st.toggle("Light theme", key="light")
    st.caption(f"Author: {AUTHOR}")
    st.caption(f"Supervisor: {SUPERVISOR}")
T = LIGHT if light else DARK
V = "".join(f"--{k}:{v};" for k, v in T.items())
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
:root{{{V}}}
html,body,.stApp,[class*="css"]{{font-family:Manrope,system-ui,sans-serif}}
.stApp{{background:var(--bg);color:var(--tx)}}
header[data-testid="stHeader"]{{background:transparent}}
.block-container{{padding-top:1.6rem;max-width:1400px}}
section[data-testid="stSidebar"]{{background:var(--panel);border-right:1px solid var(--line)}}
section[data-testid="stSidebar"] *{{color:var(--tx)}}
.brand{{display:flex;gap:10px;align-items:center;font-weight:800;font-size:15px;padding:4px 4px 14px}}
.logo{{width:30px;height:30px;border-radius:50%;background:var(--ac);display:grid;place-items:center;color:#06222a}}
.st-key-nav div[role="radiogroup"]{{gap:2px}}
.st-key-nav label{{padding:9px 12px;border-radius:8px;width:100%;margin:0;cursor:pointer}}
.st-key-nav label>div:first-child{{display:none}}
.st-key-nav label p{{color:var(--mut);font-weight:500;font-size:14px}}
.st-key-nav label{{position:relative}}
.st-key-nav label:is(:nth-child(1),:nth-child(2),:nth-child(4),:nth-child(7),:nth-child(9)){{margin-top:24px}}
.st-key-nav label:is(:nth-child(1),:nth-child(2),:nth-child(4),:nth-child(7),:nth-child(9))::before{{position:absolute;top:-19px;left:4px;font-size:10px;letter-spacing:.09em;font-weight:800;color:var(--mut)}}
.st-key-nav label:nth-child(1)::before{{content:"OVERVIEW"}}.st-key-nav label:nth-child(2)::before{{content:"MAP WORKSPACE"}}
.st-key-nav label:nth-child(4)::before{{content:"ANALYSIS"}}.st-key-nav label:nth-child(7)::before{{content:"MODEL"}}.st-key-nav label:nth-child(9)::before{{content:"INFO"}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:8px;vertical-align:-1px}}
.status{{font-size:12px;color:var(--mut);margin-top:6px}}
.st-key-nav label:hover p{{color:var(--tx)}}
.st-key-nav label:has(input:checked){{background:color-mix(in srgb,var(--ac) 14%,transparent)}}
.st-key-nav label:has(input:checked) p{{color:var(--tx);font-weight:700}}
.hd{{display:flex;gap:12px;align-items:center;padding-bottom:16px;margin-bottom:18px;border-bottom:1px solid var(--line)}}
.hd .ti{{width:36px;height:36px;border-radius:9px;background:color-mix(in srgb,var(--ac) 16%,transparent);color:var(--ac);display:grid;place-items:center;flex:none}}
.hd h1{{margin:0;font-size:24px;font-weight:800;letter-spacing:-.02em;color:var(--tx);padding:0}}
.hd p{{margin:2px 0 0;color:var(--mut)}}
div[class*="st-key-card"]{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}}
.ct{{margin:0 0 2px;font-size:14px;font-weight:700;color:var(--tx)}}
.cn{{margin:0 0 10px;color:var(--mut);font-size:12px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}}
.kpi b{{display:block;font-size:28px;font-weight:800;letter-spacing:-.02em;line-height:1.15;margin-top:6px;color:var(--tx)}}
.kpi span{{color:var(--mut);font-size:12px}}.kpi span.up{{color:var(--ok)}}.kpi span.dn{{color:var(--bad)}}
table.t{{border-collapse:collapse;width:100%}}
table.t th,table.t td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);font-size:13px;color:var(--tx)}}
table.t th{{color:var(--mut);font-weight:500;font-size:12px}}
.pill{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700}}
.hi{{background:color-mix(in srgb,var(--bad) 20%,transparent);color:var(--bad)}}
.md{{background:color-mix(in srgb,var(--ac2) 20%,transparent);color:var(--ac2)}}
.lo{{background:color-mix(in srgb,var(--ok) 20%,transparent);color:var(--ok)}}
.row{{display:grid;grid-template-columns:120px 1fr 52px;gap:10px;align-items:center;margin:9px 0;font-size:13px}}
.bar{{height:8px;border-radius:4px;background:var(--panel2);overflow:hidden}}.bar i{{display:block;height:100%}}
.info{{border-left:3px solid var(--ac);background:color-mix(in srgb,var(--ac) 9%,transparent);padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;color:var(--tx)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;font-size:12px;color:var(--mut)}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}}
.prose p{{margin:0 0 10px;color:var(--tx);font-size:14px}}
div[data-baseweb="select"]>div{{background:var(--panel);border-color:var(--line);color:var(--tx)}}
div[data-baseweb="select"] *{{color:var(--tx)}}
.kdu-footer{{margin-top:30px;padding-top:14px;border-top:1px solid var(--line);color:var(--mut);font-size:12px;text-align:center}}
.kdu-footer b{{color:var(--tx)}}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ DATA
@st.cache_data
def load_kpis():
    with open(PROC / "dashboard_kpis.json") as f:
        return json.load(f)


def band(region, suffix):
    with rasterio.open(PROC / f"{region}_{suffix}.tif") as s:
        return s.read(1)


def stretch(a, lo_p=2, hi_p=98):
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def _rgba(shape, painters):
    out = np.zeros((*shape, 4), np.uint8)
    for mask, colour in painters:
        out[mask] = colour
    return out


def _down(arr, mode, smooth=False, max_px=1600):
    im = Image.fromarray(arr, mode)
    im.thumbnail((max_px, max_px), Image.LANCZOS if smooth else Image.NEAREST)
    return np.array(im)


@st.cache_data(show_spinner=False)
def base_layers(region):
    with rasterio.open(PROC / f"{region}_feature_stack.tif") as s:
        names = list(s.descriptions)
        rgb = np.dstack([stretch(s.read(names.index(n) + 1).astype(float)) for n in ("B4", "B3", "B2")])
        bounds = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    flood, exposed = band(region, "flood_prone_mask") == 1, band(region, "exposure_map") == 1
    shape = rgb.shape[:2]
    rgb8 = (rgb * 255).astype(np.uint8)
    return {"bounds": bounds,
            "rgb": _down(np.dstack([rgb8, np.full(shape, 255, np.uint8)]), "RGBA", True),
            "flood": _down(_rgba(shape, [(flood & ~exposed, (59, 130, 246, 170))]), "RGBA"),
            "exposed": _down(_rgba(shape, [(exposed, (255, 59, 48, 235))]), "RGBA")}


@st.cache_data(show_spinner=False)
def built_mask(region, epoch):
    with rasterio.open(PROC / f"{region}_feature_stack.tif") as s:
        tr, crs, h, w = s.transform, s.crs, s.height, s.width
    with rasterio.open(RAW / region / f"{region}_ghsl_{epoch}.tif") as s:
        arr, t, c = s.read(1).astype(np.float32), s.transform, s.crs
    dst = np.empty((h, w), np.float32)
    reproject(arr, dst, src_transform=t, src_crs=c, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
    return dst > 1000


@st.cache_data(show_spinner=False)
def growth_layers(region, e0, e1):
    flood = band(region, "flood_prone_mask") == 1
    b0, b1 = built_mask(region, e0), built_mask(region, e1)
    new = b1 & ~b0
    shape = new.shape
    return {"safe": _down(_rgba(shape, [(new & ~flood, (245, 196, 0, 235))]), "RGBA"),
            "prone": _down(_rgba(shape, [(new & flood, (217, 70, 239, 235))]), "RGBA"),
            "b0": _down(_rgba(shape, [(b0, (34, 211, 238, 220))]), "RGBA"),
            "b1": _down(_rgba(shape, [(b1, (34, 211, 238, 220))]), "RGBA"),
            "prone_share": float(100 * (new & flood).sum() / max(new.sum(), 1))}


@st.cache_data(show_spinner=False)
def snapshot(region):
    L = base_layers(region)
    im = Image.fromarray(L["rgb"])
    for k in ("flood", "exposed"):
        im = Image.alpha_composite(im, Image.fromarray(L[k]))
    return np.array(im.convert("RGB"))


kpis = load_kpis()
RK = list(REGIONS)
EP = kpis[RK[0]]["ghsl_epochs"]

# ------------------------------------------------------------------ UI HELPERS
_n = 0


@contextmanager
def card(title, note=""):
    global _n
    _n += 1
    with st.container(key=f"card_{_n}"):
        st.markdown(f'<p class="ct">{title}</p><p class="cn">{note}</p>', unsafe_allow_html=True)
        yield


def kpi_row(items):
    for col, (label, value, sub, cls) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><span>{label}</span><b>{value}</b><span class="{cls}">{sub}</span></div>', unsafe_allow_html=True)


def table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<div style="overflow-x:auto"><table class="t"><tr>{head}</tr>{body}</table></div>', unsafe_allow_html=True)


def legend(items):
    st.markdown('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{t}</span>' for t, c in items) + "</div>", unsafe_allow_html=True)


def progress(rows):
    st.markdown("".join(f'<div class="row"><span>{l}</span><div class="bar"><i style="width:{v}%;background:{c}"></i></div><b>{v:.1f}%</b></div>' for l, v, c in rows), unsafe_allow_html=True)


def style(fig, h=300, legend_on=True):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=T["mut"], family="Manrope, sans-serif"),
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=legend_on, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor=T["line"], zeroline=False)
    fig.update_yaxes(gridcolor=T["line"], zeroline=False)
    return fig


def show(fig, key):
    st.plotly_chart(fig, key=key, **WIDE)


def csv_button(df, name, key):
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), name, "text/csv", key=key)


CARTO = "© OpenStreetMap contributors © CARTO"
BASEMAPS = {  # explicit tile URLs: none of these need an API key
    "Dark": ("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", CARTO),
    "Light": ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", CARTO),
    "OpenStreetMap": ("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap contributors"),
    "Satellite (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Tiles © Esri"),
}


def tile_layer(name):
    url, attr = BASEMAPS[name]
    return folium.TileLayer(tiles=url, attr=attr, name=name, subdomains="abc", max_zoom=19)


def add_overlays(m, bounds, items):
    w, s, e, n = bounds
    for img, opacity, name in items:
        folium.raster_layers.ImageOverlay(image=img, bounds=[[s, w], [n, e]], opacity=opacity, name=name).add_to(m)


def swatch(colour, text):
    return f'<span class="sw" style="background:{colour}"></span>{text}'


def generate_summary_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, PROJECT_TITLE, ln=True)
    pdf.set_font("Helvetica", "", 11); pdf.cell(0, 8, INSTITUTION, ln=True)
    pdf.cell(0, 8, f"Author: {AUTHOR}  |  Supervisor: {SUPERVISOR}", ln=True); pdf.ln(6)
    for region, label in REGIONS.items():
        k = kpis[region]
        pdf.set_font("Helvetica", "B", 13); pdf.cell(0, 8, label, ln=True)
        pdf.set_font("Helvetica", "", 10)
        for row in [f"Total built-up area: {k['total_builtup_km2']:.1f} km2",
                    f"Flood-exposed built-up area: {k['exposed_builtup_km2']:.1f} km2 ({k['pct_builtup_exposed']:.1f}%)",
                    f"Estimated exposed buildings: {k['estimated_exposed_buildings']:,} of {k['total_buildings']:,}",
                    f"Estimated population exposed: {k['population_exposed']:,.0f}",
                    f"Growth rate: {k['growth_rate_km2_per_year']:.2f} km2/year (R2={k['growth_r_squared']:.2f})",
                    f"Risk tiers: High={k['risk_pct_high']:.1f}%  Medium={k['risk_pct_medium']:.1f}%  Low={k['risk_pct_low']:.1f}%"]:
            pdf.cell(0, 7, row, ln=True)
        pdf.ln(4)
    return bytes(pdf.output())


# ------------------------------------------------------------------ HEADER
pi = PAGES.index(page)
st.markdown(f'<div class="hd"><span class="ti"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[pi]}</svg></span><div><h1>{page[3:]}</h1><p>{SUBS[pi]}</p></div></div>', unsafe_allow_html=True)
CA, CB = T["ac"], T["ac2"]
PERF = pd.DataFrame([("A→A baseline", .882, .881), ("A→B transfer", .756, .766), ("B→B baseline", .906, .901), ("B→A transfer", .786, .774)],
                    columns=["Direction", "Balanced Accuracy", "Macro F1"])


def perf_chart(key):
    fig = go.Figure([go.Bar(x=PERF["Direction"], y=PERF["Balanced Accuracy"], name="Balanced accuracy", marker_color=CA, text=PERF["Balanced Accuracy"], textposition="outside"),
                     go.Bar(x=PERF["Direction"], y=PERF["Macro F1"], name="Macro F1", marker_color=T["bad"], text=PERF["Macro F1"], textposition="outside")])
    fig.update_layout(barmode="group", yaxis_range=[0, 1.05])
    show(style(fig, 320), key)


# ================================================================== PAGES
if pi == 0:
    K = kpis.values()
    tb, te = sum(k["total_builtup_km2"] for k in K), sum(k["exposed_builtup_km2"] for k in K)
    tbl_, teb = sum(k["total_buildings"] for k in K), sum(k["estimated_exposed_buildings"] for k in K)
    pop, gr = sum(k["population_exposed"] for k in K), np.mean([k["growth_rate_km2_per_year"] for k in K])
    kpi_row([("Total built-up area", f"{tb:.1f} km²", "Both regions", ""), ("Flood-exposed area", f"{te:.1f} km²", f"{100*te/tb:.1f}% of total", "dn"),
             ("Buildings mapped", f"{tbl_:,}", "Open Buildings v3", ""), ("Exposed buildings", f"{teb:,}", f"{100*teb/tbl_:.1f}% of mapped", "dn")])
    kpi_row([("Population exposed (est.)", f"{pop:,.0f}", "WorldPop-derived", ""), ("Avg growth rate", f"{gr:.2f} km²/yr", "GHSL trend", "up"),
             ("Regions analysed", str(len(RK)), " · ".join(SHORT.values()), ""), ("Study period", f"{EP[0]}–{EP[-1]}", "GHSL epochs", "")])
    best = max(RK, key=lambda r: kpis[r]["pct_builtup_exposed"])
    st.markdown(f'<div class="info"><b>Key finding:</b> {REGIONS[best]} has the highest share of built-up area in flood-prone terrain '
                f'({kpis[best]["pct_builtup_exposed"]:.1f}%), with about {kpis[best]["estimated_exposed_buildings"]:,} buildings exposed.</div>', unsafe_allow_html=True)
    st.write("")
    with st.spinner("Loading region snapshots..."):
        for col, region in zip(st.columns(2), RK):
            k = kpis[region]
            with col, card(REGIONS[region], f"{k['pct_builtup_exposed']:.1f}% of built-up area flood-exposed · {k['growth_rate_km2_per_year']:+.2f} km²/yr growth"):
                st.image(snapshot(region), **IMG_WIDE)
                legend([("Flood-prone", "#3b82f6"), ("Exposed built-up", "#ff3b30")])
                st.button("Open in Map Viewer →", key=f"open_{region}", on_click=lambda r=region: st.session_state.update(nav=PAGES[1], mv_region=r))

elif pi == 1:
    left, mid, right = st.columns([1.15, 3.5, 1.25])
    with left:
        with card("Data frame"):
            region = st.selectbox("Study area", RK, format_func=lambda r: REGIONS[r], key="mv_region")
            basemap = st.selectbox("Basemap", list(BASEMAPS), index=1 if light else 0, key="mv_base")
        with card("Layers", "Tick to show on the map"):
            on_rgb = st.checkbox("🛰️ Sentinel-2 imagery", False, key="l_rgb")
            on_fl = st.checkbox("🟦 Flood-prone terrain", True, key="l_fl")
            on_ex = st.checkbox("🟥 Exposed built-up", True, key="l_ex")
            on_gs = st.checkbox("🟨 New growth · flood-safe", False, key="l_gs")
            on_gp = st.checkbox("🟪 New growth · flood-prone", False, key="l_gp")
            opacity = st.slider("Layer opacity", 0.2, 1.0, 0.85, 0.05, key="l_op")
            if on_gs or on_gp:
                y0, y1 = st.select_slider("Growth period", EP, (EP[0], EP[-1]), key="mv_period")
    B = base_layers(region)
    items = [(B["rgb"], 1.0, "Imagery")] * on_rgb + [(B["flood"], opacity, "Flood-prone")] * on_fl + [(B["exposed"], opacity, "Exposed")] * on_ex
    G = None
    if (on_gs or on_gp) and y0 < y1:
        with st.spinner("Computing built-up growth..."):
            G = growth_layers(region, y0, y1)
        items += [(G["safe"], opacity, "Growth safe")] * on_gs + [(G["prone"], opacity, "Growth prone")] * on_gp
    w, s_, e, n = B["bounds"]
    m = folium.Map(location=[(s_ + n) / 2, (w + e) / 2], zoom_start=12, control_scale=True, tiles=None)
    tile_layer(basemap).add_to(m)
    add_overlays(m, B["bounds"], items)
    Fullscreen().add_to(m)
    MeasureControl(position="topleft", primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
    MousePosition(position="bottomleft", separator=" | ", num_digits=5, prefix="Lat | Lon").add_to(m)
    MiniMap(tile_layer=tile_layer(basemap), toggle_display=True, position="bottomright").add_to(m)
    m.fit_bounds([[s_, w], [n, e]])
    with mid:
        st_folium(m, width=None, height=640, returned_objects=[],
                  key=f"mv_{region}_{basemap}_{on_rgb}{on_fl}{on_ex}{on_gs}{on_gp}_{opacity}_{'' if G is None else (y0, y1)}")
        st.markdown(f'<div class="status">{REGIONS[region]} · basemap: {basemap} · {len(items)} data layer(s) · use the ruler (top-left) to measure and the readout (bottom-left) for coordinates</div>', unsafe_allow_html=True)
    k = kpis[region]
    with right:
        with card("Readout", "Statistics for this study area"):
            kpi_row([("% built-up exposed", f"{k['pct_builtup_exposed']:.1f}%", f"{k['exposed_builtup_km2']:.1f} km² flood-prone ∩ built-up", "dn")])
            kpi_row([("Exposed buildings", f"{k['estimated_exposed_buildings']:,}", f"of {k['total_buildings']:,} mapped", "")])
            kpi_row([("Population exposed", f"{k['population_exposed']:,.0f}", "WorldPop estimate", "")])
            if G is not None:
                kpi_row([(f"Growth in flood zone {y0}–{y1}", f"{G['prone_share']:.1f}%", "of new built-up area", "dn")])
        with card("Legend"):
            rows = [(on_fl, "#3b82f6", "Flood-prone terrain"), (on_ex, "#ff3b30", "Exposed built-up"), (on_gs, "#f5c400", "New growth, flood-safe"), (on_gp, "#d946ef", "New growth, flood-prone")]
            st.markdown("<br>".join(swatch(c, t) for on, c, t in rows if on) or '<span class="cn">No overlay active</span>', unsafe_allow_html=True)

elif pi == 2:
    c0, c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4, 1.4])
    region = c0.selectbox("Study area", RK, format_func=lambda r: REGIONS[r], key="tc_region")
    ea = c1.selectbox("Left epoch", EP[:-1], index=0, key="tc_a")
    eb = c2.selectbox("Right epoch", EP[1:], index=len(EP) - 2, key="tc_b")
    show_fl = c3.checkbox("Overlay flood-prone terrain", True, key="tc_fl")
    tc_base = c4.selectbox("Basemap", list(BASEMAPS), index=1 if light else 0, key="tc_base")
    if ea >= eb:
        st.warning("Choose a right epoch that is later than the left epoch.")
    else:
        with st.spinner("Preparing synchronised maps..."):
            B, G = base_layers(region), growth_layers(region, ea, eb)
        w, s_, e, n = B["bounds"]
        h1, h2 = st.columns(2)
        h1.markdown(f"**◀ Built-up in {ea}**")
        h2.markdown(f"**Built-up in {eb} + new growth ▶**")
        m = DualMap(location=[(s_ + n) / 2, (w + e) / 2], zoom_start=12, tiles=None)
        tile_layer(tc_base).add_to(m.m1)
        tile_layer(tc_base).add_to(m.m2)
        add_overlays(m.m1, B["bounds"], [(G["b0"], .9, f"Built-up {ea}")] + [(B["flood"], .6, "Flood-prone")] * show_fl)
        add_overlays(m.m2, B["bounds"], [(G["b1"], .9, f"Built-up {eb}")] + [(B["flood"], .6, "Flood-prone")] * show_fl + [(G["safe"], .95, "New growth safe"), (G["prone"], .95, "New growth prone")])
        MousePosition(position="bottomleft", num_digits=5, prefix="Lat | Lon").add_to(m.m1)
        components.html(m.get_root().render(), height=560)
        legend([("Built-up", "#22d3ee"), ("Flood-prone terrain", "#3b82f6"), ("New growth, flood-safe", "#f5c400"), ("New growth, flood-prone", "#d946ef")])
        v = kpis[region]["ghsl_builtup_km2_by_epoch"]
        a, b = v[EP.index(ea)], v[EP.index(eb)]
        kpi_row([(f"Built-up {ea}", f"{a:.1f} km²", "", ""), (f"Built-up {eb}", f"{b:.1f} km²", f"{100*(b-a)/a:+.1f}% change", "up"),
                 ("New built-up area", f"{b-a:.1f} km²", f"{ea} → {eb}", ""), ("Growth inside flood-prone zones", f"{G['prone_share']:.1f}%", f"of new built-up area; land share {dict(zip(RK, (43.5, 75.1)))[region]:.1f}%", "dn")])

elif pi == 3:
    y0, y1 = st.select_slider("Compare years", options=EP, value=(EP[0], EP[-1]))
    with card("Change between epochs", f"{y0} → {y1}"):
        for col, region in zip(st.columns(2), RK):
            v = kpis[region]["ghsl_builtup_km2_by_epoch"]
            a, b = v[EP.index(y0)], v[EP.index(y1)]
            p = 100 * (b - a) / a if a > 0 else 0
            col.markdown(f'<div class="kpi"><span>{REGIONS[region]}</span><b>{b:.1f} km²</b><span class="{"up" if p >= 0 else "dn"}">{p:+.1f}% since {y0}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1, card("Built-up surface over time", "km², JRC GHSL"):
        fig = go.Figure([go.Scatter(x=EP, y=kpis[r]["ghsl_builtup_km2_by_epoch"], mode="lines+markers", name=REGIONS[r], line=dict(width=3, color=c)) for r, c in zip(RK, (CA, CB))])
        show(style(fig, 320), "ghsl_line")
    with c2, card("Growth rate", "km² per year"):
        fig = go.Figure(go.Bar(x=[SHORT[r] for r in RK], y=[kpis[r]["growth_rate_km2_per_year"] for r in RK], marker_color=[CA, CB],
                               text=[f"{kpis[r]['growth_rate_km2_per_year']:.2f}" for r in RK], textposition="outside"))
        show(style(fig, 320, False), "gr_bar")
    with card("Encroachment into flood-prone terrain", f"Share of new built-up area ({y0}–{y1}) that falls in flood-prone terrain"):
        if y0 < y1:
            with st.spinner("Computing encroachment..."):
                shares = [growth_layers(r, y0, y1)["prone_share"] for r in RK]
            base = [43.5, 75.1]  # flood-prone share of land (water excluded), Region A, Region B
            assert len(RK) == 2
            fig = go.Figure([
                go.Bar(name="Share of new growth in flood-prone terrain", x=[SHORT[r] for r in RK], y=shares, marker_color=[T["bad"]] * 2, text=[f"{v:.1f}%" for v in shares], textposition="outside"),
                go.Bar(name="Flood-prone share of land (water excluded)", x=[SHORT[r] for r in RK], y=base, marker_color="#94a3b8", text=[f"{v:.1f}%" for v in base], textposition="outside"),
            ])
            fig.update_layout(barmode="group", yaxis_range=[0, max(shares + base) * 1.3 + 1], legend=dict(orientation="h", y=-0.25))
            show(style(fig, 260, False), "encroach")
        else:
            st.caption("Pick two different years to compute encroachment.")
    st.markdown('<div class="info">An earlier from-scratch yearly classifier trend was dominated by inter-annual radiometric noise (p &gt; 0.5, non-significant). '
                'GHSL was adopted instead, giving significant trends (p &lt; 0.01, R² &gt; 0.96) in both regions.</div>', unsafe_allow_html=True)

elif pi == 4:
    c1, c2 = st.columns(2)
    with c1, card("% built-up area exposed", "Flood-prone share of built-up"):
        fig = go.Figure(go.Bar(x=[SHORT[r] for r in RK], y=[kpis[r]["pct_builtup_exposed"] for r in RK], marker_color=[CA, CB],
                               text=[f"{kpis[r]['pct_builtup_exposed']:.1f}%" for r in RK], textposition="outside"))
        fig.update_layout(yaxis_range=[0, 100])
        show(style(fig, 300, False), "cmp_exp")
    with c2, card("Growth rate", "km² per year"):
        fig = go.Figure(go.Bar(x=[SHORT[r] for r in RK], y=[kpis[r]["growth_rate_km2_per_year"] for r in RK], marker_color=[CA, CB],
                               text=[f"{kpis[r]['growth_rate_km2_per_year']:.2f}" for r in RK], textposition="outside"))
        show(style(fig, 300, False), "cmp_gr")
    a, b = kpis[RK[0]], kpis[RK[1]]
    rows = [("Built-up area (km²)", "total_builtup_km2"), ("Exposed area (km²)", "exposed_builtup_km2"), ("% exposed", "pct_builtup_exposed"),
            ("Buildings total", "total_buildings"), ("Buildings exposed", "estimated_exposed_buildings"), ("Population exposed (est.)", "population_exposed"),
            ("Growth rate (km²/yr)", "growth_rate_km2_per_year")]
    fmt = lambda v: f"{v:,.2f}" if isinstance(v, float) and v < 1000 else f"{v:,.0f}"
    with card("Side-by-side snapshot"):
        table(["Metric", REGIONS[RK[0]], REGIONS[RK[1]]], [[m, fmt(a[k]), fmt(b[k])] for m, k in rows])
    csv_button(pd.DataFrame([{"Metric": m, REGIONS[RK[0]]: a[k], REGIONS[RK[1]]: b[k]} for m, k in rows]), "regional_comparison.csv", "cmp_csv")

elif pi == 5:
    for col, region in zip(st.columns(2), RK):
        k = kpis[region]
        with col, card(REGIONS[region], "Share of study area by HAND tier"):
            fig = go.Figure(go.Pie(labels=["High", "Medium", "Low"], values=[k["risk_pct_high"], k["risk_pct_medium"], k["risk_pct_low"]], hole=.6,
                                   marker_colors=[T["bad"], T["ac2"], T["ok"]], textinfo="percent"))
            show(style(fig, 260), f"pie_{region}")
            progress([("High (< 2 m)", k["risk_pct_high"], T["bad"]), ("Medium (2–5 m)", k["risk_pct_medium"], T["ac2"]), ("Low (> 5 m)", k["risk_pct_low"], T["ok"])])
    with card("Risk summary"):
        table(["Region", "High", "Medium", "Low", "Population exposed"],
              [[REGIONS[r], f'<span class="pill hi">{k["risk_pct_high"]:.1f}%</span>', f'<span class="pill md">{k["risk_pct_medium"]:.1f}%</span>',
                f'<span class="pill lo">{k["risk_pct_low"]:.1f}%</span>', f'{k["population_exposed"]:,.0f}'] for r, k in kpis.items()])
    csv_button(pd.DataFrame([{"Region": REGIONS[r], "High %": k["risk_pct_high"], "Medium %": k["risk_pct_medium"], "Low %": k["risk_pct_low"],
                              "Population Exposed": k["population_exposed"]} for r, k in kpis.items()]), "risk_assessment.csv", "risk_csv")

elif pi == 6:
    kpi_row([("A→B degradation", "−12.6 pts", "88.2% → 75.6% balanced acc.", "dn"), ("B→A degradation", "−12.0 pts", "90.6% → 78.6% balanced acc.", "dn"),
             ("Best baseline", "90.6%", "Train B → Test B", "up"), ("Transfer range", "75.6–78.6%", "Cross-region", "")])
    c1, c2 = st.columns([3, 2])
    with c1, card("Transferability comparison", "Balanced accuracy vs macro F1"):
        perf_chart("tr_chart")
    with c2, card("Metrics"):
        table(["Direction", "Bal. acc.", "Macro F1"], [[d, a, f] for d, a, f in PERF.itertuples(index=False)])
        csv_button(PERF, "transferability_metrics.csv", "tr_csv")
    with card("Why does transfer degrade? (SHAP)", "Depth-limited surrogate model"):
        st.markdown('<div class="prose"><p>The Region-A model leans primarily on <b>NDBI</b> (mean |SHAP| = 0.112), consistent with built-up index theory. '
                    'The Region-B model instead relies mainly on <b>raw NIR reflectance and SAR backscatter</b>, with NDBI dropping to secondary rank.</p>'
                    '<p>Each model partly learned region-specific spectral signatures rather than one universal built-up concept, a plausible explanation for the ~12-point gap in both directions.</p></div>', unsafe_allow_html=True)
        doc = Path("docs/phase5_transferability_synthesis.md")
        if doc.exists():
            with st.expander("📄 Read full synthesis writeup"):
                st.markdown(doc.read_text())

elif pi == 7:
    c1, c2 = st.columns([3, 2])
    with c1, card("Built-up classifier performance", "Spatially-blocked checkerboard validation"):
        perf_chart("perf_chart")
    with c2, card("Finding"):
        st.markdown('<div class="info">Cross-region transfer shows a consistent ~12-point balanced-accuracy drop. SHAP attributes it to NDBI-driven evidence in Region A '
                    'versus visible-band/SAR-driven evidence in Region B.</div>', unsafe_allow_html=True)
        csv_button(PERF, "model_performance.csv", "perf_csv")
    with card("Known data-quality issues (resolved)"):
        table(["Issue", "Impact", "Resolution"], [
            ["HAND boundary drainage artifact", "Up to 27% NaN pixels", "Re-computed on buffered DEM (+5.5 km margin); &lt; 5% NaN"],
            ["WorldPop resampling area-scaling bug", "~100× population overcount", "Area-ratio division; verified against native totals"],
            ["From-scratch yearly classifier noise", "Non-significant trend (p &gt; 0.5)", "Switched to JRC GHSL series (p &lt; 0.01)"],
            ["Open Buildings Shapefile export conflict", "Export failure", "Stripped conflicting Geometry-typed property"]])

else:
    c1, c2 = st.columns(2)
    with c1, card("Pipeline overview"):
        st.markdown("""<div class="prose">
<p><b>1. Data acquisition.</b> Sentinel-2/1, Copernicus DEM, ESA WorldCover, WorldPop, Open Buildings and JRC GHSL via Google Earth Engine.</p>
<p><b>2. Built-up classification.</b> Random Forest on spectral bands, NDVI/NDBI, SAR backscatter and elevation; spatially-blocked checkerboard split.</p>
<p><b>3. Flood-proneness (HAND).</b> D8 flow routing on a buffered DEM, thresholded at 5 m.</p>
<p><b>4. Exposure overlay.</b> Built-up ∩ flood-prone terrain.</p>
<p><b>5. Explainability.</b> SHAP on a depth-limited surrogate model.</p>
<p><b>6. Growth trends.</b> GHSL built-up surface, 2000–2025.</p></div>""", unsafe_allow_html=True)
    with c2, card("Known limitations"):
        st.markdown("""<div class="prose">
<p>Cross-region transfer shows a ~12-point balanced-accuracy gap in both directions.</p>
<p>Administrative-boundary exposure breakdown is not yet implemented (needs GADM boundaries).</p>
<p>Population exposure is estimated from WorldPop gridded density, not census-verified counts.</p></div>""", unsafe_allow_html=True)
    with card("Data sources"):
        table(["Dataset", "Source", "Resolution", "Use"], [
            ["Sentinel-2 SR Harmonized", "ESA/Copernicus (GEE)", "10 m", "Optical bands, NDVI/NDBI"], ["Sentinel-1 GRD", "ESA/Copernicus (GEE)", "10 m", "SAR backscatter"],
            ["Copernicus DEM GLO-30", "ESA (GEE)", "30 m", "Elevation, HAND"], ["ESA WorldCover v200", "ESA (GEE)", "10 m", "Land-cover labels"],
            ["WorldPop", "WorldPop (GEE)", "100 m", "Population density"], ["Google Open Buildings v3", "Google Research (GEE)", "Vector", "Building footprints"],
            ["JRC GHSL Built-up Surface", "JRC (GEE)", "100 m", "Growth trend"]])
    st.download_button("📥 Download Summary Report (PDF)", generate_summary_pdf(), "fyp_summary_report.pdf", "application/pdf")

st.markdown(f'<div class="kdu-footer"><b>{PROJECT_TITLE}</b><br>{INSTITUTION}<br>Author: {AUTHOR} · Supervisor: {SUPERVISOR}<br>'
            'Data: Google Earth Engine (Sentinel-1/2, Copernicus DEM, ESA WorldCover, WorldPop, Google Open Buildings, JRC GHSL)</div>', unsafe_allow_html=True)