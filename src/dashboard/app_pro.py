"""
Professional FYP dashboard: Flood-Exposed Housing Detection System.
7 pages, KPI cards, interactive Folium map, Plotly charts, dark/light
toggle, PDF/CSV/PNG export.
"""
import streamlit as st
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import json
import io
import base64
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Flood-Exposed Housing Detection | KDU FYP", page_icon="🌊", layout="wide")

REGIONS = {
    "region_a_kelani": "Lower Kelani Basin (Sri Lanka)",
    "region_b_dakar": "Dakar Northern Suburbs (Senegal)",
}
PROJECT_TITLE = "Flood-Exposed Housing Detection System"
AUTHOR = "El Hadji Malick Diouf"  # <-- edit to your actual name
SUPERVISOR = "[Supervisor Name]"  # <-- edit
INSTITUTION = "Faculty of Computing, General Sir John Kotelawala Defence University"

# ---------------------------------------------------------------- THEME
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def inject_css(theme):
    if theme == "dark":
        bg, card, text, subtext, accent, border = "#0e1420", "#1a2233", "#ffffff", "#a8b3c4", "#4C9AFF", "#2e3a4f" 
    else:
        bg, card, text, subtext, accent, border = "#f5f7fa", "#ffffff", "#152238", "#5c6b7a", "#2563eb", "#dbe2ea"

    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; }}
        div[data-testid="stMetric"] {{
            background-color: {card};
            border: 1px solid {border};
            border-left: 4px solid {accent};
            border-radius: 10px;
            padding: 14px 18px;
        }}
        div[data-testid="stMetricLabel"] label {{ color: {subtext} !important; font-weight: 600; }}
        div[data-testid="stMetricValue"] {{ color: {text} !important; }}
        h1, h2, h3, h4 {{ color: {text}; font-family: 'Segoe UI', sans-serif; }}
        p, span, div {{ color: {text}; }}
        .kdu-footer {{
            margin-top: 40px; padding: 16px 0; border-top: 1px solid {border};
            color: {subtext}; font-size: 0.85em; text-align: center;
        }}
        .kdu-header {{
            padding: 8px 0 20px 0; border-bottom: 2px solid {accent}; margin-bottom: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.theme)

# ---------------------------------------------------------------- DATA LOADERS
@st.cache_data
def load_kpis():
    with open(Path("data/processed") / "dashboard_kpis.json") as f:
        return json.load(f)

@st.cache_data
def load_stretched_rgb(region):
    path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        red = src.read(names.index("B4") + 1).astype(float)
        green = src.read(names.index("B3") + 1).astype(float)
        blue = src.read(names.index("B2") + 1).astype(float)
        bounds = src.bounds
        crs = src.crs
    def stretch(band, low=2, high=98):
        lo, hi = np.percentile(band, [low, high])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    latlon_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
    return rgb, latlon_bounds

@st.cache_data
def load_growth_overlay_array(region):
    """Builds the red/yellow encroachment overlay as a clean array + lat/lon
    bounds, computed directly from GHSL rasters -- avoids relying on the
    pre-rendered matplotlib PNGs, which include titles/padding that don't
    align to real geographic coordinates on an interactive map."""
    rgb, bounds = load_stretched_rgb(region)

    ref_path = Path("data/processed") / f"{region}_feature_stack.tif"
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()

    def load_and_resample_ghsl(epoch):
        ghsl_path = Path("data/raw") / region / f"{region}_ghsl_{epoch}.tif"
        with rasterio.open(ghsl_path) as s:
            arr = s.read(1).astype(np.float32)
            t, c = s.transform, s.crs
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        from rasterio.warp import reproject, Resampling
        reproject(source=arr, destination=dst, src_transform=t, src_crs=c,
                   dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                   resampling=Resampling.bilinear)
        return dst

    ghsl_2000 = load_and_resample_ghsl(2000)
    ghsl_2025 = load_and_resample_ghsl(2025)
    builtup_2000 = ghsl_2000 > 1000
    builtup_2025 = ghsl_2025 > 1000
    new_growth = builtup_2025 & (~builtup_2000)

    flood_prone = load_band(region, "flood_prone_mask")

    overlay = rgb.copy()
    growth_safe = new_growth & (flood_prone == 0)
    growth_flood_prone = new_growth & (flood_prone == 1)
    overlay[growth_safe] = overlay[growth_safe] * 0.3 + np.array([1.0, 0.85, 0.0]) * 0.7
    overlay[growth_flood_prone] = overlay[growth_flood_prone] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7

    return overlay, bounds


@st.cache_data
def load_band(region, suffix):
    with rasterio.open(Path("data/processed") / f"{region}_{suffix}.tif") as src:
        return src.read(1)

@st.cache_data
def load_stack_band(region, band_name):
    with rasterio.open(Path("data/processed") / f"{region}_feature_stack.tif") as src:
        names = list(src.descriptions)
        return src.read(names.index(band_name) + 1)

def fig_download_button(fig, filename, key, is_plotly=True):
    if is_plotly:
        buf = fig.to_image(format="png", scale=2)
        st.download_button("📥 Download PNG", data=buf, file_name=filename, mime="image/png", key=key)
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button("📥 Download PNG", data=buf.getvalue(), file_name=filename, mime="image/png", key=key)

def df_download_button(df, filename, key):
    st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                        file_name=filename, mime="text/csv", key=key)

def generate_summary_pdf(kpis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, PROJECT_TITLE, ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, INSTITUTION, ln=True)
    pdf.cell(0, 8, f"Author: {AUTHOR}  |  Supervisor: {SUPERVISOR}", ln=True)
    pdf.ln(6)
    for region, label in REGIONS.items():
        k = kpis[region]
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, label, ln=True)
        pdf.set_font("Helvetica", "", 10)
        rows = [
            f"Total built-up area: {k['total_builtup_km2']:.1f} km2",
            f"Flood-exposed built-up area: {k['exposed_builtup_km2']:.1f} km2 ({k['pct_builtup_exposed']:.1f}%)",
            f"Estimated exposed buildings: {k['estimated_exposed_buildings']:,} of {k['total_buildings']:,}",
            f"Estimated population exposed: {k['population_exposed']:,.0f}",
            f"Growth rate: {k['growth_rate_km2_per_year']:.2f} km2/year (R2={k['growth_r_squared']:.2f})",
            f"Risk tiers: High={k['risk_pct_high']:.1f}%  Medium={k['risk_pct_medium']:.1f}%  Low={k['risk_pct_low']:.1f}%",
        ]
        for row in rows:
            pdf.cell(0, 7, row, ln=True)
        pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 8, "Auto-generated summary -- see full dashboard for details and methodology.", ln=True)
    return bytes(pdf.output())

kpis = load_kpis()

# ---------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.title("🌊 FYP Dashboard")
    st.caption(INSTITUTION)
    st.markdown("---")
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == "dark" else 1, horizontal=True)
    new_theme = theme_choice.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    st.markdown("---")
    page = st.radio("Navigate", [
        "1. Executive Summary", "2. Flood Exposure Map", "3. Built-up Growth (GHSL)",
        "4. Regional Comparison", "5. Risk Assessment", "6. Transferability",
        "7. Data Quality & Model Performance", "8. Methodology & Sources",
    ])
    st.markdown("---")
    st.caption(f"Author: {AUTHOR}")
    st.caption(f"Supervisor: {SUPERVISOR}")

# ================================================================== PAGE 1
if page == "1. Executive Summary":
    st.markdown(f'<div class="kdu-header"><h1>🌊 {PROJECT_TITLE}</h1>'
                f'<p>Cross-region built-up mapping, flood-proneness, and exposure analysis</p></div>',
                unsafe_allow_html=True)

    total_builtup = sum(k["total_builtup_km2"] for k in kpis.values())
    total_exposed = sum(k["exposed_builtup_km2"] for k in kpis.values())
    total_buildings = sum(k["total_buildings"] for k in kpis.values())
    total_exp_buildings = sum(k["estimated_exposed_buildings"] for k in kpis.values())
    total_pop_exposed = sum(k["population_exposed"] for k in kpis.values())
    avg_growth = np.mean([k["growth_rate_km2_per_year"] for k in kpis.values()])

    with st.spinner("Loading key metrics..."):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏘️ Total Built-up Area", f"{total_builtup:.1f} km²")
        c2.metric("🌊 Flood-Exposed Area", f"{total_exposed:.1f} km²", f"{100*total_exposed/total_builtup:.1f}% of total")
        c3.metric("🏠 Buildings Mapped", f"{total_buildings:,}")
        c4.metric("⚠️ Exposed Buildings", f"{total_exp_buildings:,}", f"{100*total_exp_buildings/total_buildings:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("👥 Population Exposed (est.)", f"{total_pop_exposed:,.0f}")
        c6.metric("📈 Avg Growth Rate", f"{avg_growth:.2f} km²/yr")
        c7.metric("🗺️ Regions Analyzed", "2")
        c8.metric("📅 Study Period", "2000–2025")

    st.markdown("---")
    st.subheader("Region Snapshots")
    cols = st.columns(2)
    for col, (region, label) in zip(cols, REGIONS.items()):
        with col:
            st.markdown(f"**{label}**")
            rgb, _ = load_stretched_rgb(region)
            st.image(rgb, use_container_width=True)
            k = kpis[region]
            st.caption(f"{k['pct_builtup_exposed']:.1f}% of built-up area flood-exposed · "
                       f"{k['growth_rate_km2_per_year']:+.2f} km²/yr growth")

# ================================================================== PAGE 2
elif page == "2. Flood Exposure Map":
    st.header("🗺️ Flood Exposure Map")
    region_key = st.selectbox("Region", list(REGIONS.keys()), format_func=lambda k: REGIONS[k])

    with st.spinner("Rendering interactive map layers..."):
        rgb, bounds = load_stretched_rgb(region_key)
        exposure = load_band(region_key, "exposure_map")
        flood_prone = load_band(region_key, "flood_prone_mask")

        overlay = rgb.copy()
        fp_only = (flood_prone == 1) & (exposure == 0)
        overlay[fp_only] = overlay[fp_only] * 0.5 + np.array([0.2, 0.4, 1.0]) * 0.5
        overlay[exposure == 1] = overlay[exposure == 1] * 0.4 + np.array([1.0, 0.1, 0.1]) * 0.6

        img = Image.fromarray((overlay * 255).astype(np.uint8))
        west, south, east, north = bounds
        center = [(south + north) / 2, (west + east) / 2]

        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
        folium.raster_layers.ImageOverlay(
            image=np.array(img), bounds=[[south, west], [north, east]],
            opacity=0.85, name="Flood Exposure Overlay",
        ).add_to(m)
        folium.LayerControl().add_to(m)

    st.markdown("🔵 Blue = flood-prone only &nbsp;&nbsp; 🔴 Red = flood-prone + built-up (exposed)  \n"
                "*Pan and zoom the map below; toggle layers via the control in the top-right.*")
    st_folium(m, width=None, height=600, key=f"map_{region_key}")

    k = kpis[region_key]
    c1, c2, c3 = st.columns(3)
    c1.metric("% Built-up Exposed", f"{k['pct_builtup_exposed']:.1f}%")
    c2.metric("Exposed Area", f"{k['exposed_builtup_km2']:.1f} km²")
    c3.metric("Exposed Buildings", f"{k['estimated_exposed_buildings']:,}")

# ================================================================== PAGE 3
elif page == "3. Built-up Growth (GHSL)":
    st.header("📈 Built-up Growth Analysis")
    st.caption("Source: JRC Global Human Settlement Layer (GHSL) built-up surface, 2000–2025")

    epochs = kpis["region_a_kelani"]["ghsl_epochs"]
    year_range = st.select_slider("Compare years", options=epochs, value=(epochs[0], epochs[-1]))

    fig = go.Figure()
    for region, label in REGIONS.items():
        fig.add_trace(go.Scatter(
            x=kpis[region]["ghsl_epochs"], y=kpis[region]["ghsl_builtup_km2_by_epoch"],
            mode="lines+markers", name=label, line=dict(width=3),
        ))
    fig.update_layout(xaxis_title="Year", yaxis_title="Built-up surface (km²)",
                       template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                       height=450)
    st.plotly_chart(fig, use_container_width=True)
    fig_download_button(fig, "growth_trend.png", "growth_png")

    st.markdown("---")
    st.subheader("Where is growth happening? Flood-prone vs. flood-safe")
    st.caption("Yellow = new built-up growth (2000-2025) in flood-safe terrain · "
               "Red = new built-up growth in flood-prone terrain (encroachment)")
    growth_map_region = st.selectbox("Region for growth map", list(REGIONS.keys()),
                                      format_func=lambda k: REGIONS[k], key="growth_map_region_select")

    with st.spinner("Building interactive growth overlay..."):
        overlay_arr, gbounds = load_growth_overlay_array(growth_map_region)
        west, south, east, north = gbounds
        gcenter = [(south + north) / 2, (west + east) / 2]

        gmap = folium.Map(location=gcenter, zoom_start=12, tiles="OpenStreetMap")
        img = Image.fromarray((overlay_arr * 255).astype(np.uint8))
        folium.raster_layers.ImageOverlay(
            image=np.array(img), bounds=[[south, west], [north, east]],
            opacity=0.8, name="Built-up Growth 2000-2025",
        ).add_to(gmap)
        folium.LayerControl().add_to(gmap)

    st_folium(gmap, width=None, height=600, key=f"growth_folium_{growth_map_region}")

    buf = io.BytesIO()
    Image.fromarray((overlay_arr * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button("📥 Download growth overlay (PNG)", data=buf.getvalue(),
                        file_name=f"{growth_map_region}_growth_overlay.png",
                        key=f"growth_dl_{growth_map_region}")

    st.markdown("---")
    st.subheader(f"Change: {year_range[0]} → {year_range[1]}")
    cols = st.columns(2)
    for col, (region, label) in zip(cols, REGIONS.items()):
        e = kpis[region]["ghsl_epochs"]
        vals = kpis[region]["ghsl_builtup_km2_by_epoch"]
        v0 = vals[e.index(year_range[0])]
        v1 = vals[e.index(year_range[1])]
        pct_change = 100 * (v1 - v0) / v0 if v0 > 0 else 0
        with col:
            st.metric(label, f"{v1:.1f} km²", f"{pct_change:+.1f}% since {year_range[0]}")

    st.info("⚠️ Note: an earlier from-scratch classifier-based yearly trend was found to be dominated "
            "by inter-annual radiometric noise (p > 0.5, non-significant). GHSL was adopted instead, "
            "yielding statistically significant trends (p < 0.01, R² > 0.96) for both regions.")

# ================================================================== PAGE 4
elif page == "4. Regional Comparison":
    st.header("⚖️ Regional Comparison")
    labels = list(REGIONS.values())

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=labels, y=[kpis[r]["pct_builtup_exposed"] for r in REGIONS],
            marker_color=["#4C9AFF", "#FF6B6B"], text=[f"{kpis[r]['pct_builtup_exposed']:.1f}%" for r in REGIONS],
            textposition="outside",
        )])
        fig.update_layout(title="% Built-up Area Exposed to Flooding", yaxis_range=[0, 100],
                           template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, "exposure_comparison.png", "exp_cmp")

    with col2:
        fig = go.Figure(data=[go.Bar(
            x=labels, y=[kpis[r]["growth_rate_km2_per_year"] for r in REGIONS],
            marker_color=["#4C9AFF", "#FF6B6B"], text=[f"{kpis[r]['growth_rate_km2_per_year']:.2f}" for r in REGIONS],
            textposition="outside",
        )])
        fig.update_layout(title="Growth Rate (km²/year)",
                           template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)
        fig_download_button(fig, "growth_rate_comparison.png", "grw_cmp")

    st.markdown("---")
    st.subheader("Side-by-side snapshot")
    comp_df = pd.DataFrame([
        {"Metric": "Built-up area (km²)", labels[0]: kpis["region_a_kelani"]["total_builtup_km2"],
         labels[1]: kpis["region_b_dakar"]["total_builtup_km2"]},
        {"Metric": "Exposed area (km²)", labels[0]: kpis["region_a_kelani"]["exposed_builtup_km2"],
         labels[1]: kpis["region_b_dakar"]["exposed_builtup_km2"]},
        {"Metric": "% exposed", labels[0]: kpis["region_a_kelani"]["pct_builtup_exposed"],
         labels[1]: kpis["region_b_dakar"]["pct_builtup_exposed"]},
        {"Metric": "Buildings total", labels[0]: kpis["region_a_kelani"]["total_buildings"],
         labels[1]: kpis["region_b_dakar"]["total_buildings"]},
        {"Metric": "Buildings exposed", labels[0]: kpis["region_a_kelani"]["estimated_exposed_buildings"],
         labels[1]: kpis["region_b_dakar"]["estimated_exposed_buildings"]},
        {"Metric": "Population exposed (est.)", labels[0]: kpis["region_a_kelani"]["population_exposed"],
         labels[1]: kpis["region_b_dakar"]["population_exposed"]},
        {"Metric": "Growth rate (km²/yr)", labels[0]: kpis["region_a_kelani"]["growth_rate_km2_per_year"],
         labels[1]: kpis["region_b_dakar"]["growth_rate_km2_per_year"]},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    df_download_button(comp_df, "regional_comparison.csv", "comp_csv")

# ================================================================== PAGE 5
elif page == "5. Risk Assessment":
    st.header("⚠️ Risk Assessment & Statistics")
    st.caption("Risk tiers based on HAND (Height Above Nearest Drainage): High < 2m, Medium 2–5m, Low > 5m")

    cols = st.columns(2)
    for col, (region, label) in zip(cols, REGIONS.items()):
        k = kpis[region]
        with col:
            st.markdown(f"**{label}**")
            fig = go.Figure(data=[go.Pie(
                labels=["High Risk", "Medium Risk", "Low Risk"],
                values=[k["risk_pct_high"], k["risk_pct_medium"], k["risk_pct_low"]],
                hole=0.5, marker_colors=["#FF6B6B", "#FFB454", "#6FCF97"],
            )])
            fig.update_layout(template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                               height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            fig_download_button(fig, f"{region}_risk_pie.png", f"risk_{region}")

    st.markdown("---")
    risk_df = pd.DataFrame([
        {"Region": REGIONS[r], "High Risk %": k["risk_pct_high"], "Medium Risk %": k["risk_pct_medium"],
         "Low Risk %": k["risk_pct_low"], "Population Exposed": k["population_exposed"]}
        for r, k in kpis.items()
    ])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    df_download_button(risk_df, "risk_assessment.csv", "risk_csv")

# ================================================================== PAGE 6
elif page == "6. Transferability":
    st.header("🔄 Cross-Region Transferability")
    st.caption("Does a classifier trained in one region generalize to the other?")

    transfer_data = [
        {"Direction": "Train A → Test A (baseline)", "Balanced Accuracy": 0.882, "Macro F1": 0.881},
        {"Direction": "Train A → Test B (transfer)", "Balanced Accuracy": 0.756, "Macro F1": 0.766},
        {"Direction": "Train B → Test B (baseline)", "Balanced Accuracy": 0.906, "Macro F1": 0.901},
        {"Direction": "Train B → Test A (transfer)", "Balanced Accuracy": 0.786, "Macro F1": 0.774},
    ]
    transfer_df = pd.DataFrame(transfer_data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("A→B transfer degradation", "-12.6 pts", "Balanced accuracy", delta_color="inverse")
        st.metric("A→B same-region baseline", "88.2%")
        st.metric("A→B cross-region transfer", "75.6%")
    with col2:
        st.metric("B→A transfer degradation", "-12.0 pts", "Balanced accuracy", delta_color="inverse")
        st.metric("B→A same-region baseline", "90.6%")
        st.metric("B→A cross-region transfer", "78.6%")

    st.markdown("---")
    fig = go.Figure()
    x = np.arange(len(transfer_df))
    fig.add_trace(go.Bar(x=transfer_df["Direction"], y=transfer_df["Balanced Accuracy"],
                          name="Balanced Accuracy", marker_color="#4C9AFF"))
    fig.add_trace(go.Bar(x=transfer_df["Direction"], y=transfer_df["Macro F1"],
                          name="Macro F1", marker_color="#FF6B6B"))
    fig.update_layout(barmode="group", yaxis_range=[0, 1],
                       template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                       height=450, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)
    fig_download_button(fig, "transferability_comparison.png", "transfer_png")
    df_download_button(transfer_df, "transferability_metrics.csv", "transfer_csv")

    st.markdown("---")
    st.subheader("Why does transfer degrade? (SHAP explainability)")
    st.info(
        "SHAP analysis (via a depth-limited surrogate model) shows the two region-specific "
        "models rely on **different evidence**: the Region-A-trained model leans primarily on "
        "**NDBI** (mean |SHAP| = 0.112), consistent with built-up index theory. The Region-B-trained "
        "model instead leans primarily on **raw NIR reflectance and SAR backscatter**, with NDBI "
        "importance dropping to secondary rank. This suggests each model partially learned "
        "region-specific spectral signatures rather than one universal built-up concept — a plausible "
        "mechanistic explanation for the ~12-point transfer gap observed in both directions."
    )

    doc_path = Path("docs/phase5_transferability_synthesis.md")
    if doc_path.exists():
        with st.expander("📄 Read full synthesis writeup"):
            st.markdown(doc_path.read_text())


elif page == "7. Data Quality & Model Performance":
    st.header("🔬 Data Quality & Model Performance")

    st.subheader("Built-up Classifier Performance")
    perf_df = pd.DataFrame([
        {"Direction": "Train A → Test A (baseline)", "Balanced Accuracy": 0.882, "Macro F1": 0.881},
        {"Direction": "Train A → Test B (transfer)", "Balanced Accuracy": 0.756, "Macro F1": 0.766},
        {"Direction": "Train B → Test B (baseline)", "Balanced Accuracy": 0.906, "Macro F1": 0.901},
        {"Direction": "Train B → Test A (transfer)", "Balanced Accuracy": 0.786, "Macro F1": 0.774},
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=perf_df["Direction"], y=perf_df["Balanced Accuracy"], name="Balanced Accuracy", marker_color="#4C9AFF"))
    fig.add_trace(go.Bar(x=perf_df["Direction"], y=perf_df["Macro F1"], name="Macro F1", marker_color="#FF6B6B"))
    fig.update_layout(barmode="group", yaxis_range=[0, 1],
                       template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)
    fig_download_button(fig, "model_performance.png", "perf_png")
    df_download_button(perf_df, "model_performance.csv", "perf_csv")

    st.info("Cross-region transfer shows a consistent ~12-point balanced-accuracy degradation in both "
            "directions. SHAP analysis attributes this to region-specific reliance on different spectral "
            "evidence (NDBI-driven in Region A vs. visible-band/SAR-driven in Region B).")

    st.markdown("---")
    st.subheader("Known Data Quality Issues (resolved during development)")
    issues_df = pd.DataFrame([
        {"Issue": "HAND boundary drainage artifact", "Impact": "Up to 27% NaN pixels",
         "Resolution": "Re-computed on buffered DEM (+5.5km margin); reduced to <5% NaN"},
        {"Issue": "WorldPop resampling area-scaling bug", "Impact": "~100x population overcount",
         "Resolution": "Corrected via area-ratio division; verified against native totals"},
        {"Issue": "From-scratch yearly classifier radiometric noise", "Impact": "Non-significant trend (p>0.5)",
         "Resolution": "Switched to JRC GHSL purpose-built time series (p<0.01)"},
        {"Issue": "Open Buildings Shapefile export conflict", "Impact": "Export failure",
         "Resolution": "Stripped conflicting Geometry-typed property before export"},
    ])
    st.dataframe(issues_df, use_container_width=True, hide_index=True)

# ================================================================== PAGE 7
elif page == "8. Methodology & Sources":
    st.header("📚 Methodology & Data Sources")
    st.markdown("""
    ### Pipeline Overview
    1. **Data Acquisition** — Sentinel-2/1, Copernicus DEM, ESA WorldCover, WorldPop, Google Open
       Buildings, and JRC GHSL, all sourced via Google Earth Engine for two study regions.
    2. **Built-up Classification** — Random Forest on spectral bands, NDVI/NDBI, SAR backscatter,
       and elevation; validated via spatially-blocked checkerboard train/test split.
    3. **Flood-Proneness (HAND)** — D8 flow routing on a buffered DEM; thresholded at 5m.
    4. **Exposure Overlay** — Intersection of built-up classification and flood-prone terrain.
    5. **Explainability** — SHAP via a depth-limited surrogate model for tractable exact computation.
    6. **Growth Trends** — JRC GHSL built-up surface time series (2000–2025).
    """)

    st.subheader("Data Sources")
    sources_df = pd.DataFrame([
        {"Dataset": "Sentinel-2 SR Harmonized", "Source": "ESA/Copernicus (GEE)", "Resolution": "10m", "Use": "Optical bands, NDVI/NDBI"},
        {"Dataset": "Sentinel-1 GRD", "Source": "ESA/Copernicus (GEE)", "Resolution": "10m", "Use": "SAR backscatter"},
        {"Dataset": "Copernicus DEM GLO-30", "Source": "ESA (GEE)", "Resolution": "30m", "Use": "Elevation, HAND"},
        {"Dataset": "ESA WorldCover v200", "Source": "ESA (GEE)", "Resolution": "10m", "Use": "Land-cover labels"},
        {"Dataset": "WorldPop", "Source": "WorldPop (GEE)", "Resolution": "100m", "Use": "Population density"},
        {"Dataset": "Google Open Buildings v3", "Source": "Google Research (GEE)", "Resolution": "vector", "Use": "Building footprints"},
        {"Dataset": "JRC GHSL Built-up Surface", "Source": "JRC (GEE)", "Resolution": "100m", "Use": "Growth trend"},
    ])
    st.dataframe(sources_df, use_container_width=True, hide_index=True)

    st.subheader("Known Limitations")
    st.markdown("""
    - Cross-region transferability shows a ~12-point balanced-accuracy gap in both directions.
    - Administrative-boundary-level exposure breakdown is not yet implemented (requires GADM
      boundary data, not currently in the pipeline).
    - Population exposure figures are estimates derived from WorldPop's gridded density product,
      not census-verified counts.
    """)

    st.markdown("---")
    st.subheader("📄 Export Summary Report")
    pdf_bytes = generate_summary_pdf(kpis)
    st.download_button("📥 Download Summary Report (PDF)", data=pdf_bytes,
                        file_name="fyp_summary_report.pdf", mime="application/pdf")

# ---------------------------------------------------------------- FOOTER
st.markdown(f"""
<div class="kdu-footer">
    <strong>{PROJECT_TITLE}</strong><br>
    {INSTITUTION}<br>
    Author: {AUTHOR} &nbsp;|&nbsp; Supervisor: {SUPERVISOR}<br>
    Data sources: Google Earth Engine (Sentinel-1/2, Copernicus DEM, ESA WorldCover, WorldPop, Google Open Buildings, JRC GHSL)
</div>
""", unsafe_allow_html=True)
