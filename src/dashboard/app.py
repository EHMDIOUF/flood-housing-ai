"""
Streamlit dashboard v2: flood-exposed housing detection system.
Professional styling, KPI cards, region-comparison charts, and
download buttons for charts (PNG) and summary data (CSV).
"""
import streamlit as st
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
from pathlib import Path

st.set_page_config(page_title="Flood-Exposed Housing Detection", page_icon="🌊", layout="wide")

REGIONS = {
    "region_a_kelani": "Lower Kelani Basin (Sri Lanka)",
    "region_b_dakar": "Dakar Northern Suburbs (Senegal)",
}

# --- Custom styling ---
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background-color: #1e2530;
        border: 1px solid #3a4556;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] label { color: #a8b3c4 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricDelta"] { color: #6fcf97 !important; }
    h1, h2, h3 { color: #ffffff; }
    .stRadio > label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


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

    def stretch(band, low=2, high=98):
        lo, hi = np.percentile(band, [low, high])
        return np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

    return np.dstack([stretch(red), stretch(green), stretch(blue)])


@st.cache_data
def load_band(region, product_suffix):
    path = Path("data/processed") / f"{region}_{product_suffix}.tif"
    with rasterio.open(path) as src:
        return src.read(1)


def fig_download_button(fig, filename, label="📥 Download chart (PNG)"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")


def df_download_button(df, filename, label="📥 Download data (CSV)"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


kpis = load_kpis()

# --- Sidebar navigation ---
st.sidebar.title("🌊 FYP Dashboard")
st.sidebar.caption("Flood-Exposed Housing Detection\nKotelawala Defence University")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard Summary", "🗺️ Region Explorer", "📈 Growth & Comparison", "ℹ️ About / Data Sources"],
)

# ==========================================================================
if page == "📊 Dashboard Summary":
    st.title("🌊 Flood-Exposed Housing Detection — Summary")
    st.caption("Cross-region built-up mapping, flood-proneness, and exposure analysis")

    total_builtup = sum(k["total_builtup_km2"] for k in kpis.values())
    total_exposed = sum(k["exposed_builtup_km2"] for k in kpis.values())
    total_buildings = sum(k["total_buildings"] for k in kpis.values())
    total_exposed_buildings = sum(k["estimated_exposed_buildings"] for k in kpis.values())

    st.subheader("Combined (both regions)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total built-up area", f"{total_builtup:.1f} km²")
    c2.metric("Flood-exposed built-up area", f"{total_exposed:.1f} km²",
              f"{100*total_exposed/total_builtup:.1f}% of built-up")
    c3.metric("Total buildings mapped", f"{total_buildings:,}")
    c4.metric("Estimated exposed buildings", f"{total_exposed_buildings:,}",
              f"{100*total_exposed_buildings/total_buildings:.1f}% of buildings")

    st.markdown("---")
    st.subheader("Per-region breakdown")

    cols = st.columns(2)
    for col, (region, label) in zip(cols, REGIONS.items()):
        k = kpis[region]
        with col:
            st.markdown(f"#### {label}")
            m1, m2 = st.columns(2)
            m1.metric("Built-up area", f"{k['total_builtup_km2']:.1f} km²")
            m2.metric("Exposed area", f"{k['exposed_builtup_km2']:.1f} km²")
            m3, m4 = st.columns(2)
            m3.metric("% built-up exposed", f"{k['pct_builtup_exposed']:.1f}%")
            m4.metric("Growth rate", f"{k['growth_rate_km2_per_year']:.2f} km²/yr")
            st.metric("Estimated exposed buildings", f"{k['estimated_exposed_buildings']:,}",
                      f"{k['pct_buildings_exposed']:.1f}% of {k['total_buildings']:,} total")

    # Summary CSV download
    summary_df = pd.DataFrame([
        {
            "region": REGIONS[r],
            "total_builtup_km2": k["total_builtup_km2"],
            "exposed_builtup_km2": k["exposed_builtup_km2"],
            "pct_builtup_exposed": k["pct_builtup_exposed"],
            "growth_rate_km2_per_year": k["growth_rate_km2_per_year"],
            "growth_r_squared": k["growth_r_squared"],
            "total_buildings": k["total_buildings"],
            "estimated_exposed_buildings": k["estimated_exposed_buildings"],
            "pct_buildings_exposed": k["pct_buildings_exposed"],
        }
        for r, k in kpis.items()
    ])
    st.markdown("---")
    df_download_button(summary_df, "dashboard_summary.csv", "📥 Download full summary (CSV)")
    st.dataframe(summary_df, use_container_width=True)

# ==========================================================================
elif page == "🗺️ Region Explorer":
    st.title("🗺️ Region Explorer")
    region_key = st.selectbox("Select region", list(REGIONS.keys()), format_func=lambda k: REGIONS[k])
    k = kpis[region_key]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Built-up area", f"{k['total_builtup_km2']:.1f} km²")
    c2.metric("Exposed area", f"{k['exposed_builtup_km2']:.1f} km²")
    c3.metric("% exposed", f"{k['pct_builtup_exposed']:.1f}%")
    c4.metric("Exposed buildings", f"{k['estimated_exposed_buildings']:,}")

    tab1, tab2 = st.tabs(["True Color", "Flood Exposure Overlay"])

    with tab1:
        rgb = load_stretched_rgb(region_key)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(f"{REGIONS[region_key]} — Sentinel-2 true color")
        st.pyplot(fig)
        fig_download_button(fig, f"{region_key}_truecolor.png")

    with tab2:
        rgb = load_stretched_rgb(region_key)
        exposure = load_band(region_key, "exposure_map")
        flood_prone = load_band(region_key, "flood_prone_mask")
        overlay = rgb.copy()
        fp_only = (flood_prone == 1) & (exposure == 0)
        overlay[fp_only] = overlay[fp_only] * 0.5 + np.array([0.2, 0.4, 1.0]) * 0.5
        overlay[exposure == 1] = overlay[exposure == 1] * 0.4 + np.array([1.0, 0.1, 0.1]) * 0.6

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title("Blue = flood-prone only | Red = flood-prone + built-up (exposed)")
        st.pyplot(fig)
        fig_download_button(fig, f"{region_key}_exposure_overlay.png")

# ==========================================================================
elif page == "📈 Growth & Comparison":
    st.title("📈 Growth Trend & Region Comparison")

    st.subheader("Built-up growth over time (GHSL)")
    fig, ax = plt.subplots(figsize=(9, 5))
    for region, label in REGIONS.items():
        k = kpis[region]
        ax.plot(k["ghsl_epochs"], k["ghsl_builtup_km2_by_epoch"], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel("Built-up surface (km²)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    fig_download_button(fig, "growth_comparison.png")

    growth_df = pd.DataFrame({
        "year": kpis["region_a_kelani"]["ghsl_epochs"],
        REGIONS["region_a_kelani"]: kpis["region_a_kelani"]["ghsl_builtup_km2_by_epoch"],
        REGIONS["region_b_dakar"]: kpis["region_b_dakar"]["ghsl_builtup_km2_by_epoch"],
    })
    df_download_button(growth_df, "growth_by_year.csv")

    st.markdown("---")
    st.subheader("Region comparison: exposure & growth rate")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = list(REGIONS.values())
        values = [kpis[r]["pct_builtup_exposed"] for r in REGIONS]
        bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("% of built-up area exposed to flooding")
        ax.set_ylim(0, 100)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
        st.pyplot(fig)
        fig_download_button(fig, "exposure_comparison.png")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        values = [kpis[r]["growth_rate_km2_per_year"] for r in REGIONS]
        bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Growth rate (km²/year)")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
        st.pyplot(fig)
        fig_download_button(fig, "growth_rate_comparison.png")

# ==========================================================================
elif page == "ℹ️ About / Data Sources":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Methodology
    This system detects housing exposed to flood risk across two study
    regions using a multi-stage geospatial ML pipeline:

    1. **Built-up classification**: Random Forest trained on Sentinel-2
       optical bands, Sentinel-1 SAR backscatter, NDVI/NDBI spectral
       indices, and elevation, validated via a spatially-blocked
       checkerboard train/test split.
    2. **Flood-proneness**: HAND (Height Above Nearest Drainage) index
       computed via D8 flow routing on a buffered Copernicus GLO-30 DEM,
       thresholded at 5m per literature convention.
    3. **Exposure overlay**: intersection of built-up classification and
       flood-prone terrain.
    4. **Explainability**: SHAP analysis via a depth-limited surrogate
       model (production RF trees were too large for tractable exact
       SHAP computation).
    5. **Growth trends**: JRC Global Human Settlement Layer (GHSL)
       built-up surface time series (2000-2025), used in place of a
       from-scratch multi-temporal classifier after the latter was found
       to be dominated by inter-annual radiometric noise.

    ### Data Sources
    | Dataset | Source | Resolution | Use |
    |---|---|---|---|
    | Sentinel-2 SR Harmonized | ESA/Copernicus via Earth Engine | 10m | Optical bands, NDVI/NDBI |
    | Sentinel-1 GRD | ESA/Copernicus via Earth Engine | 10m | SAR backscatter (VV/VH) |
    | Copernicus DEM GLO-30 | ESA via Earth Engine | 30m | Elevation, HAND computation |
    | ESA WorldCover v200 (2021) | ESA via Earth Engine | 10m | Land-cover base labels |
    | WorldPop | WorldPop via Earth Engine | 100m | Population density |
    | Google Open Buildings v3 | Google Research via Earth Engine | vector | Building footprints, labels |
    | JRC GHSL Built-up Surface (P2023A) | Joint Research Centre via Earth Engine | 100m | Multi-temporal growth trend |

    ### Study Regions
    - **Region A**: Lower Kelani River basin, Colombo/Gampaha corridor, Sri Lanka
    - **Region B**: Northern Dakar suburbs (Pikine, Guédiawaye, Keur Massar), Senegal

    ### Known Limitations
    - Cross-region transferability shows a consistent ~12-point balanced-accuracy
      gap in both directions (see Transferability Synthesis for details).
    - GHSL is used for long-term (2000-2025) trend estimation rather than a
      classifier-based approach, due to radiometric instability in from-scratch
      annual composites.
    - HAND computation required a buffered DEM re-export to resolve boundary
      drainage artifacts affecting up to 27% of pixels in the original crop.
    """)

    doc_path = Path("docs/phase5_transferability_synthesis.md")
    if doc_path.exists():
        with st.expander("📄 Full Cross-Region Transferability Synthesis"):
            st.markdown(doc_path.read_text())

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit | Data via Google Earth Engine")
