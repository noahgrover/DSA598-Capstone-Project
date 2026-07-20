# =========================================================================================================================================
# Streamlit Archival Knowledge Graph Dashboard
# =========================================================================================================================================

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configurations
st.set_page_config(
    page_title="Archival Knowledge Graph Analytics",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Description
st.title("🕸️ Archival Entity Linking & Semantic Graph Dashboard")
st.markdown("""
This dashboard visualizes the structural and qualitative improvements introduced by our advanced NER,
Linking, NIL Clustering, and W3C Semantic Enrichment pipeline.
""")

# Visualization Properties[cite: 1]
VISUAL_CONFIG = {
    "schema:Person": {"group": "Person", "color": "#FF5733"},
    "schema:Organization": {"group": "Organization", "color": "#2ECC71"},
    "local:NORP": {"group": "Demographic (NORP)", "color": "#9B59B6"},
    "schema:Place": {"group": "Location", "color": "#3498DB"},
    "schema:Event": {"group": "Historical Event", "color": "#F1C40F"},
    "schema:Thing": {"group": "Other", "color": "#95A5A6"}
}

@st.cache_data
def load_and_parse_jsonld(filename="enriched.jsonld"):
    # Dynamically locate the file relative to this script
    script_dir = Path(__file__).parent
    filepath = script_dir / filename

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Could not find the target file: `{filepath}`. Please ensure your enrichment script has run.")
        return None, None

    records = data.get("@graph", [])
    flat_entities = []

    total_records = len(records)
    total_nil_mentions = 0
    total_wikidata_mentions = 0

    for r in records:
        cohort = r.get("cohort", "Unknown Cohort")
        for ent in r.get("entities", []):
            ent_id = ent.get("@id", "")

            if ent_id.startswith("wd:"):
                resolution_type = "Wikidata Resolved"
                total_wikidata_mentions += 1
            elif ent_id.startswith("local:entity/"):
                resolution_type = "NIL Clustered"
                total_nil_mentions += 1
            else:
                resolution_type = "Unlinked Entity"

            raw_type = ent.get("@type", "schema:Thing")
            resolved_type = raw_type[-1] if isinstance(raw_type, list) else raw_type

            geo_data = ent.get("geo", {})
            lat = geo_data.get("latitude") if isinstance(geo_data, dict) else None
            lon = geo_data.get("longitude") if isinstance(geo_data, dict) else None

            def get_first_or_string(val):
                if isinstance(val, list):
                    return ", ".join([v.replace("wd:", "") for v in val])
                return str(val).replace("wd:", "") if val else None

            flat_entities.append({
                "Entity ID": ent_id,
                "Surface Text": ent.get("entity_span", ""),
                "Official Name": ent.get("officialName", ent.get("entity_span", "")),
                "NER Class": resolved_type.replace("schema:", "").replace("local:", ""),
                "Confidence": ent.get("ner_confidence", 1.0),
                "Resolution Type": resolution_type,
                "Cohort": cohort,
                "Visual Group": ent.get("visualGroup", "Other"),
                "Description": ent.get("description", "No description available."),
                "Latitude": lat,
                "Longitude": lon,
                "Image URL": ent.get("image"),
                "Occupation": get_first_or_string(ent.get("occupation")),
                "Gender Identity": get_first_or_string(ent.get("genderIdentity")),
                "Ethnic Group/Tribe": get_first_or_string(ent.get("ethnicGroup")),
                "Religion": get_first_or_string(ent.get("religion")),
                "Country": get_first_or_string(ent.get("country"))
            })

    df_entities = pd.DataFrame(flat_entities)
    stats = {
        "total_records": total_records,
        "total_mentions": len(df_entities),
        "total_nil": total_nil_mentions,
        "total_wd": total_wikidata_mentions,
        "unique_resolved": df_entities["Entity ID"].nunique() if len(df_entities) > 0 else 0
    }
    return df_entities, stats

df, stats = load_and_parse_jsonld()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("📊 Filter Controls")
    selected_cohorts = st.sidebar.multiselect(
        "Select Historical Cohorts",
        options=list(df["Cohort"].unique()),
        default=list(df["Cohort"].unique())
    )

    df_filtered = df[df["Cohort"].isin(selected_cohorts)]

    # ⚠️ Empty State Guard
    if df_filtered.empty:
        st.warning("⚠️ Please select at least one cohort in the sidebar to display data.")
    else:
        # Metrics Row wrapped in a container for a premium UI card look
        with st.container(border=True):
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                # Updated to dynamically show the filtered entity count
                st.metric("Total Entity Mentions", len(df_filtered))
            with m2:
                st.metric("Unique Entity Nodes", df_filtered["Entity ID"].nunique())
            with m3:
                wd_links = len(df_filtered[df_filtered["Resolution Type"] == "Wikidata Resolved"])
                st.metric("Wikidata Links", wd_links)
            with m4:
                nil_links = len(df_filtered[df_filtered["Resolution Type"] == "NIL Clustered"])
                st.metric("NIL Clusters", nil_links)
            with m5:
                relational_columns = ["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country"]
                populated_count = df_filtered[relational_columns].notna().sum().sum()
                # Corrected logic: calculate demographic density on the current filtered entities, not the global stat
                avg_paths = populated_count / len(df_filtered) if len(df_filtered) > 0 else 0
                st.metric("Avg Demographics / Mention", f"{avg_paths:.2f}x")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Archival Geospatial Map",
            "📊 Demographic & Crossover Insights",
            "🔍 Interactive Entity Explorer",
            "📈 Pipeline Quality Diagnostics"
        ])

        # --- GIS Map ---
        with tab1:
            st.subheader("Geospatial Entity Distribution")
            df_geo = df_filtered[df_filtered["Latitude"].notna() & df_filtered["Longitude"].notna()]
            if len(df_geo) > 0:
                fig_map = px.scatter_mapbox(
                    df_geo,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Official Name",
                    hover_data=["Surface Text", "Cohort", "Description"],
                    color="Visual Group",
                    zoom=2,
                    height=600
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geospatial data coordinates found in filtered dataset.")

        # --- Tab 2: Demographic Analysis ---
        with tab2:
            st.subheader("Archival Intersectionality & Demographic Distributions")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🏢 Top Occupations (P106)")
                df_occ = df_filtered["Occupation"].dropna().str.split(", ").explode().value_counts().reset_index()
                if not df_occ.empty:
                    fig_occ = px.bar(df_occ.head(10), x="count", y="Occupation", orientation='h', color_discrete_sequence=["#2ECC71"])
                    st.plotly_chart(fig_occ, use_container_width=True)
            with col2:
                st.markdown("#### 🪶 Ethnic & Tribal Identities (P172)")
                df_eth = df_filtered["Ethnic Group/Tribe"].dropna().str.split(", ").explode().value_counts().reset_index()
                if not df_eth.empty:
                    fig_eth = px.bar(df_eth.head(10), x="count", y="Ethnic Group/Tribe", orientation='h', color_discrete_sequence=["#9B59B6"])
                    st.plotly_chart(fig_eth, use_container_width=True)

        # --- Tab 3: Search and Explore Directory ---
        with tab3:
            st.subheader("Knowledge Graph Node Directory")
            search_query = st.text_input("🔍 Search nodes...", "")
            df_display = df_filtered.copy()
            if search_query:
                df_display = df_display[df_display["Surface Text"].str.contains(search_query, case=False, na=False)]
            st.dataframe(df_display[["Entity ID", "Surface Text", "Official Name", "NER Class", "Resolution Type", "Confidence"]], use_container_width=True, hide_index=True)

        # --- Tab 4: Resolution Type Ratios ---
        with tab4:
            st.subheader("Resolution Diagnostics")
            res_counts = df_filtered["Resolution Type"].value_counts().reset_index()
            fig_res = px.pie(res_counts, values="count", names="Resolution Type", color_discrete_map={"Wikidata Resolved": "#2ECC71", "NIL Clustered": "#3498DB", "Unlinked Entity": "#E74C3C"})
            st.plotly_chart(fig_res, use_container_width=True)
