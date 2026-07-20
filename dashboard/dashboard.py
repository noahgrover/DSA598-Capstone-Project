# =========================================================================================================================================
# Streamlit Archival Knowledge Graph Dashboard
# =========================================================================================================================================

import json
from pathlib import Path
from collections import Counter
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

@st.cache_data
def load_and_parse_jsonld(filename="enriched.jsonld"):
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
    
    # Dynamically track how many source records belong to each cohort
    records_per_cohort = Counter()
    total_nil_mentions = 0
    total_wikidata_mentions = 0

    for r in records:
        cohort = r.get("cohort", "Unknown Cohort")
        records_per_cohort[cohort] += 1  # Increment record count for this cohort
        
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
    return df_entities, records_per_cohort

df, records_per_cohort = load_and_parse_jsonld()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("📊 Filter Controls")
    selected_cohorts = st.sidebar.multiselect(
        "Select Historical Cohorts",
        options=list(df["Cohort"].unique()),
        default=list(df["Cohort"].unique())
    )

    df_filtered = df[df["Cohort"].isin(selected_cohorts)]

    # Empty State Guard
    if df_filtered.empty:
        st.warning("⚠️ Please select at least one cohort in the sidebar to display data.")
    else:
        # Calculate total source records represented by the user's selected cohorts
        filtered_records_count = sum(records_per_cohort[cohort] for cohort in selected_cohorts)

        # Metrics Row wrapped in a container (Now with 6 columns)
        with st.container(border=True):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Total Records", filtered_records_count)
            with m2:
                st.metric("Entity Mentions", len(df_filtered))
            with m3:
                st.metric("Unique Entity Nodes", df_filtered["Entity ID"].nunique())
            with m4:
                wd_links = len(df_filtered[df_filtered["Resolution Type"] == "Wikidata Resolved"])
                st.metric("Wikidata Links", wd_links)
            
            # Calculate total populated demographic fields
            relational_columns = ["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country"]
            populated_count = df_filtered[relational_columns].notna().sum().sum()

            with m5:
                # Metric 1: Demographics per Mention
                avg_demo = populated_count / len(df_filtered) if len(df_filtered) > 0 else 0
                st.metric("Demographics / Mention", f"{avg_demo:.2f}x")
                
            with m6:
                # Metric 2: Paths per Record
                avg_paths = populated_count / filtered_records_count if filtered_records_count > 0 else 0
                st.metric("Paths / Record", f"{avg_paths:.2f}x")

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
