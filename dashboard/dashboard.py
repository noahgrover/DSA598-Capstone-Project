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

# --- Tab 2: Enhanced Demographic Analysis ---
        with tab2:
            st.subheader("Archival Intersectionality & Demographic Distributions")
            
            # --- Row 1: Dynamic Profiles ---
            st.markdown("### 📊 Dynamic Demographic Profiler")
            demo_options = ["Occupation", "Ethnic Group/Tribe", "Gender Identity", "Religion", "Country"]
            selected_demo = st.selectbox("Select Target Attribute Profile:", options=demo_options, index=0)
            
            df_demo = df_filtered[[selected_demo, "Cohort"]].dropna()
            
            if not df_demo.empty:
                # Properly split and expand multi-valued lists (e.g., "Author, Politician") for exact tallies
                df_demo[selected_demo] = df_demo[selected_demo].str.split(", ")
                df_demo = df_demo.explode(selected_demo)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### Top 10 {selected_demo}s (Overall)")
                    top_10 = df_demo[selected_demo].value_counts().reset_index().head(10)
                    fig_demo = px.bar(top_10, x="count", y=selected_demo, orientation='h', color_discrete_sequence=["#2ECC71"])
                    fig_demo.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_demo, use_container_width=True)
                    
                with col2:
                    st.markdown(f"#### {selected_demo} Distribution by Historical Cohort")
                    cohort_counts = df_demo.groupby(["Cohort", selected_demo]).size().reset_index(name="count")
                    # Filter chart categories to top 10 values to preserve clean readability
                    cohort_counts = cohort_counts[cohort_counts[selected_demo].isin(top_10[selected_demo])]
                    fig_cohort = px.bar(cohort_counts, x="count", y=selected_demo, color="Cohort", orientation='h', barmode="stack")
                    fig_cohort.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_cohort, use_container_width=True)
            else:
                st.info(f"No extracted data found for '{selected_demo}' within the selected filters.")
                
            st.markdown("---")
            
            # --- Row 2: Matrix Intersections ---
            st.markdown("### 🔀 Intersectionality Matrix")
            st.markdown("Cross-reference any two vectors below to locate structural overlaps hidden across your semantic metadata graph.")
            
            cx_col1, cx_col2 = st.columns(2)
            with cx_col1:
                attr_x = st.selectbox("Select X-Axis Intersection Attribute:", options=demo_options, index=0)
            with cx_col2:
                attr_y = st.selectbox("Select Y-Axis Intersection Attribute:", options=demo_options, index=4)
                
            if attr_x == attr_y:
                st.error("⚠️ Cross-analysis requires selecting two distinct demographic vectors.")
            else:
                df_cross = df_filtered[[attr_x, attr_y]].dropna()
                if not df_cross.empty:
                    # Isolate primary/first value to maintain clear matrix relationships
                    df_cross[attr_x] = df_cross[attr_x].apply(lambda x: x.split(", ")[0])
                    df_cross[attr_y] = df_cross[attr_y].apply(lambda x: x.split(", ")[0])
                    
                    # Restrict grid density to the top 8 elements on each axis to keep text clear
                    top_x_items = df_cross[attr_x].value_counts().head(8).index
                    top_y_items = df_cross[attr_y].value_counts().head(8).index
                    df_cross_filtered = df_cross[df_cross[attr_x].isin(top_x_items) & df_cross[attr_y].isin(top_y_items)]
                    
                    if not df_cross_filtered.empty:
                        cross_matrix = df_cross_filtered.groupby([attr_x, attr_y]).size().reset_index(name="Co-occurrences")
                        fig_heatmap = px.density_heatmap(
                            cross_matrix, 
                            x=attr_x, 
                            y=attr_y, 
                            z="Co-occurrences",
                            text_auto=True,
                            color_continuous_scale="Viridis"
                        )
                        fig_heatmap.update_layout(xaxis_title=attr_x, yaxis_title=attr_y)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("No explicit intersections found for the top elements of these attributes.")
                else:
                    st.info("No co-occurring data coordinates available for this configuration.")

        # --- Tab 3: Search and Explore Directory ---
        with tab3:
            st.subheader("Knowledge Graph Node Directory")
            search_query = st.text_input("🔍 Search nodes...", "")
            df_display = df_filtered.copy()
            if search_query:
                df_display = df_display[df_display["Surface Text"].str.contains(search_query, case=False, na=False)]
            st.dataframe(df_display[["Entity ID", "Surface Text", "Official Name", "NER Class", "Resolution Type", "Confidence"]], use_container_width=True, hide_index=True)

        # --- Tab 4: Pipeline Quality Diagnostics ---
        with tab4:
            st.subheader("Pipeline Quality & Resolution Diagnostics")
            
            # Top Row: Resolution & Confidence
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                st.markdown("#### Entity Resolution Types")
                res_counts = df_filtered["Resolution Type"].value_counts().reset_index()
                fig_res = px.pie(res_counts, values="count", names="Resolution Type", color_discrete_map={"Wikidata Resolved": "#2ECC71", "NIL Clustered": "#3498DB", "Unlinked Entity": "#E74C3C"})
                st.plotly_chart(fig_res, use_container_width=True)
                
            with d_col2:
                st.markdown("#### NER Confidence Scores")
                # Histogram to see where the bulk of confidence scores lie
                fig_conf = px.histogram(df_filtered, x="Confidence", nbins=20, color_discrete_sequence=["#3498DB"])
                fig_conf.update_layout(yaxis_title="Entity Count", xaxis_title="Confidence Score")
                st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown("---")
            
            # Bottom Row: Classes & Completeness
            d_col3, d_col4 = st.columns(2)
            
            with d_col3:
                st.markdown("#### Extracted NER Classes")
                class_counts = df_filtered["NER Class"].value_counts().reset_index()
                fig_class = px.bar(class_counts, x="count", y="NER Class", orientation='h', color_discrete_sequence=["#F1C40F"])
                st.plotly_chart(fig_class, use_container_width=True)
                
            with d_col4:
                st.markdown("#### Metadata Completeness (Fill Rate)")
                # Calculate the percentage of non-null values for key relational attributes
                attributes = ["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country"]
                completeness = [(df_filtered[col].notna().sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0 for col in attributes]
                
                df_comp = pd.DataFrame({"Attribute": attributes, "Fill Rate (%)": completeness})
                fig_comp = px.bar(df_comp, x="Fill Rate (%)", y="Attribute", orientation='h', color_discrete_sequence=["#9B59B6"])
                fig_comp.update_xaxes(range=[0, 100])
                st.plotly_chart(fig_comp, use_container_width=True)
