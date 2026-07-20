# =========================================================================================================================================
# Streamlit Archival Knowledge Graph Dashboard (OPTIMIZED & DECOUPLED)
# =========================================================================================================================================

import json
import re
from pathlib import Path
from collections import Counter
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configurations
st.set_page_config(
    page_title="Archival Knowledge Graph Analytics",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🕸️ Archival Entity Linking & Semantic Graph Dashboard")
st.markdown("""
This dashboard visualizes the structural and qualitative improvements introduced by our advanced NER,
Linking, NIL Clustering, and W3C Semantic Enrichment pipeline. Live API calls have been moved upstream 
for instant rendering.
""")

def safe_int(value):
    """Safely converts mixed data types to integers for the timeline."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None

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
    records_per_cohort = Counter()

    for r in records:
        cohort = r.get("cohort", "Unknown Cohort")
        records_per_cohort[cohort] += 1  
        
        for ent in r.get("entities", []):
            ent_id = ent.get("@id", "")

            # Robust Resolution Typing
            if ent_id.startswith("wd:") or "wikidata.org" in ent_id:
                resolution_type = "Wikidata Resolved"
            elif ent_id.startswith("local:entity/") or "b0" in ent_id:
                resolution_type = "NIL Clustered"
            else:
                resolution_type = "Unlinked Entity"

            raw_type = ent.get("@type", "schema:Thing")
            resolved_type = raw_type[-1] if isinstance(raw_type, list) else raw_type

            geo_data = ent.get("geo", {})
            lat = geo_data.get("latitude") if isinstance(geo_data, dict) else None
            lon = geo_data.get("longitude") if isinstance(geo_data, dict) else None

            # Flattening strictly to schema.org standardized fields injected upstream
            flat_entities.append({
                "Entity ID": ent_id,
                "Surface Text": ent.get("entity_span", ""),
                "Official Name": ent.get("officialName", ent.get("entity_span", "")),
                "NER Class": str(resolved_type).replace("schema:", "").replace("local:", ""),
                "Confidence": float(ent.get("ner_confidence", 1.0)),
                "Mapping Confidence": float(ent.get("mapping_confidence_score", 0.0)),
                "LLM Reasoning": ent.get("llm_reasoning", "No reasoning provided."),
                "Mentions Count": int(ent.get("mentions_count", 1)),
                "Source URL": ent.get("schema:url", ""),
                "Historical Significance": ent.get("historical_significance", ""),
                "Resolution Type": resolution_type,
                "Cohort": cohort,
                "Location Type": ent.get("schema:locationType", "Unknown"),
                "Latitude": lat,
                "Longitude": lon,
                
                # Categorical fields (expecting native Python lists from the JSON)
                "Occupation": ent.get("occupation", []),
                "Gender Identity": ent.get("genderIdentity", []),
                "Ethnic Group/Tribe": ent.get("ethnicGroup", []),
                "Religion": ent.get("religion", []),
                "Country": ent.get("country", []),
                "Affiliation": ent.get("schema:affiliation", []),
                
                # Strict Timeline Mapping
                "Target Year": safe_int(ent.get("schema:startDate")),
                "End Year": safe_int(ent.get("schema:endDate"))
            })

    df_entities = pd.DataFrame(flat_entities)
    return df_entities, records_per_cohort

df, records_per_cohort = load_and_parse_jsonld()

if df is not None:
    st.sidebar.header("📊 Filter Controls")
    selected_cohorts = st.sidebar.multiselect(
        "Select Historical Cohorts",
        options=list(df["Cohort"].unique()),
        default=list(df["Cohort"].unique())
    )

    df_filtered = df[df["Cohort"].isin(selected_cohorts)]

    if df_filtered.empty:
        st.warning("⚠️ Please select at least one cohort in the sidebar to display data.")
    else:
        filtered_records_count = sum(records_per_cohort[cohort] for cohort in selected_cohorts)

        # Metrics Row
        with st.container(border=True):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1: st.metric("Total Records", filtered_records_count)
            with m2: st.metric("Entity Mentions", df_filtered["Mentions Count"].sum())
            with m3: st.metric("Unique Nodes", df_filtered["Entity ID"].nunique())
            with m4:
                wd_links = len(df_filtered[df_filtered["Resolution Type"] == "Wikidata Resolved"])
                st.metric("Wikidata Links", wd_links)
            with m5:
                # Count non-empty lists
                populated_count = df_filtered[["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Affiliation"]].apply(lambda col: col.map(bool)).sum().sum()
                avg_demo = populated_count / len(df_filtered) if len(df_filtered) > 0 else 0
                st.metric("Demographics / Node", f"{avg_demo:.2f}x")
            with m6:
                avg_conf = df_filtered["Mapping Confidence"].mean() * 100
                st.metric("Avg Map Confidence", f"{avg_conf:.1f}%")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Archival Geospatial Map",
            "📊 Demographic & Crossover Insights",
            "⏳ Archival Calendar Timeline",
            "🔍 Interactive Entity Explorer",
            "📈 Pipeline Quality Diagnostics"
        ])
        
        # --- TAB 1: GIS Map ---
        with tab1:
            st.subheader("Geospatial Entity Distribution")
            df_geo = df_filtered[df_filtered["Latitude"].notna() & df_filtered["Longitude"].notna()]
            
            if not df_geo.empty:
                loc_types = ["All"] + list(df_geo["Location Type"].unique())
                selected_loc_type = st.selectbox("Filter by Location Type:", options=loc_types)
                
                if selected_loc_type != "All":
                    df_geo = df_geo[df_geo["Location Type"] == selected_loc_type]
                    
                fig_map = px.scatter_mapbox(
                    df_geo,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Official Name",
                    hover_data=["Surface Text", "Cohort", "Location Type"],
                    color="Location Type",
                    size="Mentions Count", # Dynamic node sizing
                    zoom=2,
                    height=600
                )
                fig_map.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No geospatial data coordinates found in filtered dataset.")

        # --- TAB 2: Enhanced Demographic Analysis ---
        with tab2:
            st.subheader("Archival Intersectionality & Demographic Distributions")
            
            # --- Row 1: Dynamic Profiles ---
            st.markdown("### 📊 Dynamic Demographic Profiler")
            demo_options = ["Occupation", "Ethnic Group/Tribe", "Gender Identity", "Religion", "Country", "Affiliation"]
            selected_demo = st.selectbox("Select Target Attribute Profile:", options=demo_options, index=0)
            
            # Filter out empty lists, then natively explode them
            df_demo = df_filtered[df_filtered[selected_demo].astype(bool)][[selected_demo, "Cohort"]]
            
            if not df_demo.empty:
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
            with cx_col1: attr_x = st.selectbox("Select X-Axis Intersection Attribute:", options=demo_options, index=0)
            with cx_col2: attr_y = st.selectbox("Select Y-Axis Intersection Attribute:", options=demo_options, index=5) # Default to Affiliation
                
            if attr_x == attr_y:
                st.error("⚠️ Cross-analysis requires selecting two distinct demographic vectors.")
            else:
                df_cross = df_filtered[df_filtered[attr_x].astype(bool) & df_filtered[attr_y].astype(bool)][[attr_x, attr_y]]
                if not df_cross.empty:
                    # Extract primary value from the list for clear matrix visualization
                    df_cross[attr_x] = df_cross[attr_x].apply(lambda x: x[0])
                    df_cross[attr_y] = df_cross[attr_y].apply(lambda x: x[0])
                    
                    top_x = df_cross[attr_x].value_counts().head(8).index
                    top_y = df_cross[attr_y].value_counts().head(8).index
                    df_cross_f = df_cross[df_cross[attr_x].isin(top_x) & df_cross[attr_y].isin(top_y)]
                    
                    if not df_cross_f.empty:
                        cross_matrix = df_cross_f.groupby([attr_x, attr_y]).size().reset_index(name="Co-occurrences")
                        fig_heatmap = px.density_heatmap(
                            cross_matrix, x=attr_x, y=attr_y, z="Co-occurrences",
                            text_auto=True, color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("No explicit intersections found for the top elements.")
                else:
                    st.info("No co-occurring data coordinates available.")
        
        # --- TAB 3: Timeline ---
        with tab3:
            st.subheader("⏳ Calendar-Year Narrative Timeline")
            
            df_time = df_filtered[df_filtered["NER Class"].str.contains("Person|Event", case=False, na=False)].copy()
            df_time = df_time[df_time["Target Year"].notna()]

            if df_time.empty:
                st.info("No elements with valid timelines or local date stamps match your active filters.")
            else:
                df_time = df_time.sort_values(by="Target Year")
                fig_timeline = go.Figure()

                for idx, row in df_time.iterrows():
                    name = row["Official Name"]
                    start = int(row["Target Year"])
                    ent_type = "Person" if "Person" in row["NER Class"] else "Event"
                    color = "#3498DB" if ent_type == "Person" else "#E67E22"
                    
                    # Enhanced Hover Information
                    hover = f"<b>{name}</b><br>Type: {ent_type}<br>Sig: {row['Historical Significance']}"

                    if ent_type == "Person" and pd.notna(row["End Year"]):
                        end = int(row["End Year"])
                        hover += f"<br>Lifespan: {start} – {end}"
                        fig_timeline.add_trace(go.Scatter(
                            x=[start, end], y=[name, name],
                            mode="lines+markers",
                            line=dict(color=color, width=4),
                            marker=dict(size=10, symbol="circle", color=[color, color]),
                            hovertext=hover, hoverinfo="text", showlegend=False
                        ))
                    else:
                        hover += f"<br>Year: {start}"
                        fig_timeline.add_trace(go.Scatter(
                            x=[start], y=[name, name],
                            mode="markers",
                            marker=dict(size=14, symbol="diamond", color=color),
                            hovertext=hover, hoverinfo="text", showlegend=False
                        ))

                fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color="#3498DB"), name="Person Lifespan"))
                fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, symbol="diamond", color="#E67E22"), name="Event Milestone"))

                fig_timeline.update_layout(
                    xaxis_title="Linear Calendar Axis (Years)",
                    yaxis=dict(autorange="reversed", title="", tickmode='linear'),
                    height=200 + (len(df_time) * 32),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    margin=dict(l=220)
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        # --- TAB 4: Search and Explore Directory ---
        with tab4:
            st.subheader("Knowledge Graph Node Directory")
            search_query = st.text_input("🔍 Search nodes...", "")
            df_display = df_filtered.copy()
            if search_query:
                df_display = df_display[df_display["Surface Text"].str.contains(search_query, case=False, na=False)]
            
            # Format URLs to be clickable in the dataframe
            st.dataframe(
                df_display[["Entity ID", "Surface Text", "Official Name", "NER Class", "Mentions Count", "Source URL", "Resolution Type"]], 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Source URL": st.column_config.LinkColumn("Source URL")
                }
            )

        # --- TAB 5: Pipeline Quality Diagnostics ---
        with tab5:
            st.subheader("Pipeline Quality & Resolution Diagnostics")
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("#### Entity Resolution Types")
                res_counts = df_filtered["Resolution Type"].value_counts().reset_index()
                fig_res = px.pie(res_counts, values="count", names="Resolution Type", color_discrete_map={"Wikidata Resolved": "#2ECC71", "NIL Clustered": "#3498DB", "Unlinked Entity": "#E74C3C"})
                st.plotly_chart(fig_res, use_container_width=True)
                
            with d_col2:
                st.markdown("#### Knowledge Graph Mapping Confidence")
                fig_conf = px.histogram(df_filtered, x="Mapping Confidence", nbins=20, color_discrete_sequence=["#9B59B6"])
                fig_conf.update_layout(yaxis_title="Entity Count", xaxis_title="Confidence Score (0.0 to 1.0)")
                st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown("---")
            
            st.markdown("#### LLM Reasoning Audit (NIL & Ambiguous Entities)")
            st.markdown("Audit the logical steps generated by the LLM for entity clustering and disambiguation.")
            df_reasoning = df_filtered[df_filtered["Resolution Type"] == "NIL Clustered"][["Official Name", "Mapping Confidence", "LLM Reasoning"]]
            st.dataframe(df_reasoning, use_container_width=True, hide_index=True)
