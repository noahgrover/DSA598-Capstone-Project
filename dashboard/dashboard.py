# =========================================================================================================================================
# Streamlit Archival Knowledge Graph Dashboard
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
Linking, NIL Clustering, and W3C Semantic Enrichment pipeline.
""")

def extract_year_from_text(text):
    """Helper utility to parse a numeric year out of messy strings or ISO dates."""
    if not text:
        return None
    match = re.search(r'\b(\d{3,4})\b', str(text))
    if match:
        return int(match.group(1))
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

    def extract_labels(val):
        """Extracts human-readable names from rich JSON-LD objects."""
        if isinstance(val, list):
            return [v.get("schema:name", v.get("@id", "").replace("wd:", "")) if isinstance(v, dict) else str(v) for v in val]
        elif isinstance(val, dict):
            return [val.get("schema:name", val.get("@id", "").replace("wd:", ""))]
        elif val:
            return [str(val).replace("wd:", "")]
        return []

    # Map your string icon labels into rendering emojis for Streamlit/Plotly
    icon_mapping = {
        "person": "👤",
        "group": "🏛️",
        "demographics": "👥",
        "place": "📍",
        "event": "📜",
        "help": "❓"
    }

    for r in records:
        cohort = r.get("cohort", "Unknown Cohort")
        records_per_cohort[cohort] += 1  
        
        for ent in r.get("entities", []):
            ent_id = ent.get("@id", "")

            if ent_id.startswith("wd:"):
                resolution_type = "Wikidata Resolved"
            elif ent_id.startswith("local:entity/"):
                resolution_type = "NIL Clustered"
            else:
                resolution_type = "Unlinked Entity"

            raw_type = ent.get("@type", "schema:Thing")
            resolved_type = raw_type[-1] if isinstance(raw_type, list) else raw_type

            geo_data = ent.get("geo", {})
            lat = geo_data.get("latitude") if isinstance(geo_data, dict) else None
            lon = geo_data.get("longitude") if isinstance(geo_data, dict) else None

            # Base Demographics
            occ = extract_labels(ent.get("occupation"))
            gender = extract_labels(ent.get("genderIdentity"))
            ethnic = extract_labels(ent.get("ethnicGroup"))
            religion = extract_labels(ent.get("religion"))
            country = extract_labels(ent.get("country"))
            
            # Newly Mapped Rich Properties (Convicted Of Removed)
            ideology = extract_labels(ent.get("politicalIdeology"))
            member = extract_labels(ent.get("memberOf"))
            participant = extract_labels(ent.get("participant"))
            
            # Extract and format VIAF Link
            viaf_url = None
            same_as = ent.get("schema:sameAs")
            if same_as and isinstance(same_as, str) and same_as.startswith("viaf:"):
                viaf_id = same_as.replace("viaf:", "")
                viaf_url = f"https://viaf.org/viaf/{viaf_id}/"

            # Parse Icon
            icon_str = ent.get("visualIcon", "help")
            emoji_icon = icon_mapping.get(icon_str, "❓")
            
            # Handle start/end bounds dynamically
            start_keys = ["dateOfBirth", "startDate", "schema:startDate"]
            end_keys = ["dateOfDeath", "endDate", "schema:endDate"]
            
            local_start = None
            for k in start_keys:
                if k in ent:
                    local_start = extract_year_from_text(ent[k])
                    if local_start: break
            
            local_end = None
            for k in end_keys:
                if k in ent:
                    local_end = extract_year_from_text(ent[k])
                    if local_end: break

            flat_entities.append({
                "Icon": emoji_icon,
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
                
                # Relational Lists
                "Occupation": occ,
                "Gender Identity": gender,
                "Ethnic Group/Tribe": ethnic,
                "Religion": religion,
                "Country": country,
                "Political Ideology": ideology,
                "Member Of": member,
                "Participant In": participant,
                "VIAF Link": viaf_url,
                
                "Target Year": local_start,
                "End Year": local_end
            })

    # Join list attributes into simple display strings
    list_fields = [
        "Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country", 
        "Political Ideology", "Member Of", "Participant In"
    ]
    
    for item in flat_entities:
        for field in list_fields:
            item[field] = ", ".join(item[field]) if item[field] else None

    df_entities = pd.DataFrame(flat_entities)
    return df_entities, records_per_cohort

df, records_per_cohort = load_and_parse_jsonld()

if df is not None:
    # accessible, high-contrast color map for cohorts (IBM Colorblind-Safe Palette)
    unique_cohorts = sorted(list(df["Cohort"].unique()))
    HIGH_CONTRAST_COHORT_COLORS = ["#005AB5", "#DC3220", "#FFC20A"] 
    
    COHORT_COLOR_MAP = {
        cohort: HIGH_CONTRAST_COHORT_COLORS[i % len(HIGH_CONTRAST_COHORT_COLORS)] 
        for i, cohort in enumerate(unique_cohorts)
    }

    # Sidebar UI layout logic
    st.sidebar.header("📊 Filter Controls")
    selected_cohorts = st.sidebar.multiselect(
        "Select Historical Cohorts",
        options=unique_cohorts,
        default=unique_cohorts
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
            with m2: st.metric("Entity Mentions", len(df_filtered))
            with m3: st.metric("Unique Entity Nodes", df_filtered["Entity ID"].nunique())
            with m4:
                viaf_links = df_filtered["VIAF Link"].notna().sum()
                st.metric("VIAF Authority Links", viaf_links)
            with m5:
                relational_columns = ["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country", "Political Ideology", "Member Of", "Participant In"]
                populated_count = df_filtered[relational_columns].notna().sum().sum()
                avg_demo = populated_count / len(df_filtered) if len(df_filtered) > 0 else 0
                st.metric("Metadata / Mention", f"{avg_demo:.2f}x")
            with m6:
                avg_paths = populated_count / filtered_records_count if filtered_records_count > 0 else 0
                st.metric("Paths / Record", f"{avg_paths:.2f}x")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Archival Geospatial Map",
            "📊 Demographic & Crossover Insights",
            "⏳ Archival Calendar Timeline",
            "🔍 Interactive Entity Explorer",
            "📈 Pipeline Quality Diagnostics"
        ])
        
        # --- GIS Map ---
        with tab1:
            st.subheader("Geospatial Entity Distribution")
            df_geo = df_filtered[df_filtered["Latitude"].notna() & df_filtered["Longitude"].notna()].copy()
            if len(df_geo) > 0:
                # Add Icon to hover title
                df_geo["Hover Title"] = df_geo["Icon"] + " " + df_geo["Official Name"]
                
                fig_map = px.scatter_mapbox(
                    df_geo,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Hover Title",
                    hover_data=["Surface Text", "Cohort", "Member Of", "Participant In", "Description"],
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
            st.subheader("Archival Intersectionality & Metadata Distributions")
            
            # --- Row 1: Dynamic Profiles ---
            st.markdown("### 📊 Dynamic Metadata Profiler")
            # Removed Convicted Of
            demo_options = [
                "Occupation", "Ethnic Group/Tribe", "Gender Identity", "Religion", "Country", 
                "Political Ideology", "Member Of", "Participant In"
            ]
            selected_demo = st.selectbox("Select Target Attribute Profile:", options=demo_options, index=0)
            
            df_demo = df_filtered[[selected_demo, "Cohort"]].dropna()
            
            if not df_demo.empty:
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
                    cohort_counts = cohort_counts[cohort_counts[selected_demo].isin(top_10[selected_demo])]
                    fig_cohort = px.bar(
                        cohort_counts, 
                        x="count", 
                        y=selected_demo, 
                        color="Cohort", 
                        orientation='h', 
                        barmode="stack",
                        color_discrete_map=COHORT_COLOR_MAP
                    )
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
                # Default to Political Ideology if available for an interesting default cross-reference
                attr_y = st.selectbox("Select Y-Axis Intersection Attribute:", options=demo_options, index=5)
                
            if attr_x == attr_y:
                st.error("⚠️ Cross-analysis requires selecting two distinct demographic vectors.")
            else:
                df_cross = df_filtered[[attr_x, attr_y]].dropna()
                if not df_cross.empty:
                    df_cross[attr_x] = df_cross[attr_x].apply(lambda x: x.split(", ")[0])
                    df_cross[attr_y] = df_cross[attr_y].apply(lambda x: x.split(", ")[0])
                    
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
        
       # --- Tab 3: Quantitative Temporal Analytics (Color Synced & Pastel) ---
        with tab3:
            st.subheader("⏳ Chronological Distribution & Historical Velocity")
            st.markdown("""
            This layout pairs your high-level category swimlanes with macro temporal metrics 
            and velocity tracking to show exactly where your archive aggregates in time.
            """)

            # Filter out records without a timestamp
            df_time = df_filtered[df_filtered["Target Year"].notna()].copy()

            if df_time.empty:
                st.info("No elements with valid Wikidata timelines or local date stamps match your active filters.")
            else:
                # Force uniform types for chronological calculations
                df_time["Target Year"] = df_time["Target Year"].astype(int)
                df_time["Timeline Label"] = df_time["Icon"] + " " + df_time["Official Name"]
                
                # 🎨 UNIFIED PASTEL COLOR MAP (Ensures 1:1 color parity across all charts)
                PASTEL_COLOR_MAP = {
                    "Person": "#92A8D1",        # Soft Serenity Blue
                    "Organization": "#F7CAC9",  # Pastel Rose Pink
                    "Place": "#B5EAD7",         # Mint Green
                    "Event": "#FFDAC1",         # Soft Apricot Orange
                    "NORP": "#E0BBE4",          # Light Lavender
                    "Thing": "#FFF5BA"          # Pale Cream Yellow
                }
                DEFAULT_PASTEL = "#C7CEEA"      # Muted Periwinkle fallback for unexpected classes

                # ==========================================
                # QUANTITATIVE TEMPORAL METRICS
                # ==========================================
                with st.container(border=True):
                    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                    
                    # 1. Total absolute era coverage
                    start_era = int(df_time["Target Year"].min())
                    end_era = int(df_time["Target Year"].max())
                    t_col1.metric("Chronological Span", f"{start_era} – {end_era}")
                    
                    # 2. Average lifespan of historical actors (excluding events)
                    df_lifespan = df_time[df_time["End Year"].notna() & (~df_time["NER Class"].str.contains("Event", na=False))].copy()
                    if not df_lifespan.empty:
                        avg_life = int((df_lifespan["End Year"].astype(int) - df_lifespan["Target Year"]).mean())
                        t_col2.metric("Avg. Entity Lifespan", f"{avg_life} Years")
                    else:
                        t_col2.metric("Avg. Entity Lifespan", "Static / N/A")
                    
                    # 3. Mode calculation for historical density concentration
                    df_time["Decade"] = (df_time["Target Year"] // 10) * 10
                    peak_decade = df_time["Decade"].mode()
                    if not peak_decade.empty:
                        t_col3.metric("Peak Active Decade", f"{int(peak_decade.iloc[0])}s")
                    else:
                        t_col3.metric("Peak Active Decade", "N/A")
                        
                    # 4. Total count of milestone incidents
                    total_events = df_time["NER Class"].str.contains("Event", na=False).sum()
                    t_col4.metric("Point-in-Time Events", f"{total_events} Milestones")

                st.markdown("---")

                # ==========================================
                # VIEW 1: MACRO TEMPORAL SWIMLANES (Color Synced)
                # ==========================================
                st.markdown("### 🗺️ Macro Density Swimlanes")
                st.markdown("_Look for vertical alignments across tracks to identify cross-category historical triggers._")

                fig_macro = go.Figure()
                categories = df_time["NER Class"].unique()
                
                for cat in categories:
                    df_cat = df_time[df_time["NER Class"] == cat]
                    hover_texts = [
                        f"<b>{row['Timeline Label']}</b><br>Cohort: {row['Cohort']}<br>Start: {row['Target Year']}<br>Desc: {row['Description']}"
                        for _, row in df_cat.iterrows()
                    ]

                    # Extract the locked pastel color for this category
                    assigned_color = PASTEL_COLOR_MAP.get(cat, DEFAULT_PASTEL)

                    fig_macro.add_trace(go.Scatter(
                        x=df_cat["Target Year"],
                        y=[cat] * len(df_cat),
                        mode="markers",
                        marker=dict(
                            size=16, 
                            symbol="diamond" if "Event" in cat else "circle", 
                            color=assigned_color,  # Explicitly overriding color assignment
                            line=dict(width=1, color="white")
                        ),
                        text=hover_texts,
                        hoverinfo="text",
                        name=cat
                    ))

                fig_macro.update_layout(
                    xaxis_title="Calendar Year (Absolute Axis)",
                    yaxis_title="Semantic Swimlanes",
                    height=280,
                    margin=dict(l=150, r=20, t=20, b=40),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_macro, use_container_width=True)

                st.markdown("---")

                # ==========================================
                # VIEW 2: QUANTITATIVE HISTORICAL PULSE (Color Synced)
                # ==========================================
                st.markdown("### 📈 Historical Velocity (Decadal Node Density)")
                st.markdown("_Aggregates graph initialization frequency into intervals to highlight structural data gaps or historical surges._")
                
                # Group by decade and type to showcase stacked composition over time
                decade_counts = df_time.groupby(["Decade", "NER Class"]).size().reset_index(name="Node Count")
                decade_counts["Decade Display"] = decade_counts["Decade"].astype(str) + "s"
                decade_counts = decade_counts.sort_values("Decade")

                fig_pulse = px.bar(
                    decade_counts,
                    x="Decade Display",
                    y="Node Count",
                    color="NER Class",
                    barmode="stack",
                    # Explicitly map the same dictionary map to the categorical bar trace
                    color_discrete_map=PASTEL_COLOR_MAP, 
                    height=350
                )
                
                fig_pulse.update_layout(
                    xaxis_title="Historical Decades",
                    yaxis_title="Entities Introduced",
                    legend_title_text="Entity Class",
                    margin=dict(l=40, r=20, t=20, b=40)
                )
                st.plotly_chart(fig_pulse, use_container_width=True)
        
        # --- Tab 4: Search and Explore Directory ---
        with tab4:
            st.subheader("Knowledge Graph Node Directory")
            search_query = st.text_input("🔍 Search nodes...", "")
            df_display = df_filtered.copy()
            if search_query:
                df_display = df_display[df_display["Surface Text"].str.contains(search_query, case=False, na=False)]
            
            # Configure data columns to make VIAF URL a clickable link
            cols_to_show = ["Icon", "Official Name", "Surface Text", "NER Class", "Political Ideology", "Member Of", "VIAF Link", "Resolution Type"]
            
            st.dataframe(
                df_display[cols_to_show], 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "VIAF Link": st.column_config.LinkColumn(
                        "Authority Data (VIAF)",
                        help="Cross-database linkage to the Virtual International Authority File",
                        display_text="View Profile"
                    )
                }
            )

        # --- Tab 5: Pipeline Quality Diagnostics ---
        with tab5:
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
                # Removed Convicted Of
                attributes = ["Occupation", "Political Ideology", "Member Of", "Participant In", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country", "VIAF Link"]
                completeness = [(df_filtered[col].notna().sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0 for col in attributes]
                
                df_comp = pd.DataFrame({"Attribute": attributes, "Fill Rate (%)": completeness})
                fig_comp = px.bar(df_comp, x="Fill Rate (%)", y="Attribute", orientation='h', color_discrete_sequence=["#9B59B6"])
                fig_comp.update_xaxes(range=[0, 100])
                fig_comp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_comp, use_container_width=True)
