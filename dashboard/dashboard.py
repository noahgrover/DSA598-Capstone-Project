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
            
            # Newly Mapped Rich Properties
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
    # --- GLOBAL SYSTEM PALETTES (IBM Colorblind-Safe Framework) ---
    unique_cohorts = sorted(list(df["Cohort"].unique()))
    HIGH_CONTRAST_COHORT_COLORS = ["#005AB5", "#DC3220", "#FFC20A"] 
    COHORT_COLOR_MAP = {
        cohort: HIGH_CONTRAST_COHORT_COLORS[i % len(HIGH_CONTRAST_COHORT_COLORS)] 
        for i, cohort in enumerate(unique_cohorts)
    }

    IBM_LABEL_COLOR_MAP = {
        "Person": "#648FFF",        # Cornflower Blue
        "Organization": "#785EF0",  # Indigo/Purple
        "Place": "#FE6100",         # High-Contrast Orange
        "Event": "#FFB000",         # Golden Yellow
        "NORP": "#D01C8B",          # Vivid Magenta
        "Thing": "#A3A8B8"          # Fallback Slate Gray
    }

    # Sidebar Filters
    st.sidebar.header("📊 Filter Controls")
    selected_cohorts = st.sidebar.multiselect(
        "Select Historical Cohorts",
        options=unique_cohorts,
        default=unique_cohorts
    )

    # Apply your sidebar filter mask to the main dataset
    df_filtered = df[df["Cohort"].isin(selected_cohorts)]
    
    # Safely derive total documents matching active cohort filter selections
    filtered_records_count = sum(records_per_cohort[c] for c in selected_cohorts)
        
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
            df_geo["Hover Title"] = df_geo["Icon"] + " " + df_geo["Official Name"]
            
            fig_map = px.scatter_mapbox(
                df_geo,
                lat="Latitude",
                lon="Longitude",
                hover_name="Hover Title",
                hover_data=["Surface Text", "Cohort", "Member Of", "Participant In", "Description"],
                color="NER Class",                      # 1. Switched from 'Visual Group' to 'NER Class'
                color_discrete_map=IBM_LABEL_COLOR_MAP, # 2. Enforced the global color blind safe palette
                zoom=2,
                height=600
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=20, b=0) # Tightens up margins for a cleaner UI
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geospatial data coordinates found in filtered dataset.")

    # --- Tab 2: Enhanced Demographic Analysis ---
    with tab2:
        st.subheader("Archival Intersectionality & Metadata Distributions")
        
        st.markdown("### 📊 Dynamic Metadata Profiler")
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
                
                # FIX 1: Swapped out random green for official IBM Magenta 50 (#DC267F)
                fig_demo = px.bar(
                    top_10, 
                    x="count", 
                    y=selected_demo, 
                    orientation='h', 
                    color_discrete_sequence=["#DC267F"] 
                )
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
        
        st.markdown("### 🔀 Intersectionality Matrix")
        st.markdown("Cross-reference any two vectors below to locate structural overlaps hidden across your semantic metadata graph.")
        
        cx_col1, cx_col2 = st.columns(2)
        with cx_col1:
            attr_x = st.selectbox("Select X-Axis Intersection Attribute:", options=demo_options, index=0)
        with cx_col2:
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
                    
                    # FIX 2: Swapped out "Viridis" for a high-contrast, single-hue IBM Ultramarine sequence
                    # Transitions cleanly from a neutral off-white background right up to dominant IBM Blue
                    fig_heatmap = px.density_heatmap(
                        cross_matrix, 
                        x=attr_x, 
                        y=attr_y, 
                        z="Co-occurrences",
                        text_auto=True,
                        color_continuous_scale=["#F4F6FF", "#648FFF", "#002D9C"]
                    )
                    fig_heatmap.update_layout(xaxis_title=attr_x, yaxis_title=attr_y)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No explicit intersections found for the top elements of these attributes.")
            else:
                st.info("No co-occurring data coordinates available for this configuration.")
        
    # --- Tab 3: Quantitative Temporal Analytics (Unified IBM Colors) ---
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
            # VIEW 1: MACRO SWIMLANES (Enforcing Master Palette)
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

                # Grab the locked color from our global master dictionary
                assigned_color = IBM_LABEL_COLOR_MAP.get(cat, IBM_LABEL_COLOR_MAP["Thing"])

                fig_macro.add_trace(go.Scatter(
                    x=df_cat["Target Year"],
                    y=[cat] * len(df_cat),
                    mode="markers",
                    marker=dict(
                        size=16, 
                        symbol="diamond" if "Event" in cat else "circle", 
                        color=assigned_color,
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
            # VIEW 2: HISTORICAL PULSE (Enforcing Master Palette)
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
                color_discrete_map=IBM_LABEL_COLOR_MAP, 
                height=350
            )
            
            fig_pulse.update_layout(
                xaxis_title="Historical Decades",
                yaxis_title="Entities Introduced",
                legend_title_text="Entity Class",
                margin=dict(l=40, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_pulse, use_container_width=True)

    # --- Tab 4: Interactive Entity Explorer ---
    with tab4:
        st.subheader("🔍 Interactive Entity Explorer & Profile Graph")
        st.markdown("""
        Search, filter, and inspect specific semantic nodes within your graph. Selecting an entity 
        will pull its full relational dossier, authority records, multi-layered attributes, and corpus occurrences.
        """)

        if df_filtered.empty:
            st.info("No entities match your active cohort filters to explore.")
        else:
            # 1. Top Search and Filter Bar
            exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 1])
            with exp_col1:
                search_query = st.text_input("🔍 Search by Entity Name or Surface Text:", value="")
            with exp_col2:
                class_options = ["All Categories"] + list(df_filtered["NER Class"].dropna().unique())
                selected_class = st.selectbox("Filter Explorer by Class:", options=class_options)
            with exp_col3:
                res_options = ["All Resolutions"] + list(df_filtered["Resolution Type"].dropna().unique())
                selected_res = st.selectbox("Filter Explorer by Resolution:", options=res_options)

            # Apply Search Filters
            df_exp = df_filtered.copy()
            if search_query:
                df_exp = df_exp[
                    df_exp["Official Name"].str.contains(search_query, case=False, na=False) |
                    df_exp["Surface Text"].str.contains(search_query, case=False, na=False)
                ]
            if selected_class != "All Categories":
                df_exp = df_exp[df_exp["NER Class"] == selected_class]
            if selected_res != "All Resolutions":
                df_exp = df_exp[df_exp["Resolution Type"] == selected_res]

            if df_exp.empty:
                st.warning("No records match your specific search criteria.")
            else:
                # 2. Deduplication & Mention Aggregation
                mention_counts = df_exp.groupby("Entity ID").size().to_dict()
                
                df_nodes = df_exp.drop_duplicates(subset=["Entity ID"]).copy()
                df_nodes["Mentions"] = df_nodes["Entity ID"].map(mention_counts)
                df_nodes = df_nodes.sort_values(by="Mentions", ascending=False)

                # =========================================================
                # 3. REORDERED SPLIT LAYOUT (Entity Card First - 60% Width)
                # =========================================================
                detail_col, master_col = st.columns([3, 2])

                # Process selection on the right column first to obtain `selected_row`
                with master_col:
                    st.markdown(f"**Unique Graph Nodes ({len(df_nodes)})**")
                    
                    df_nodes["Select Label"] = (
                        df_nodes["Icon"] + " " + 
                        df_nodes["Official Name"] + 
                        " (" + df_nodes["Mentions"].astype(str) + " mention" + 
                        df_nodes["Mentions"].apply(lambda x: "s" if x > 1 else "") + ")"
                    )

                    selected_label = st.selectbox(
                        "Click below to select an active entity dossier:", 
                        options=df_nodes["Select Label"].tolist(),
                        label_visibility="collapsed"
                    )
                    
                    df_table_view = df_nodes[["Icon", "Official Name", "NER Class", "Mentions", "Resolution Type"]].copy()
                    
                    st.dataframe(
                        df_table_view, 
                        use_container_width=True, 
                        height=520,
                        hide_index=True
                    )

                # Render the detailed Entity Profile Card on the left (first position)
                with detail_col:
                    selected_row = df_nodes[df_nodes["Select Label"] == selected_label].iloc[0]

                    with st.container(border=True):
                        # Title Header Block
                        st.markdown(f"### {selected_row['Icon']} {selected_row['Official Name']}")
                        st.caption(
                            f"**ID:** `{selected_row['Entity ID']}` | "
                            f"**Class:** {selected_row['NER Class']} | "
                            f"**Total Mentions:** {selected_row['Mentions']}"
                        )
                        
                        # Type-safe image rendering
                        raw_img = selected_row.get("Image URL")
                        if isinstance(raw_img, list) and len(raw_img) > 0:
                            raw_img = raw_img[0]
                        if isinstance(raw_img, dict):
                            raw_img = raw_img.get("schema:url", raw_img.get("@id", ""))

                        if isinstance(raw_img, str) and raw_img.strip() and raw_img.strip().lower() not in ("none", "nan"):
                            try:
                                st.image(raw_img.strip(), use_container_width=True)
                            except Exception:
                                st.caption("🖼️ *(Image link present but unreachable)*")

                        # Description
                        st.markdown(f"**Description:** \n> {selected_row['Description']}")
                        
                        st.markdown("---")
                        st.markdown("**Graph Attributes & Linked Assertions:**")

                        # Metadata display grid
                        metadata_fields = {
                            "Primary Mention Text": selected_row["Surface Text"],
                            "Timeline Active Bounds": f"{selected_row['Target Year'] or '???'} – {selected_row['End Year'] or '???'}",
                            "Resolution Link Status": selected_row["Resolution Type"],
                            "Primary Cohort": selected_row["Cohort"],
                            "Occupation/Role": selected_row["Occupation"],
                            "Affiliated Country": selected_row["Country"],
                            "Member Of": selected_row["Member Of"],
                            "Participant In": selected_row["Participant In"],
                            "Political Ideology": selected_row["Political Ideology"],
                            "Ethnic Group/Tribe": selected_row["Ethnic Group/Tribe"],
                            "Stated Religion": selected_row["Religion"],
                            "Gender Identity": selected_row["Gender Identity"],
                        }

                        for label, val in metadata_fields.items():
                            if pd.notna(val) and str(val).strip() and val != "??? – ???":
                                st.markdown(f"**{label}:** {val}")

                        st.markdown("---")
                        st.markdown("**📄 Corpus Mentions & Provenance:**")

                        # Look up all mentions in active cohort dataset for this Entity ID
                        all_mentions = df_filtered[df_filtered["Entity ID"] == selected_row["Entity ID"]]

                        # Surface text variations across documents
                        surface_variants = [str(s) for s in all_mentions["Surface Text"].unique() if pd.notna(s) and str(s).strip()]
                        if surface_variants:
                            variant_tags = " ".join([f"`{v}`" for v in surface_variants])
                            st.markdown(f"**Archival Text Variants:** {variant_tags}")

                        # Breakdown across cohorts
                        cohort_counts = all_mentions["Cohort"].value_counts()
                        with st.expander(f"View Distribution Across Cohorts ({len(all_mentions)} total mentions)", expanded=False):
                            for c_name, count in cohort_counts.items():
                                st.markdown(f"• **{c_name}**: {count} mention{'s' if count > 1 else ''}")

                        st.markdown("---")

                        # Authority Link Button
                        if pd.notna(selected_row["VIAF Link"]):
                            st.link_button("🌐 Open VIAF Authority File", selected_row["VIAF Link"], use_container_width=True)
                        elif "wd:" in selected_row["Entity ID"]:
                            wd_url = f"https://www.wikidata.org/wiki/{selected_row['Entity ID'].replace('wd:', '')}"
                            st.link_button("🌐 View Entity on Wikidata", wd_url, use_container_width=True)

# --- Tab 5: Pipeline Quality Diagnostics ---
    with tab5:
        st.subheader("📈 Pipeline Quality Diagnostics & NIL Cluster Analytics")
        st.markdown("""
        Assess structural accuracy, resolution efficacy, and metadata completeness across the extraction pipeline.
        This includes deep-dive metrics into **NIL (Not-In-Lexicon) clustering**—grouping local, unlinked entities across the archive.
        """)

        if df_filtered.empty:
            st.info("No data available for quality diagnostics based on current filters.")
        else:
            # 1. High-Level Quality & NIL Metrics
            df_nil = df_filtered[df_filtered["Resolution Type"] == "NIL Clustered"]
            df_resolved = df_filtered[df_filtered["Resolution Type"] == "Wikidata Resolved"]
            df_unlinked = df_filtered[df_filtered["Resolution Type"] == "Unlinked Entity"]
            
            total_mentions = len(df_filtered)
            nil_mentions_count = len(df_nil)
            unique_nil_clusters = df_nil["Entity ID"].nunique()
            
            # Clustering Efficiency: % of non-Wikidata entities successfully grouped into local clusters
            total_non_wikidata = nil_mentions_count + len(df_unlinked)
            nil_cluster_efficiency = (nil_mentions_count / total_non_wikidata * 100) if total_non_wikidata > 0 else 0
            
            avg_conf = df_filtered["Confidence"].mean() if "Confidence" in df_filtered and not df_filtered["Confidence"].isna().all() else 0.0

            with st.container(border=True):
                q1, q2, q3, q4, q5 = st.columns(5)
                with q1: st.metric("Wikidata Linking Rate", f"{(len(df_resolved)/total_mentions*100):.1f}%")
                with q2: st.metric("Total NIL Mentions", nil_mentions_count)
                with q3: st.metric("Unique NIL Clusters", unique_nil_clusters)
                with q4: st.metric("NIL Clustering Efficiency", f"{nil_cluster_efficiency:.1f}%")
                with q5: st.metric("Avg NER Confidence", f"{avg_conf:.2f}")

            st.markdown("---")

            # 2. Section 1: Resolution Breakdown & NIL Cluster Deep-Dive
            res_col1, res_col2 = st.columns(2)

            # EXTENDED IBM CARBON PALETTE (Distinct Diagnostic Shades)
            RESOLUTION_COLOR_MAP = {
                "Wikidata Resolved": "#009D9A",  # IBM Carbon Teal 40
                "NIL Clustered": "#4589FF",      # IBM Carbon Cerulean / Blue 40
                "Unlinked Entity": "#8D8D8D"     # IBM Carbon Cool Gray 50
            }

            with res_col1:
                st.markdown("#### Entity Resolution Distribution")
                res_counts = df_filtered["Resolution Type"].value_counts().reset_index()
                fig_res = px.pie(
                    res_counts, 
                    values="count", 
                    names="Resolution Type", 
                    color="Resolution Type",
                    color_discrete_map=RESOLUTION_COLOR_MAP,
                    hole=0.4
                )
                fig_res.update_traces(textposition='inside', textinfo='percent+label')
                fig_res.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_res, use_container_width=True)

            with res_col2:
                st.markdown("#### Largest NIL Entity Clusters")
                if not df_nil.empty:
                    top_nil = df_nil.groupby("Official Name").agg(
                        mentions=("Entity ID", "count"),
                        cohort_span=("Cohort", "nunique")
                    ).reset_index().sort_values("mentions", ascending=False).head(8)

                    # IBM Carbon Deep Teal (#005D5D) for cluster representation
                    fig_nil = px.bar(
                        top_nil,
                        x="mentions",
                        y="Official Name",
                        orientation="h",
                        color_discrete_sequence=["#005D5D"],
                        hover_data=["cohort_span"]
                    )
                    fig_nil.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis_title="Mentions in Cluster",
                        yaxis_title="NIL Cluster Entity",
                        margin=dict(t=30, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_nil, use_container_width=True)
                else:
                    st.info("No NIL clustered entities present in current selection.")
