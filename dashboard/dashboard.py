# =========================================================================================================================================
# DSA 598 Capstone Project - Streamlit Dashboard - Grover
# =========================================================================================================================================

import json
import re
from pathlib import Path
from collections import Counter
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# page configurations
st.set_page_config(
    page_title="Marginalized Metadata Enrichment",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MARGINALIZED METADATA ENRICHMENT: DASHBOARD")
st.markdown("""
This dashboard analyzes the structural and qualitative improvements to archival metadata extracted from the Digital Public Library of America (DPLA) for three distinct, historically marginalized cohorts. The pipeline:
- Extracts title and description fields from digital records stored in DPLA;
- Passes them into flattened JSON records;
- Recognizes and extracts named entities;
- Locates viable Wikidata candidates;
- Links the correct candidate to each entity;
- Produces enriched JSONLD for linked entities and clusters NIL (out-of-network) entities.
""")

def extract_year_from_text(text):
    # helper utility to parse a numeric year out of messy strings or ISO dates.
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
        # extracts human-readable names from rich JSON-LD objects.
        if isinstance(val, list):
            return [v.get("schema:name", v.get("@id", "").replace("wd:", "")) if isinstance(v, dict) else str(v) for v in val]
        elif isinstance(val, dict):
            return [val.get("schema:name", val.get("@id", "").replace("wd:", ""))]
        elif val:
            return [str(val).replace("wd:", "")]
        return []

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

            # base demographics
            occ = extract_labels(ent.get("occupation"))
            gender = extract_labels(ent.get("genderIdentity"))
            ethnic = extract_labels(ent.get("ethnicGroup"))
            religion = extract_labels(ent.get("religion"))
            country = extract_labels(ent.get("country"))
            
            # rich properties
            ideology = extract_labels(ent.get("politicalIdeology"))
            member = extract_labels(ent.get("memberOf"))
            participant = extract_labels(ent.get("participant"))
            
            # extracting and formatting VIAF links
            viaf_url = None
            same_as = ent.get("schema:sameAs")
            if same_as and isinstance(same_as, str) and same_as.startswith("viaf:"):
                viaf_id = same_as.replace("viaf:", "")
                viaf_url = f"https://viaf.org/viaf/{viaf_id}/"

            # parse icons
            icon_str = ent.get("visualIcon", "help")
            emoji_icon = icon_mapping.get(icon_str, "❓")
            
            # dynamically handle start/end bounds
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

    # join list attributes into simple display strings
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
    # IBM colorblind-safe color scheme
    unique_cohorts = sorted(list(df["Cohort"].unique()))
    HIGH_CONTRAST_COHORT_COLORS = ["#005AB5", "#DC3220", "#FFC20A"] 
    COHORT_COLOR_MAP = {
        cohort: HIGH_CONTRAST_COHORT_COLORS[i % len(HIGH_CONTRAST_COHORT_COLORS)] 
        for i, cohort in enumerate(unique_cohorts)
    }

    IBM_LABEL_COLOR_MAP = {
        "Person": "#648FFF",        # cornflower blue
        "Organization": "#785EF0",  # indigo/purple
        "Place": "#FE6100",         # high-contrast orange
        "Event": "#FFB000",         # golden yellow
        "NORP": "#D01C8B",          # vivid magenta
        "Thing": "#A3A8B8"          # fallback slate gray
    }

    # sidebar filters
    st.sidebar.header("Global Dashboard Filters")
    selected_cohorts = st.sidebar.pills(
    "Select historical cohort(s)",
    options=unique_cohorts,
    default=unique_cohorts,
    selection_mode="multi"  # allows selecting multiple cohort buttons
    )

    # apply filter mask to data
    df_filtered = df[df["Cohort"].isin(selected_cohorts)]
    
    filtered_records_count = sum(records_per_cohort[c] for c in selected_cohorts)
        
    # persistent metrics row
    with st.container(border=True):
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: st.metric("Total Records", filtered_records_count)
        with m2:
            relational_columns = ["Occupation", "Gender Identity", "Ethnic Group/Tribe", "Religion", "Country", "Political Ideology", "Member Of", "Participant In"]
            populated_count = df_filtered[relational_columns].notna().sum().sum()
            avg_paths = populated_count / filtered_records_count if filtered_records_count > 0 else 0
            st.metric("Paths / Record", f"{avg_paths:.2f}x")    
        with m3: st.metric("Entity Mentions", len(df_filtered))
        with m4:
            avg_demo = populated_count / len(df_filtered) if len(df_filtered) > 0 else 0
            st.metric("Metadata / Mention", f"{avg_demo:.2f}x")
        with m5: st.metric("Unique Entity Nodes", df_filtered["Entity ID"].nunique())
        with m6:
            viaf_links = df_filtered["VIAF Link"].notna().sum()
            st.metric("VIAF Authority Links", viaf_links)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "| Entity Explorer",
        "| Semantic Density Network",
        "| Demographic Analysis",
        "| Geospatial Map",
        "| Timeline",
        "| Pipeline Quality Diagnostics"
    ])

# =========================================================================================================================================
# Tab 1: Entity Explorer
# =========================================================================================================================================
    
    with tab1:
        st.markdown("""
        Search, filter, and inspect specific semantic nodes within the graph. Selecting an entity from the list of unique nodes on the right
        will pull its full relational dossier, authority records, multi-layered attributes, and corpus occurrences.
        """)

        if df_filtered.empty:
            st.info("No entities match your active cohort filters to explore.")
        else:
            # top search and filter bar
            exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 1])
            with exp_col1:
                search_query = st.text_input("🔍 SEARCH (NAME OR SURFACE TEXT):", value="")
            with exp_col2:
                class_options = ["All Categories"] + list(df_filtered["NER Class"].dropna().unique())
                selected_class = st.selectbox("FILTER ENTITIES (CLASS):", options=class_options)
            with exp_col3:
                res_options = ["All Resolutions"] + list(df_filtered["Resolution Type"].dropna().unique())
                selected_res = st.selectbox("FILTER ENTITIES (RESOLUTION):", options=res_options)

            # apply search filters
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
                df_nodes = df_nodes.sort_values(by="Mentions", ascending=False).reset_index(drop=True)

                # splits layout
                detail_col, master_col = st.columns([3, 2])

                # render master selection table on the right column
                with master_col:
                    st.markdown(f"**Unique Graph Nodes ({len(df_nodes)})**")
                    st.caption("💡 *Click any row in the table to inspect its dossier on the left.*")

                    df_table_view = df_nodes[["Icon", "Official Name", "NER Class", "Mentions", "Resolution Type"]].copy()

                    # interactive dataframe with native row selection
                    selection_event = st.dataframe(
                        df_table_view, 
                        use_container_width=True, 
                        height=520,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row"
                    )

                    # default to row index 0 if nothing selected
                    selected_index = 0
                    if hasattr(selection_event, "selection") and selection_event.selection and selection_event.selection.rows:
                        selected_index = selection_event.selection.rows[0]

                # render the detailed entity profile card on the left
                with detail_col:
                    selected_row = df_nodes.iloc[selected_index]

                    with st.container(border=True):
                        # Title Header Block
                        st.markdown(f"### {selected_row.get('Icon', '📌')} {selected_row.get('Official Name', 'Unknown')}")
                        st.caption(
                            f"**ID:** `{selected_row.get('Entity ID', 'N/A')}` | "
                            f"**Class:** {selected_row.get('NER Class', 'N/A')} | "
                            f"**Total Mentions:** {selected_row.get('Mentions', 1)}"
                        )

                        # image rendering with size constraints
                        raw_img = selected_row.get("Image URL")
                        if isinstance(raw_img, list) and len(raw_img) > 0:
                            raw_img = raw_img[0]
                        if isinstance(raw_img, dict):
                            raw_img = raw_img.get("schema:url", raw_img.get("@id", ""))

                        if isinstance(raw_img, str) and raw_img.strip() and raw_img.strip().lower() not in ("none", "nan"):
                            try:
                                st.image(raw_img.strip(), width=180, caption=selected_row.get('Official Name'))
                            except Exception:
                                st.caption("🖼️ *(Image link present but unreachable)*")

                        # description
                        if pd.notna(selected_row.get('Description')) and str(selected_row.get('Description')).strip():
                            st.markdown(f"**Description:** \n> {selected_row['Description']}")

                        st.markdown("---")
                        st.markdown("**Graph Attributes & Linked Assertions:**")

                        # metadata display grid
                        target_yr = selected_row.get('Target Year')
                        end_yr = selected_row.get('End Year')
                        time_bounds = f"{target_yr if pd.notna(target_yr) else '???'} – {end_yr if pd.notna(end_yr) else '???'}"

                        metadata_fields = {
                            "Primary Mention Text": selected_row.get("Surface Text"),
                            "Timeline Active Bounds": time_bounds,
                            "Primary Cohort": selected_row.get("Cohort"),
                            "Occupation/Role": selected_row.get("Occupation"),
                            "Affiliated Country": selected_row.get("Country"),
                            "Member Of": selected_row.get("Member Of"),
                            "Participant In": selected_row.get("Participant In"),
                            "Political Ideology": selected_row.get("Political Ideology"),
                            "Ethnic Group/Tribe": selected_row.get("Ethnic Group/Tribe"),
                            "Stated Religion": selected_row.get("Religion"),
                            "Gender Identity": selected_row.get("Gender Identity"),
                        }

                        for label, val in metadata_fields.items():
                            if pd.notna(val) and str(val).strip() and val != "??? – ???":
                                st.markdown(f"**{label}:** {val}")

                        st.markdown("---")
                        st.markdown("**📄 Corpus Mentions & Provenance:**")

                        # look up all mentions in active cohort dataset for this entity ID
                        all_mentions = df_filtered[df_filtered["Entity ID"] == selected_row["Entity ID"]]

                        # surface text variations across documents
                        surface_variants = [str(s) for s in all_mentions["Surface Text"].unique() if pd.notna(s) and str(s).strip()]
                        if surface_variants:
                            variant_tags = " ".join([f"`{v}`" for v in surface_variants])
                            st.markdown(f"**Archival Text Variants:** {variant_tags}")

                        # breakdown across cohorts
                        cohort_counts = all_mentions["Cohort"].value_counts()
                        with st.expander(f"View Distribution Across Cohorts ({len(all_mentions)} total mentions)", expanded=False):
                            for c_name, count in cohort_counts.items():
                                st.markdown(f"• **{c_name}**: {count} mention{'s' if count > 1 else ''}")

                        st.markdown("---")

                        # authority link button
                        viaf_link = selected_row.get("VIAF Link")
                        if pd.notna(viaf_link) and str(viaf_link).strip().startswith("http"):
                            st.link_button("🌐 Open VIAF Authority File", str(viaf_link), use_container_width=True)
                        elif pd.notna(selected_row.get("Entity ID")) and "wd:" in str(selected_row["Entity ID"]):
                            wd_url = f"https://www.wikidata.org/wiki/{str(selected_row['Entity ID']).replace('wd:', '')}"
                            st.link_button("🌐 View Entity on Wikidata", wd_url, use_container_width=True)

# =========================================================================================================================================
# Tab 2: Semantic Density Network 
# =========================================================================================================================================
    
    with tab2:
        st.markdown("""
        Explore structural relationships and co-occurrence density across entity nodes. 
        Node sizes represent **mention density** (degree/frequency), edge thickness indicates **co-occurrence strength**, 
        and colors correspond to **NER entity classes**.
        """)

        # guard against empty datasets
        df_net_clean = df_filtered.dropna(subset=["Entity ID"]).copy() if not df_filtered.empty else pd.DataFrame()

        if df_net_clean.empty:
            st.info("No entity data available to construct network topology.")
        else:
            # cast key columns to string to prevent float/str mismatch crashes
            df_net_clean["Entity ID"] = df_net_clean["Entity ID"].astype(str)
            df_net_clean["Official Name"] = df_net_clean["Official Name"].fillna("Unknown Entity").astype(str)
            df_net_clean["NER Class"] = df_net_clean["NER Class"].fillna("UNKNOWN").astype(str)
            df_net_clean["Icon"] = df_net_clean["Icon"].fillna("📌").astype(str)

            # controls to adjust network complexity
            net_col1, net_col2 = st.columns([2, 1])
            with net_col1:
                top_n = st.slider("Limit Top Entities by Mention Count (for clarity):", min_value=10, max_value=100, value=30, step=5)
            with net_col2:
                layout_algorithm = st.selectbox(
                    "Graph Layout Algorithm:", 
                    ["Spring (Fruchterman-Reingold)", "Circular"]
                )

            # filter down to top N entities
            mention_counts = df_net_clean.groupby("Entity ID").size().to_dict()
            top_entity_ids = sorted(mention_counts, key=mention_counts.get, reverse=True)[:top_n]
        
            df_net_subset = df_net_clean[df_net_clean["Entity ID"].isin(top_entity_ids)].copy()

            if len(df_net_subset) < 2:
                st.warning("Not enough distinct entities selected to form a relational network.")
            else:
                try:
                    # construct networkx graph
                    G = nx.Graph()

                    # add nodes safely
                    for _, row in df_net_subset.drop_duplicates(subset=["Entity ID"]).iterrows():
                        e_id = row["Entity ID"]
                        G.add_node(
                            e_id, 
                            name=row["Official Name"], 
                            ner_class=row["NER Class"], 
                            mentions=mention_counts.get(e_id, 1), 
                            icon=row["Icon"]
                        )

                    # add edges based on shared cohort co-occurrence
                    if "Cohort" in df_net_subset.columns:
                        for cohort_name, group in df_net_subset.groupby("Cohort"):
                            if pd.isna(cohort_name):
                                continue
                            e_ids = [str(x) for x in group["Entity ID"].unique() if str(x) in G.nodes]
                            # Create co-occurrence edges between cohort members
                            for i in range(len(e_ids)):
                                for j in range(i + 1, len(e_ids)):
                                    u, v = e_ids[i], e_ids[j]
                                    if u != v:
                                        if G.has_edge(u, v):
                                            G[u][v]["weight"] += 1
                                        else:
                                            G.add_edge(u, v, weight=1)

                    # compute node positions based on selected layout
                    if "Circular" in layout_algorithm:
                        pos = nx.circular_layout(G)
                    else:
                        pos = nx.spring_layout(G, k=0.55, seed=42)

                    # extract edge traces for plotly
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1.0, color="rgba(140, 140, 140, 0.35)"),
                        hoverinfo="none",
                        mode="lines"
                    )

                    # extract node traces for plotly
                    node_x, node_y, node_colors, node_sizes, node_hover_text, node_labels = [], [], [], [], [], []

                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                    
                        meta = G.nodes[node]
                        ner_cls = meta.get("ner_class", "UNKNOWN")
                        mentions = meta.get("mentions", 1)
                        name = meta.get("name", node)
                        icon = meta.get("icon", "📌")
                    
                        node_labels.append(name)
                        node_colors.append(IBM_LABEL_COLOR_MAP.get(ner_cls, "#8D8D8D"))
                        node_sizes.append(max(12, min(38, 8 + mentions * 2)))
                        node_hover_text.append(f"<b>{icon} {name}</b><br>Class: {ner_cls}<br>Total Mentions: {mentions}")

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode="markers+text",
                        hoverinfo="text",
                        text=node_labels,
                        textposition="top center",
                        textfont=dict(size=10, color="#E0E0E0"),
                        hovertext=node_hover_text,
                        marker=dict(
                            color=node_colors,
                            size=node_sizes,
                            line=dict(width=1.5, color="#FFFFFF")
                        )
                    )

                    # render plotly graph
                    fig_net = go.Figure(data=[edge_trace, node_trace])
                    fig_net.update_layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=10, l=10, r=10, t=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=650
                    )

                    st.plotly_chart(fig_net, use_container_width=True)

                except Exception as e:
                    st.error(f"Error rendering network topology: {str(e)}")
        
# =========================================================================================================================================
# Tab 3 : 
# =========================================================================================================================================
    
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

# =========================================================================================================================================
# Tab 4 : 
# =========================================================================================================================================
    
    with tab4:
        st.subheader("🌐 Geospatial Entity Distribution & Spatial Density")
        st.markdown("""
        Inspect the geographical footprint of resolved entity nodes across your archival corpus. 
        Toggle between **Weighted Markers** to inspect individual entity locations and categories, 
        or **Density Heatmap** to identify broader historical epicenters and spatial concentrations.
        """)
        
        df_geo = df_filtered[df_filtered["Latitude"].notna() & df_filtered["Longitude"].notna()].copy()
        
        if not df_geo.empty:
            # 1. Aggregate mention counts per unique geospatial entity node
            geo_mention_counts = df_geo.groupby("Entity ID").size().to_dict()
            df_geo_nodes = df_geo.drop_duplicates(subset=["Entity ID"]).copy()
            df_geo_nodes["Mentions"] = df_geo_nodes["Entity ID"].map(geo_mention_counts)
            
            # 2. Build detailed hover labels with mention counts
            df_geo_nodes["Hover Title"] = (
                df_geo_nodes["Icon"] + " " + 
                df_geo_nodes["Official Name"] + 
                " (" + df_geo_nodes["Mentions"].astype(str) + " mention" + 
                df_geo_nodes["Mentions"].apply(lambda x: "s" if x > 1 else "") + ")"
            )
            
            # 3. View Switcher Control
            map_view = st.radio(
                "Select Map Display Mode:", 
                options=["Weighted Markers", "Density Heatmap"], 
                horizontal=True
            )
            
            # 4. Conditional Map Rendering
            if map_view == "Weighted Markers":
                fig_map = px.scatter_mapbox(
                    df_geo_nodes,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Hover Title",
                    hover_data={
                        "Mentions": True,
                        "NER Class": True,
                        "Cohort": True,
                        "Country": True,
                        "Latitude": False,
                        "Longitude": False
                    },
                    color="NER Class",
                    color_discrete_map=IBM_LABEL_COLOR_MAP,
                    size="Mentions",
                    size_max=28,
                    zoom=2,
                    height=600
                )
            else:
                # Custom IBM Ultramarine sequential gradient for dark theme map
                fig_map = px.density_mapbox(
                    df_geo_nodes,
                    lat="Latitude",
                    lon="Longitude",
                    z="Mentions",
                    hover_name="Hover Title",
                    hover_data={
                        "Mentions": True,
                        "Country": True,
                        "Latitude": False,
                        "Longitude": False
                    },
                    radius=22,
                    zoom=2,
                    height=600,
                    color_continuous_scale=["#161616", "#648FFF", "#785EF0", "#DC267F"]
                )
            
            # Hardcoded to carto-darkmatter tile style
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geospatial coordinates found for the selected entity filters.")

# =========================================================================================================================================
# Tab 5 : 
# =========================================================================================================================================
    
    with tab5:
        st.subheader("📈 Pipeline Quality Diagnostics & Model Benchmarks")
        st.markdown("""
        Assess structural accuracy, resolution efficacy, and metadata completeness across the extraction pipeline.
        This includes empirical ground-truth benchmark evaluation (In-KB Linking, NIL Precision/Recall/F1, and Candidate Recall@5) 
        segmented by historical archival cohort.
        """)

        if df_filtered.empty:
            st.info("No data available for quality diagnostics based on current filters.")
        else:
            # 1. High-Level Quality & KPI Metric Highlights
            df_nil = df_filtered[df_filtered["Resolution Type"] == "NIL Clustered"]
            df_resolved = df_filtered[df_filtered["Resolution Type"] == "Wikidata Resolved"]
            
            total_mentions = len(df_filtered)
            nil_mentions_count = len(df_nil)
            unique_nil_clusters = df_nil["Entity ID"].nunique()
            
            avg_conf = df_filtered["Confidence"].mean() if "Confidence" in df_filtered and not df_filtered["Confidence"].isna().all() else 0.0

            with st.container(border=True):
                q1, q2, q3, q4, q5 = st.columns(5)
                with q1: st.metric("Wikidata Linking Rate", f"{(len(df_resolved)/total_mentions*100):.1f}%" if total_mentions > 0 else "0.0%")
                with q2: st.metric("Global In-KB F1", "81.22%")
                with q3: st.metric("Global NIL F1", "84.06%")
                with q4: st.metric("Candidate Recall@5", "72.06%")
                with q5: st.metric("Avg NER Confidence", f"{avg_conf:.2f}")

            st.markdown("---")

            # 2. Section 1: Empirical Ground-Truth Model Benchmarks
            st.markdown("### 🎯 Empirical Model Performance Benchmarks")
            st.markdown("Ground-truth evaluation results across global baseline and segmented historical archival cohorts.")

            benchmark_data = [
                {
                    "Scope / Cohort": "GLOBAL BASELINE (All Cohorts)",
                    "Candidate Recall@5": 72.06,
                    "In-KB Precision": 92.54,
                    "In-KB Recall": 72.37,
                    "In-KB F1": 81.22,
                    "NIL Precision": 84.06,
                    "NIL Recall": 84.06,
                    "NIL F1": 84.06
                },
                {
                    "Scope / Cohort": "Cohort B (LGBTQIA+ Histories)",
                    "Candidate Recall@5": 86.59,
                    "In-KB Precision": 94.59,
                    "In-KB Recall": 89.74,
                    "In-KB F1": 92.11,
                    "NIL Precision": 80.70,
                    "NIL Recall": 80.70,
                    "NIL F1": 80.70
                },
                {
                    "Scope / Cohort": "Cohort A (Racial/Ethnic Minorities)",
                    "Candidate Recall@5": 66.29,
                    "In-KB Precision": 94.92,
                    "In-KB Recall": 65.12,
                    "In-KB F1": 77.24,
                    "NIL Precision": 84.21,
                    "NIL Recall": 84.21,
                    "NIL F1": 84.21
                },
                {
                    "Scope / Cohort": "Cohort C (Indigenous Populations)",
                    "Candidate Recall@5": 65.35,
                    "In-KB Precision": 88.24,
                    "In-KB Recall": 64.52,
                    "In-KB F1": 74.53,
                    "NIL Precision": 91.67,
                    "NIL Recall": 91.67,
                    "NIL F1": 91.67
                }
            ]
            df_benchmarks = pd.DataFrame(benchmark_data)

            m_col1, m_col2 = st.columns([1, 1])

            with m_col1:
                st.markdown("#### Cohort Metric Comparison")
                
                df_bench_melted = df_benchmarks.melt(
                    id_vars=["Scope / Cohort"],
                    value_vars=["Candidate Recall@5", "In-KB F1", "NIL F1"],
                    var_name="Metric",
                    value_name="Score (%)"
                )
                
                # Dedicated, 100% unique IBM Carbon Palette for benchmark metrics
                BENCHMARK_COLOR_MAP = {
                    "Candidate Recall@5": "#EE5396",  # IBM Carbon Pink 50
                    "In-KB F1":           "#D2A100",  # IBM Carbon Gold 30
                    "NIL F1":             "#0F62FE"   # IBM Carbon Signature Blue 60
                }

                fig_bench = px.bar(
                    df_bench_melted,
                    x="Scope / Cohort",
                    y="Score (%)",
                    color="Metric",
                    barmode="group",
                    color_discrete_map=BENCHMARK_COLOR_MAP,
                    height=320
                )
                
                fig_bench.add_hline(
                    y=75,
                    line_dash="dash",
                    line_color="#6F6F6F",
                    annotation_text="75% Baseline Threshold",
                    annotation_position="bottom right",
                    annotation_font=dict(size=11, color="#6F6F6F")
                )

                fig_bench.update_layout(
                    xaxis_title="",
                    yaxis_title="Score (%)",
                    yaxis=dict(range=[0, 100]),
                    margin=dict(t=20, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bench, use_container_width=True)

            with m_col2:
                st.markdown("#### Performance Matrix & Definitions")
                
                metric_descriptions = {
                    "Candidate Recall@5": "Rate at which the true entity is retrieved within the top 5 search candidates.",
                    "In-KB Precision": "Accuracy: % of pipeline-assigned Wikidata links that are historically correct.",
                    "In-KB Recall": "Coverage: % of actual Wikidata entities successfully captured and linked.",
                    "In-KB F1": "Harmonic mean of In-KB Precision and Recall (Overall linking performance).",
                    "NIL Precision": "Accuracy: % of local unlinked entity clusters that are correctly grouped.",
                    "NIL Recall": "Coverage: % of actual unlinked entities successfully grouped together.",
                    "NIL F1": "Harmonic mean of NIL Precision and Recall (Overall clustering performance)."
                }
                
                df_matrix = df_benchmarks.set_index("Scope / Cohort").T.reset_index()
                
                df_matrix.rename(columns={
                    "index": "Metric",
                    "GLOBAL BASELINE (All Cohorts)": "Global Baseline",
                    "Cohort A (Racial/Ethnic Minorities)": "Cohort A",
                    "Cohort B (LGBTQIA+ Histories)": "Cohort B",
                    "Cohort C (Indigenous Populations)": "Cohort C"
                }, inplace=True)
                
                df_matrix.insert(1, "Description", df_matrix["Metric"].map(metric_descriptions))
                
                numeric_cols = ["Global Baseline", "Cohort A", "Cohort B", "Cohort C"]
                for col in numeric_cols:
                    if col in df_matrix.columns:
                        df_matrix[col] = df_matrix[col].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(
                    df_matrix,
                    use_container_width=True,
                    hide_index=True,
                    height=320
                )

            st.markdown("---")

            # 3. Section 2: Resolution Breakdown & NIL Cluster Deep-Dive
            res_col1, res_col2 = st.columns(2)

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

            st.markdown("---")

            # 4. Section 3: Model Confidence & Data Completeness
            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                st.markdown("#### NER Model Confidence Distribution")
                fig_conf = px.histogram(
                    df_filtered, 
                    x="Confidence", 
                    nbins=20, 
                    color_discrete_sequence=["#491D8B"]
                )
                fig_conf.update_layout(
                    yaxis_title="Entity Count", 
                    xaxis_title="Confidence Score (0.0 - 1.0)",
                    margin=dict(t=30, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_conf, use_container_width=True)

            with diag_col2:
                st.markdown("#### Semantic Metadata Fill Rate (%)")
                attributes = [
                    "Occupation", "Political Ideology", "Member Of", 
                    "Participant In", "Gender Identity", "Ethnic Group/Tribe", 
                    "Religion", "Country", "VIAF Link"
                ]
                completeness = [
                    (df_filtered[col].notna().sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0 
                    for col in attributes
                ]
                
                df_comp = pd.DataFrame({"Attribute": attributes, "Fill Rate (%)": completeness})
                fig_comp = px.bar(
                    df_comp, 
                    x="Fill Rate (%)", 
                    y="Attribute", 
                    orientation='h', 
                    color_discrete_sequence=["#198038"]
                )
                fig_comp.update_xaxes(range=[0, 100])
                fig_comp.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    margin=dict(t=30, b=10, l=10, r=10)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
# =========================================================================================================================================
# Tab 6: 
# =========================================================================================================================================
     with tab6:
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
