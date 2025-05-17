import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import time
import faiss
import os
import base64
from dotenv import load_dotenv
from fpdf import FPDF

# OpenAI API-Key über .env laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Fehlender OpenAI API-Key. Bitte `.env` Datei erstellen mit `OPENAI_API_KEY=...`")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Header: logo and title centered
logo_path = os.path.join(os.path.dirname(__file__), "data", "mindra_logo.png")

st.set_page_config(
    layout="wide",
    page_title="MINDRA – Interview Knowledge Base AI-Chat",
    page_icon=logo_path,
)

# Logo und Titel zentriert
if os.path.isfile(logo_path):
    # Encode logo for inline HTML
    logo_bytes = open(logo_path, "rb").read()
    logo_b64 = base64.b64encode(logo_bytes).decode()
    html_header = f"""
    <div style="text-align:center; margin-bottom:30px;">
      <img src="data:image/png;base64,{logo_b64}" width="160" style="margin-bottom:0px;" />
      <h1 font-size:2rem;">MINDRA – Interview Knowledge Base AI-Chat</h1>
    </div>
    """
    st.markdown(html_header, unsafe_allow_html=True)
else:
    st.warning("Logo 'mindra_logo.png' nicht gefunden.")
st.divider()

# Einleitungstext in zwei Spalten aufteilen
st.markdown(
    """
**Willkommen bei MINDRA**
– Dein interaktives Projekt-Dashboard und zentrale Kommunikationsplattform für alle Stakeholder. Hier kannst du qualitative Interviewdaten analysieren, explorieren und interaktiv abfragen. Nutze die verschiedenen Funktionen, um tiefere Einblicke in die Daten zu gewinnen und fundierte Entscheidungen zu treffen.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
**App-Funktionen**
- Qualitativ kodierte Interviewdaten explorieren, analysieren und abfragen
- Paralleler Zugriff auf die Excel-Quellen oder direkt interaktiv im Browser
- Einfache Filterung und Visualisierung der Daten

**Datenübersicht**
- 8 Cluster qualitativer Interviews aus relevanten Experten-Gesprächen
- Über 1'500 sauber eingeordnete Aussagen mit jeweils 7 zugeteilten Attributen (Rolle, Firma, Typ, Cluster, Beschreibung, Aussage, NR)
- Saubere Daten und sauberes Training der A.I.
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
**Interaktivität**
- Ausklappbare Filter- und Verteilungs-Abschnitte in Echtzeit
- Filter nach Cluster, Typ, Datei und Stichwortsuche
- Interaktive Balkendiagramme für Cluster- und Typ-Verteilung

**Chat-Modi**
- *Deskriptiv*: Prägnante Zusammenfassungen der ausgewählten Daten
- *Interpretierend*: Detaillierte Erklärungen, Analysen und konkrete Empfehlungen
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Eingabe: Frage stellen und Modus wählen
st.subheader("Frage stellen")
query = st.text_area("Frage an die Interview-Daten:", "")
answer_mode = st.radio(
    "Antwortmodus", ("Nur deskriptiv (zusammenfassend)", "Interpretierend (mit Empfehlungen)")
)


# OpenAI API-Key über .env laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Fehlender OpenAI API-Key. Bitte `.env` Datei erstellen mit `OPENAI_API_KEY=...`")
    st.stop()

# OpenAI Python v1 Client initialisieren
client = OpenAI(api_key=openai_api_key)

# Daten einlesen und bereinigen
# Spalten: NR, Rolle, Firma, Typ, Cluster, Beschreibung, Aussage, Quelle (jede Zeile = eine Aussage).
@st.cache_data(show_spinner=False)
def load_and_clean_data():
    # Track original columns for each file
    original_columns = {}
    files = {
        "(1) Use Cases": "file1.csv",
        "(2.1) Nutzen": "file2.1.csv",
        "(2.2) Hard-Savings": "file2.2.csv",
        "(3.1) Aufwand Qualitativ": "file3.1.csv",
        "(3.2) Aufwand Quantitativ": "file3.2.csv",
        "(4.1) Herausforderungen": "file4.1.csv",
        "(4.2) Risiken": "file4.2.csv",
        "(5) Best Practices": "file5.csv"
    }
    data_frames = {}
    for label, filename in files.items():
        try:
            # CSV einlesen; dtype=str sorgt dafür, dass alles als Text gelesen wird
            data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
            df = pd.read_csv(os.path.join(data_path, filename), dtype=str, keep_default_na=False, engine='python')
            # Record the raw column names before cleaning
            original_columns[label] = df.columns.tolist()
        except Exception as e:
            st.error(f"Fehler beim Lesen von {filename}: {e}")
            continue
        # Whitespace an beiden Enden in allen Zellen entfernen
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # Dateispezifische Bereinigungen:
        if label == "file2.1.csv":
            # "Connected Use Case" und "EDM oder EDC?" in Beschreibung integrieren
            usecase_col = "Connected Use Case (wie wurde das erreicht?) "
            edm_col = "EDM oder EDC? "
            if usecase_col in df.columns and edm_col in df.columns:
                new_besch = []
                for _, row in df.iterrows():
                    besch = row.get("Beschreibung", "")
                    usecase = row.get(usecase_col, "")
                    edm_edc = row.get(edm_col, "")
                    parts = []
                    if usecase:
                        parts.append(f"Use Case: {usecase}")
                    if edm_edc:
                        parts.append(f"Kontext: {edm_edc}")
                    extra = " | ".join(parts) if parts else ""
                    # Bestehende Beschreibung mit Zusatzinfos verknüpfen
                    if besch and extra:
                        new_besch.append(f"{besch} | {extra}")
                    elif besch:
                        new_besch.append(besch)
                    else:
                        new_besch.append(extra)
                df["Beschreibung"] = new_besch
                # Entferne die zwei zusätzlichen Spalten
                df.drop(columns=[usecase_col, edm_col], inplace=True)
        if label == "file2.2.csv":
            # Spezielle Spalten (Einsparungen etc.) in Cluster/Beschreibung integrieren
            # Unbenannte letzte Spalte als "Quelle" interpretieren, falls vorhanden
            for col in list(df.columns):
                if col.startswith("Unnamed"):
                    df.rename(columns={col: "Quelle"}, inplace=True)
            if "Quelle" not in df.columns:
                df["Quelle"] = ""
            # Werte aus "Bereiche...", "konkrete Einsparungen", "Skala", "Allg. Aussagen" verarbeiten
            area_col = "Bereiche in die eingespart wurde"
            savings_col = "konkrete Einsparungen / Zahlen"
            scale_col = "Skala Kosteneinsparungen 1–5"
            measure_col = "Allgemeine Aussagen zur Messung"
            if area_col in df.columns:
                new_cluster = []
                new_desc = []
                for _, row in df.iterrows():
                    area = row.get(area_col, "")
                    concrete = row.get(savings_col, "")
                    scale = row.get(scale_col, "")
                    measure = row.get(measure_col, "")
                    # Cluster: benutze Bereich, falls vorhanden
                    cluster_val = area if area else ""
                    new_cluster.append(cluster_val)
                    # Beschreibung aus restlichen Feldern zusammenbauen
                    parts = []
                    if concrete:
                        parts.append(f"Einsparung: {concrete}")
                    if scale:
                        parts.append(f"Skala: {scale}")
                    if measure:
                        parts.append(f"Messung: {measure}")
                    new_desc.append(" | ".join(parts) if parts else "")
                df["Cluster"] = new_cluster
                df["Beschreibung"] = new_desc
                # Alte Spalten entfernen
                drop_cols = [area_col, savings_col, scale_col, measure_col]
                for c in drop_cols:
                    if c in df.columns:
                        df.drop(columns=c, inplace=True)
        if label == "file3.2.csv":
            # "Dauer min." und "Dauer max." Spalten entfernen (Info bereits in Beschreibung enthalten)
            if "Dauer min." in df.columns:
                df.drop(columns=["Dauer min."], inplace=True)
            if "Dauer max." in df.columns:
                df.drop(columns=["Dauer max."], inplace=True)
            # Unbenannte leere Spalten entfernen
            unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
            if unnamed_cols:
                df.drop(columns=unnamed_cols, inplace=True)
        if label in ["file4.csv", "file5.csv"]:
            # Inhalte aus evtl. unbenannten Spalten mit Beschreibung zusammenführen (falls durch Kommas versprengt)
            unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
            if unnamed_cols:
                new_besch = []
                for _, row in df.iterrows():
                    besch = row.get("Beschreibung", "")
                    extra_parts = [str(row.get(col, "")).strip() for col in unnamed_cols if row.get(col, "")]
                    extra_parts = [part for part in extra_parts if part and part.lower() != 'nan']
                    if extra_parts:
                        extra_text = ", ".join(extra_parts)
                        new_besch.append(f"{besch}, {extra_text}" if besch else extra_text)
                    else:
                        new_besch.append(besch)
                df["Beschreibung"] = new_besch
                df.drop(columns=unnamed_cols, inplace=True)
        # Stelle sicher, dass "Quelle" Spalte existiert
        if "Quelle" not in df.columns:
            df["Quelle"] = ""
        # Spalten in gewünschter Reihenfolge anordnen
        cols = ["NR", "Rolle", "Firma", "Typ", "Cluster", "Beschreibung", "Aussage", "Quelle"]
        df = df.reindex(columns=cols)
        # Originale Zeilennummer im Ursprungs-CSV speichern (1-basierter Index inkl. Header)
        df["Line"] = df.index + 2
        data_frames[label] = df.fillna("")
    # Alle Teil-Datensätze zusammenführen und Dateiname als Spalte hinzufügen
    combined = pd.DataFrame()
    for label, df in data_frames.items():
        df = df.copy()
        df["Datei"] = label
        combined = pd.concat([combined, df], ignore_index=True)
    return combined, original_columns


data_df, column_info = load_and_clean_data()

# Filter und Verteilung in expandierbaren Spalten
col1, col2 = st.columns(2)
with col1:
    with st.expander("Filter", expanded=False):
        # Quelle (Auswertungs-Masterfile) auswählen
        file_options = sorted(data_df["Datei"].unique())
        selected_files = st.multiselect("Quelle (Auswertungs-Masterfile) auswählen", file_options, default=file_options)

        # Nur die Spalte "Cluster" basierend auf ausgewählten Dateien anzeigen
        cluster_options = []
        for label in selected_files:
            clusters = [c for c in data_df.loc[data_df["Datei"] == label, "Cluster"].unique() if c]
            for c in sorted(clusters):
                cluster_options.append(f"{label}: {c}")
        selected_clusters = st.multiselect("Cluster auswählen", cluster_options)

        # Interview Typ auswählen
        type_options = sorted({t for t in data_df["Typ"] if t})
        selected_types = st.multiselect("Interview Typ auswählen", type_options)

        # Stichwortsuche
        keyword = st.text_input("Stichwortsuche in Aussagen/Beschreibung")

        # Filter auf DataFrame anwenden
        filtered_df = data_df.copy()
        if selected_files:
            filtered_df = filtered_df[filtered_df["Datei"].isin(selected_files)]
        if selected_clusters:
            mask = pd.Series(False, index=filtered_df.index)
            for sel in selected_clusters:
                file_label, cluster_val = sel.split(": ", 1)
                mask |= (filtered_df["Datei"] == file_label) & (filtered_df["Cluster"] == cluster_val)
            filtered_df = filtered_df[mask]
        if selected_types:
            filtered_df = filtered_df[filtered_df["Typ"].isin(selected_types)]
        if keyword:
            kw = keyword.lower()
            mask_kw = (
                filtered_df["Aussage"].str.lower().str.contains(kw)
                | filtered_df["Beschreibung"].str.lower().str.contains(kw)
            )
            filtered_df = filtered_df[mask_kw]
with col2:
    with st.expander("Datenverteilung", expanded=False):
        if not filtered_df.empty:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.markdown("**Cluster-Verteilung (Anzahl Aussagen)**")
                # Nur die Spalte "Cluster" verwenden
                cluster_counts = filtered_df["Cluster"].value_counts()
                st.bar_chart(cluster_counts)
            with subcol2:
                st.markdown("**Typ-Verteilung (Anzahl Aussagen)**")
                # Nur tatsächlich vorhandene Typen verwenden (leere Einträge ausblenden)
                type_counts = filtered_df.loc[filtered_df["Typ"].str.strip() != "", "Typ"].value_counts()
                st.bar_chart(type_counts)
        else:
            st.write("Keine Daten für ausgewählte Filter.")


# 5. Vorbereitung semantische Suche (FAISS Index mit OpenAI Embeddings)
# ---------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def prepare_faiss_index(df: pd.DataFrame):
    texts = []
    metadata = []
    for _, row in df.iterrows():
        # Erstelle zu durchsuchenden Text (kombiniere relevante Felder für bessere Trefferquote)
        text = f"{row['Rolle']} {row['Firma']} {row['Typ']} {row['Cluster']} {row['Beschreibung']} {row['Aussage']}"
        texts.append(text)
        metadata.append((row["Datei"], row["Line"]))
    # Embeddings in Batches abrufen (Model: text-embedding-ada-002)
    batch_size = 100
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(model="text-embedding-ada-002", input=batch_texts)
        batch_embs = [np.array(e.embedding, dtype="float32") for e in response.data]
        embeddings_list.extend(batch_embs)
    embeddings_matrix = np.vstack(embeddings_list)
    # FAISS Index mit Kosinus-Ähnlichkeit (via inner product auf normalisierten Vektoren)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_normed = embeddings_matrix / (norms + 1e-10)
    index = faiss.IndexFlatIP(embeddings_normed.shape[1])
    index.add(embeddings_normed)
    return index, metadata, embeddings_normed

# 6. Beantwortung der Frage mittels GPT-4
# ---------------------------------------------------------------------------------------------------------------------------------------------
if st.button("Antwort generieren"):
    # Sicherstellen, dass Voraussetzungen erfüllt sind
    if not openai_api_key:
        st.error("Bitte zuerst einen gültigen OpenAI API-Key eingeben.")
    elif not query.strip():
        st.error("Bitte eine Frage eingeben.")
    elif filtered_df.empty:
        st.warning("Für die gewählten Filter sind keine Daten vorhanden.")
    else:
        # Column info description for GPT
        columns_description = "Spalten in den Quelldateien:\n"
        for fname, cols in column_info.items():
            columns_description += f"- {fname}: {', '.join(cols)}\n"
        # Fortschrittsbalken und Statusanzeige initialisieren
        progress_bar = st.progress(10)
        status_text = st.info("🤖 AI analysiert die Daten und bereitet die Antwort vor...")
        # FAISS Index erstellen (oder aus Cache laden)
        faiss_index, meta_list, embed_matrix = prepare_faiss_index(filtered_df)
        progress_bar.progress(25)
        if faiss_index is None:
            st.stop()
        # Anfrage-Embedding erzeugen
        try:
            q_response = client.embeddings.create(model="text-embedding-ada-002", input=[query])
            progress_bar.progress(40)
            q_emb = q_response.data[0].embedding
        except Exception as e:
            st.error(f"Fehler bei der Embedding-Erstellung der Frage: {e}")
            st.stop()
        q_vec = np.array(q_emb, dtype="float32")
        # Query-Vektor normalisieren
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        # Dynamische Obergrenze für Top-Dokumente basierend auf verfügbarer Datenmenge
        max_k = 5  # Maximale Anzahl der Top-Dokumente
        num_candidates = min(max_k, len(filtered_df))
        # Nur suchen, wenn es mindestens ein Dokument gibt
        if num_candidates > 0:
            D, I = faiss_index.search(q_vec.reshape(1, -1), num_candidates)
            raw_indices = I[0]
        else:
            raw_indices = []
        # Nur eindeutige Indizes behalten und auf num_candidates begrenzen
        unique_indices = []
        for idx in raw_indices:
            if idx < len(filtered_df) and idx not in unique_indices:
                unique_indices.append(idx)
                if len(unique_indices) >= num_candidates:
                    break
        top_indices = unique_indices
        # Kontext aus den Top-Ergebnissen zusammenstellen
        context_snippets = []
        references = []
        for idx in top_indices:
            if idx < len(filtered_df):
                row = filtered_df.iloc[idx]
                role = row["Rolle"]; company = row["Firma"]; typ = row["Typ"]
                statement = row["Aussage"]
                source_file = row["Datei"]; source_line = row["Line"]
                references.append((source_file, source_line, role, company, typ, statement))
                context_snippets.append(f"[{source_file} Zeile {source_line}] {role} @ {company} ({typ}): {statement}")
        # GPT-4 Prompt erstellen
        system_prompt = (
            "Du bist ein Assistent, der Fragen anhand von Interview-Aussagen beantwortet.\n"
        )
        if answer_mode.startswith("Nur deskriptiv"):
            system_prompt += "Gib eine zusammenfassende, deskriptive Antwort basierend auf den bereitgestellten Aussagen. "
            system_prompt += "Füge keine eigenen Interpretationen oder Empfehlungen hinzu.\n"
        else:
            system_prompt += "Gib eine interpretierende Antwort mit möglichen Erklärungen oder Empfehlungen basierend auf den Aussagen. "
            system_prompt += "Du darfst Schlussfolgerungen ziehen, aber stütze dich auf die Inhalte der Aussagen.\n"
        system_prompt += "Zitiere relevante Quellen in der Form (Datei, Zeile) für jede wichtige Aussage in deiner Antwort."
        # Kontext der Aussagen als zusätzliche System-Nachricht
        context_text = "Relevante Aussagen:\n" + "\n".join(context_snippets)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": columns_description},
            {"role": "system", "content": context_text},
            {"role": "user", "content": query.strip()}
        ]
        # GPT-4 Anfrage ausführen
        try:
            completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages, temperature=0.2, max_tokens=4000)
        except Exception as e:
            st.error(f"Fehler bei der GPT-4 Anfrage: {e}")
            st.stop()
        answer_text = completion.choices[0].message.content
        progress_bar.progress(100)
        status_text.success("✅ Antwort ist bereit")
        # Antwort anzeigen
        st.subheader("Antwort")
        st.write(answer_text)
        # Quellen (die herangezogenen Aussagen) anzeigen im gewünschten Layout
        st.subheader("Quellen aus den Interviews")
        for (src_file, src_line, role, company, typ, statement) in references:
            st.markdown(f"**{role}, {company} ({typ})**")  # Rolle, Firma, Typ oben
            quote_text = "> " + statement.replace("\n", "\n> ")
            st.markdown(quote_text)  # Aussage im Blockquote (gepolstertes Textfeld)
            st.markdown(f"*Quelle: {src_file}, Zeile {src_line}*")  # Quelle am Ende
            st.divider()  # Trennlinie zwischen den Quellen
        # 7. Export-Funktion: Antwort und Quellen als Markdown oder CSV
        # ------------------------------------------------------------
        output_md = f"**Frage:** {query}\n\n**Antwort:**\n{answer_text}\n\n**Quellen:**\n"
        for (src_file, src_line, role, company, typ, statement) in references:
            output_md += f"- **{role}, {company} ({typ})** – {statement} *(Quelle: {src_file}, Zeile {src_line})*\n"
        output_csv_df = pd.DataFrame([{
            "Frage": query,
            "Antwort": answer_text,
            "Quellen": "; ".join([f"{sfile} Zeile {sline}" for (sfile, sline, *_ ) in references])
        }])
        csv_data = output_csv_df.to_csv(index=False).encode("utf-8")
        st.download_button("Antwort und Quellen als Markdown herunterladen", output_md, file_name="antwort_und_quellen.md")
        st.download_button("Antwort und Quellen als CSV herunterladen", csv_data, file_name="antwort_und_quellen.csv")

        # PDF-Export der Antwort und Quellen
        def create_pdf_bytes(query, answer_text, references):
            def sanitize(text):
                # Entfernt Zeichen außerhalb des Latin-1 Bereichs
                return ''.join(ch if ord(ch) < 256 else ' ' for ch in text)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Dokument-Header
            pdf.set_font("Arial", 'I', 8)
            pdf.cell(0, 6, sanitize("Antworten generiert durch MINDRA – basierend auf Interviewdaten"), 0, 1, 'C')
            pdf.ln(4)

            # Frage-Section (fett)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, sanitize(f"Frage: {query}"))
            pdf.ln(2)

            # Antwort-Section (fett)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, "Zusammenfassung:")
            # Antworttext kursiv darstellen
            pdf.set_font("Arial", 'I', 10)
            pdf.multi_cell(0, 8, sanitize(answer_text))
            pdf.ln(3)
            # Seite für Quellen beginnen
            pdf.add_page()

            # Quellen-Section (fett)
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, "Quellen:")
            pdf.ln(1)

            # Jede Quelle schön abgetrennt und gestylt
            for src_file, src_line, role, company, typ, statement in references:
                # Rollen/Unternehmen als fette Überschrift
                pdf.set_font("Arial", 'B', 10)
                pdf.multi_cell(0, 6, sanitize(f"{role}, {company} ({typ})"))

                # Aussage kursiv eingerückt
                pdf.set_font("Arial", 'I', 9)
                pdf.multi_cell(0, 6, "  " + sanitize(f"\"{statement}\""))

                # Quelle klein und normal
                pdf.set_font("Arial", size=8)
                pdf.multi_cell(0, 5, sanitize(f"Quelle: {src_file}, Zeile {src_line}"))

                # Abstand oberhalb der Trennlinie
                pdf.ln(4)
                # Horizontale Trennlinie
                y = pdf.get_y()
                pdf.set_draw_color(0, 0, 0)
                pdf.set_line_width(0.2)
                pdf.line(10, y, pdf.w - 10, y)
                # Abstand unterhalb der Trennlinie
                pdf.ln(4)
            return pdf.output(dest="S").encode("latin-1")
        try:
            pdf_bytes = create_pdf_bytes(query, answer_text, references)
            st.download_button(
                "Antwort und Quellen als PDF herunterladen",
                pdf_bytes,
                file_name="antwort_und_quellen.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF-Export fehlgeschlagen: {e}")
