import re

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import colormaps as mcolormaps
import matplotlib.colors as mcolors
import altair as alt

st.set_page_config(page_title="DAM Results Heatmap", layout="wide")
st.title("📊 DAM Results — Ωριαία Ανάλυση")

uploaded_file = st.file_uploader("Ανέβασε το Excel αρχείο (.xlsx)", type=["xlsx"])

# Sheets like "Export GR-IT", "Import GR-BG", ... are per-country flow sheets,
# each in the same long format (market / delivery_ts / value) as the standard sheets.
COUNTRY_SHEET_RE = re.compile(r"^(export|import)\s+(?:gr-(\w+)|(\w+)-gr)$", re.IGNORECASE)

# All timestamps are shifted forward by this many hours before being displayed
# (e.g. to move from UTC to local delivery time).
TIMESTAMP_SHIFT_HOURS = 2


def build_hourly_pivot(
    df: pd.DataFrame,
    value_col: str = "value",
    sheet_name: str = "",
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    df = df.copy()
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"], errors="coerce") + pd.Timedelta(
        hours=TIMESTAMP_SHIFT_HOURS
    )
    df = df.dropna(subset=["delivery_ts"])

    df["date"] = df["delivery_ts"].dt.date
    df["hour"] = df["delivery_ts"].dt.hour + 1  # 1..24

    if df.empty:
        pivot = pd.DataFrame(columns=list(range(1, 25)), dtype=float)
        pivot.index.name = "date"
    else:
        pivot = df.pivot_table(
            index="date", columns="hour", values=value_col, aggfunc="mean"
        )

    for h in range(1, 25):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[list(range(1, 25))]

    # Όλες οι μέρες του εύρους, ακόμα κι αν δεν έχουν καμία γραμμή στα δεδομένα
    if start_date is not None and end_date is not None:
        all_days = [d.date() for d in pd.date_range(start_date, end_date)]
        pivot = pivot.reindex(all_days)
        pivot.index.name = "date"

    # Ό,τι λείπει (ώρα ή ολόκληρη μέρα) = 0
    pivot = pivot.fillna(0)

    pivot["SUM"] = pivot[list(range(1, 25))].sum(axis=1)
    pivot["AVG"] = pivot[list(range(1, 25))].mean(axis=1)

    if "mcp" in (sheet_name or "").lower():
        pivot["SUM"] = np.nan

    pivot.insert(pivot.columns.get_loc("SUM"), "", np.nan)

    pivot = pivot.sort_index(ascending=False)
    return pivot

def style_pivot_table(pivot: pd.DataFrame, hour_cols: list):
    """Per-row (per-day) background coloring for the hourly columns: for a
    given day, the lowest hour is green and the highest hour is red,
    independent of what happens on other days. SUM and AVG keep their own
    per-column scale (so those totals are still comparable day to day). 0
    is scaled like any other real value and is never painted black."""
    cmap = mcolormaps["RdYlGn_r"]  # low -> green, high -> red

    def _styles_for(values):
        vals = np.asarray(values, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
            if vmin == vmax:
                vmax = vmin + 1.0  # avoid a degenerate (constant) range

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        styles = []
        for v in values:
            if pd.isna(v):
                styles.append("background-color: transparent;")
            else:
                rgba = cmap(norm(float(v)))
                hex_color = mcolors.to_hex(rgba)
                r, g, b = mcolors.to_rgb(hex_color)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.55 else "white"
                styles.append(f"background-color: {hex_color}; color: {text_color};")
        return styles

    def colorize_row(row: pd.Series):
        return _styles_for(row.to_numpy())

    def colorize_col(col: pd.Series):
        return _styles_for(col.to_numpy())

    styled = (
        pivot.style
        .apply(colorize_row, subset=hour_cols, axis=1)
        .apply(colorize_col, subset=["SUM"], axis=0)
        .apply(colorize_col, subset=["AVG"], axis=0)
        .format(precision=0, na_rep="")
    )
    return styled


def date_range_picker(min_date, max_date, default_range,key):
    date_range = st.date_input(
        "Επίλεξε εύρος ημερομηνιών:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=key,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        return date_range
    return min_date, max_date
def render_implicit_chart(raw_df, sheet_name, start_date, end_date, key_prefix="std"):
    if not {"delivery_ts", "value"}.issubset(raw_df.columns):
        st.error(
            f"Το sheet '{sheet_name}' δεν έχει τις απαραίτητες "
            f"στήλες 'delivery_ts' και 'value'."
        )
        return

    # Ίδια λογική με τα υπόλοιπα sheets (shift ώρας, μηδενισμός κενών ημερών)
    pivot = build_hourly_pivot(
        raw_df,
        sheet_name=sheet_name,
        start_date=start_date,
        end_date=end_date,
    )
    pivot = pivot[(pivot.index >= start_date) & (pivot.index <= end_date)]

    st.subheader(sheet_name)

    if pivot.empty:
        st.info("Δεν υπάρχουν δεδομένα για το επιλεγμένο διάστημα.")
        return

    metric = st.radio(
        "Τιμή ανά ημέρα:",
        ["SUM", "AVG"],
        horizontal=True,
        key=f"metric_{key_prefix}_{sheet_name}",
    )

    chart_df = (
        pivot[[metric]]
        .rename(columns={metric: "value"})
        .sort_index()
        .reset_index()
    )
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df["sign"] = np.where(chart_df["value"] >= 0, "Θετικό", "Αρνητικό")
    chart_df["label"] = chart_df["date"].dt.strftime("%d/%m")
    day_order = chart_df["label"].tolist()  # ήδη ταξινομημένο χρονολογικά

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "label:O",
                title="Ημερομηνία",
                sort=day_order,
                axis=alt.Axis(labelAngle=-45),
                scale=alt.Scale(paddingInner=0.15, paddingOuter=0.05),
            ),
            y=alt.Y("value:Q", title=metric),
            color=alt.Color(
                "sign:N",
                scale=alt.Scale(
                    domain=["Θετικό", "Αρνητικό"],
                    range=["#2e7d32", "#c62828"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ημερομηνία", format="%d/%m/%Y"),
                alt.Tooltip("value:Q", title=metric, format=",.0f"),
            ],
        )
        .properties(height=400)
    )

    # Γραμμή του μηδενός για να φαίνεται ο οριζόντιος άξονας
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")

    st.altair_chart(chart + zero_line, use_container_width=True)

    st.download_button(
        label=f"⬇️ Κατέβασε τον πίνακα ({sheet_name}) ως CSV",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}_{sheet_name}",
    )
def render_standard_sheet(
    raw_df: pd.DataFrame,
    sheet_name: str,
    start_date,
    end_date, 
    key_prefix: str = "std"
):


    if "implicit" in sheet_name.lower():
        render_implicit_chart(raw_df, sheet_name, start_date, end_date, key_prefix)
        return

    
    if not {"delivery_ts", "value"}.issubset(raw_df.columns):
        st.error(
            f"Το sheet '{sheet_name}' δεν έχει τις απαραίτητες "
            f"στήλες 'delivery_ts' και 'value'."
        )
        return

    pivot_full = build_hourly_pivot(
        raw_df,
        sheet_name=sheet_name,
        start_date=start_date,
        end_date=end_date,
    )

    pivot = pivot_full  # ήδη περιορισμένο στο εύρος από το reindex


    # Χρησιμοποιούμε το ΙΔΙΟ date range για όλα τα sheets
    pivot = pivot_full[
        (pivot_full.index >= start_date) &
        (pivot_full.index <= end_date)
    ]

    st.subheader(sheet_name)

    if pivot.empty:
        st.info("Δεν υπάρχουν δεδομένα για το επιλεγμένο διάστημα.")
        return

    hour_cols = list(range(1, 25))

    styled = style_pivot_table(
        pivot,
        hour_cols
    )

    table_height = min(35 * (len(pivot) + 1) + 3, 900)

    st.dataframe(
        styled,
        use_container_width=True,
        height=table_height,
    )
    st.download_button(
        label=f"⬇️ Κατέβασε τον πίνακα ({sheet_name}) ως CSV",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}_{sheet_name}",
    )


if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    all_dates = []

    for sheet in sheet_names:
        try:
            temp_df = pd.read_excel(
                xls,
                sheet_name=sheet,
                usecols=["delivery_ts"]
            )

            temp_dates = (
                pd.to_datetime(
                    temp_df["delivery_ts"],
                    errors="coerce"
                )
                + pd.Timedelta(hours=TIMESTAMP_SHIFT_HOURS)
            )

            temp_dates = temp_dates.dropna()

            if not temp_dates.empty:
                all_dates.extend(
                    temp_dates.dt.date.tolist()
                )

        except Exception:
            pass
    if not all_dates:
        st.error("Δεν βρέθηκαν έγκυρες ημερομηνίες στο αρχείο.")
        st.stop()

    # Βρίσκουμε τις μοναδικές ημερομηνίες ταξινομημένες
    unique_dates = sorted(list(set(all_dates)))
    global_min_date = unique_dates[0]
    global_max_date = unique_dates[-1]

    # Υπολογισμός προεπιλογής: οι 10 πιο πρόσφατες ημέρες
    default_start_date = unique_dates[-10] if len(unique_dates) >= 10 else global_min_date
    default_end_date = global_max_date

    st.markdown("### 📅 Περίοδος εμφάνισης")

    # Αν δεν υπάρχει ήδη αποθηκευμένη τιμή στο session, αρχικοποιούμε με τις 10 τελευταίες ημέρες
    if "global_date_range" not in st.session_state:
        st.session_state["global_date_range"] = (default_start_date, default_end_date)

    date_range = st.date_input(
        "Επίλεξε εύρος ημερομηνιών:",
        value=st.session_state["global_date_range"],
        min_value=global_min_date,
        max_value=global_max_date,
        key="global_date_range",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = global_min_date, global_max_date

    st.markdown("---")

    # Detect per-country Export/Import sheets, e.g. "Export GR-IT", "Import GR-BG"
    country_sheet_matches = {}  # sheet_name -> (direction, country_code)
    for s in sheet_names:
        m = COUNTRY_SHEET_RE.match(s.strip())
        if m:
            direction = m.group(1).capitalize()
            country = (m.group(2) or m.group(3)).upper()
            country_sheet_matches[s] = (direction, country)
        elif s.strip().lower().startswith("net "):
            country_sheet_matches[s] = ("Net", s.strip()[4:])

    standard_sheets = [s for s in sheet_names if s not in country_sheet_matches]

    tab1, tab2 = st.tabs(["📁 Dam Results", "🌍 Imports / Exports"])

    with tab1:
        if sheet_names:
            default_selection = standard_sheets
            selected_sheets = st.multiselect(
                "Επίλεξε κατηγορίες (μπορείς πάνω από μία):",
                standard_sheets,
                default=default_selection,
                key="select_standard",
            )

            if not selected_sheets:
                st.info("Επίλεξε τουλάχιστον μία κατηγορία για να εμφανιστούν δεδομένα.")

            for sheet in selected_sheets:
                raw_df = pd.read_excel(xls, sheet_name=sheet)
                render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="tab1")
                st.markdown("---")
                st.markdown("---")
        else:
            st.info("Δεν βρέθηκαν sheets δεδομένων.")

    with tab2:
        net_sheets = sorted([
                                    s for s in sheet_names
                                    if ("net" in s.lower() or "total" in s.lower())
                                    and "residual" not in s.lower()
                                ])
        import_sheets = sorted([s for s, (d, _c) in country_sheet_matches.items() if d == "Import"])
        export_sheets = sorted([s for s, (d, _c) in country_sheet_matches.items() if d == "Export"])

        if not net_sheets and not import_sheets and not export_sheets:
            st.info("Δεν βρέθηκαν σχετικά sheets για Imports / Exports / Nets στο αρχείο.")
        else:
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📊 Net & Totals", "📥 Αναλυτικές Εισαγωγές", "📤 Αναλυτικές Εξαγωγές"])

            with sub_tab1:
                st.markdown("#### 🔄 Καθαρές Ροές (Net) & Total Imports / Exports")
                if net_sheets:
                    selected_nets = st.multiselect(
                        "Επίλεξε Net / Total sheets:",
                        net_sheets,
                        default=net_sheets,
                        key="select_nets",
                    )
                    for sheet in selected_nets:
                        raw_df = pd.read_excel(xls, sheet_name=sheet)
                        render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="net")
                        st.markdown("---")
                else:
                    st.info("Δεν βρέθηκαν sheets τύπου 'Net ...' ή 'Total ...'.")

            with sub_tab2:
                st.markdown("#### 📥 Αναλυτικές Εισαγωγές ανά χώρα (Import XX-GR)")
                if import_sheets:
                    selected_imports = st.multiselect(
                        "Επίλεξε χώρες εισαγωγών:",
                        import_sheets,
                        default=import_sheets,
                        key="select_imports",
                    )
                    for sheet in selected_imports:
                        raw_df = pd.read_excel(xls, sheet_name=sheet)
                        # ΔΙΟΡΘΩΣΗ ΕΔΩ: Χρήση key_prefix αντί για end_datekey_prefix
                        render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="imp")
                        st.markdown("---")
                else:
                    st.info("Δεν βρέθηκαν αναλυτικά sheets εισαγωγών.")

            with sub_tab3:
                st.markdown("#### 📤 Αναλυτικές Εξαγωγές ανά χώρα (Export GR-XX)")
                if export_sheets:
                    selected_exports = st.multiselect(
                        "Επίλεξε χώρες εξαγωγών:",
                        export_sheets,
                        default=export_sheets,
                        key="select_exports",
                    )
                    for sheet in selected_exports:
                        raw_df = pd.read_excel(xls, sheet_name=sheet)
                        render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="exp")
                        st.markdown("---")
                else:
                    st.info("Δεν βρέθηκαν αναλυτικά sheets εξαγωγών.")
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
