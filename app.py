import re

import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import colormaps as mcolormaps
import matplotlib.colors as mcolors

st.set_page_config(page_title="DAM Results Heatmap", layout="wide")
st.title("📊 DAM Results — Ωριαία Ανάλυση")

uploaded_file = st.file_uploader("Ανέβασε το Excel αρχείο (.xlsx)", type=["xlsx"])

# Sheets like "Export GR-IT", "Import GR-BG", ... are per-country flow sheets,
# each in the same long format (market / delivery_ts / value) as the standard sheets.
COUNTRY_SHEET_RE = re.compile(r"^(export|import)\s+gr-(\w+)$", re.IGNORECASE)

# All timestamps are shifted forward by this many hours before being displayed
# (e.g. to move from UTC to local delivery time).
TIMESTAMP_SHIFT_HOURS = 2


def build_hourly_pivot(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Pivot table: γραμμές = ημερομηνία, στήλες = ώρα (1-24), + SUM/AVG.
    Timestamps are shifted by TIMESTAMP_SHIFT_HOURS before pivoting, and any
    hour with no data is treated as 0 (not left blank/NaN)."""
    df = df.copy()
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"], errors="coerce") + pd.Timedelta(
        hours=TIMESTAMP_SHIFT_HOURS
    )
    df = df.dropna(subset=["delivery_ts"])

    df["date"] = df["delivery_ts"].dt.date
    df["hour"] = df["delivery_ts"].dt.hour + 1  # 1..24

    pivot = df.pivot_table(
        index="date", columns="hour", values=value_col, aggfunc="mean"
    )

    for h in range(1, 25):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[[h for h in range(1, 25)]]

    # Any missing quarter/hour is treated as 0, not left blank.
    pivot[list(range(1, 25))] = pivot[list(range(1, 25))].fillna(0)

    pivot["SUM"] = pivot[range(1, 25)].sum(axis=1)
    pivot["AVG"] = pivot[range(1, 25)].mean(axis=1)
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
                styles.append("background-color: #eeeeee; color: #999999;")
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
        .format(precision=0, na_rep="0")
    )
    return styled


def date_range_picker(min_date, max_date, key):
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


def render_standard_sheet(raw_df: pd.DataFrame, sheet_name: str):
    if not {"delivery_ts", "value"}.issubset(raw_df.columns):
        st.error("Το sheet αυτό δεν έχει στήλες 'delivery_ts' και 'value'.")
        return

    pivot_full = build_hourly_pivot(raw_df)
    min_date, max_date = pivot_full.index.min(), pivot_full.index.max()
    start_date, end_date = date_range_picker(min_date, max_date, key=f"dr_{sheet_name}")

    pivot = pivot_full[(pivot_full.index >= start_date) & (pivot_full.index <= end_date)]

    st.subheader(f"{sheet_name} — Πίνακας ανά ημέρα / ώρα")

    hour_cols = list(range(1, 25))
    styled = style_pivot_table(pivot, hour_cols)
    st.dataframe(styled, width="stretch", height=550)

    st.download_button(
        label=f"⬇️ Κατέβασε τον πίνακα ({sheet_name}) ως CSV",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{sheet_name}",
    )


if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    # Detect per-country Export/Import sheets, e.g. "Export GR-IT", "Import GR-BG"
    country_sheet_matches = {}  # sheet_name -> (direction, country_code)
    for s in sheet_names:
        m = COUNTRY_SHEET_RE.match(s.strip())
        if m:
            direction = m.group(1).capitalize()
            country = m.group(2).upper()
            country_sheet_matches[s] = (direction, country)

    tab1, tab2 = st.tabs(["📁 Όλα τα Δεδομένα", "🌍 Imports / Exports ανά Χώρα"])

    with tab1:
        if sheet_names:
            default_selection = [sheet_names[0]] if sheet_names else []
            selected_sheets = st.multiselect(
                "Επίλεξε κατηγορίες (μπορείς πάνω από μία):",
                sheet_names,
                default=default_selection,
                key="select_standard",
            )

            if not selected_sheets:
                st.info("Επίλεξε τουλάχιστον μία κατηγορία για να εμφανιστούν δεδομένα.")

            for sheet in selected_sheets:
                raw_df = pd.read_excel(xls, sheet_name=sheet)
                render_standard_sheet(raw_df, sheet)
                st.markdown("---")
                st.markdown("---")
        else:
            st.info("Δεν βρέθηκαν sheets δεδομένων.")

    with tab2:
        import_sheets = sorted(
            s for s, (d, _c) in country_sheet_matches.items() if d == "Import"
        )
        export_sheets = sorted(
            s for s, (d, _c) in country_sheet_matches.items() if d == "Export"
        )

        if not import_sheets and not export_sheets:
            st.info(
                "Δεν βρέθηκαν sheets τύπου 'Export GR-XX' / 'Import GR-XX' στο αρχείο."
            )
        else:
            st.markdown("#### 📥 Εισαγωγές (Imports) ανά χώρα")
            if import_sheets:
                selected_imports = st.multiselect(
                    "Επίλεξε χώρες εισαγωγών:",
                    import_sheets,
                    default=[],
                    key="select_imports",
                )
                for sheet in selected_imports:
                    raw_df = pd.read_excel(xls, sheet_name=sheet)
                    render_standard_sheet(raw_df, sheet)
                    st.markdown("---")
            else:
                st.info("Δεν βρέθηκαν sheets εισαγωγών ('Import GR-XX').")

            st.markdown("#### 📤 Εξαγωγές (Exports) ανά χώρα")
            if export_sheets:
                selected_exports = st.multiselect(
                    "Επίλεξε χώρες εξαγωγών:",
                    export_sheets,
                    default=[],
                    key="select_exports",
                )
                for sheet in selected_exports:
                    raw_df = pd.read_excel(xls, sheet_name=sheet)
                    render_standard_sheet(raw_df, sheet)
                    st.markdown("---")
            else:
                st.info("Δεν βρέθηκαν sheets εξαγωγών ('Export GR-XX').")
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
