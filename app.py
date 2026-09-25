import re

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="DAM Results Heatmap", layout="wide")
st.title("📊 DAM Results — Ωριαία Ανάλυση")

uploaded_file = st.file_uploader("Ανέβασε το Excel αρχείο (.xlsx)", type=["xlsx"])

# Sheets like "Export GR-IT", "Import GR-BG", ... are per-country flow sheets,
# each in the same long format (market / delivery_ts / value) as the standard sheets.
COUNTRY_SHEET_RE = re.compile(r"^(export|import)\s+gr-(\w+)$", re.IGNORECASE)


def build_hourly_pivot(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Pivot table: γραμμές = ημερομηνία, στήλες = ώρα (1-24), + SUM/AVG."""
    df = df.copy()
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"], errors="coerce")
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

    pivot["SUM"] = pivot.sum(axis=1, skipna=True)
    pivot["AVG"] = pivot[range(1, 25)].mean(axis=1, skipna=True)
    pivot = pivot.sort_index(ascending=False)

    return pivot


def style_pivot_table(pivot: pd.DataFrame, hour_cols: list, cmap_name: str = "RdYlGn_r"):
    """Custom background coloring that treats real 0 values as normal data
    (so they get a proper color, e.g. the 'low' end of the colormap) and only
    uses a neutral grey (never black) for genuinely missing cells."""
    color_cols = hour_cols + ["SUM", "AVG"]

    vals = pivot[color_cols].to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]

    if finite_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
        if vmin == vmax:
            # avoid a degenerate (divide-by-zero) normalization range
            vmax = vmin + 1.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)

    def colorize(series: pd.Series):
        styles = []
        for v in series:
            if pd.isna(v):
                styles.append("background-color: #eeeeee; color: #999999;")
            else:
                rgba = cmap(norm(float(v)))
                hex_color = mcolors.to_hex(rgba)
                # pick readable text color based on background luminance
                r, g, b = mcolors.to_rgb(hex_color)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.55 else "white"
                styles.append(f"background-color: {hex_color}; color: {text_color};")
        return styles

    styled = pivot.style.apply(colorize, subset=color_cols).format(precision=0, na_rep="—")
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
    st.dataframe(styled, use_container_width=True, height=550)

    st.download_button(
        label=f"⬇️ Κατέβασε τον πίνακα ({sheet_name}) ως CSV",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{sheet_name}",
    )


def build_country_comparison(sheets_dict: dict) -> pd.DataFrame:
    """sheets_dict: {country_code: raw_df} each with delivery_ts/value columns.
    Returns a wide dataframe indexed by delivery_ts, one column per country."""
    combined = None
    for country, df in sheets_dict.items():
        if not {"delivery_ts", "value"}.issubset(df.columns):
            continue
        d = df[["delivery_ts", "value"]].copy()
        d["delivery_ts"] = pd.to_datetime(d["delivery_ts"], errors="coerce")
        d = d.dropna(subset=["delivery_ts"]).rename(columns={"value": country})
        d = d.set_index("delivery_ts")
        combined = d if combined is None else combined.join(d, how="outer")
    if combined is None:
        return pd.DataFrame()
    return combined.sort_index()


def render_country_group(direction_label: str, sheets_dict: dict):
    combined = build_country_comparison(sheets_dict)
    if combined.empty:
        st.info(f"Δεν βρέθηκαν δεδομένα για {direction_label}.")
        return

    min_date, max_date = combined.index.min().date(), combined.index.max().date()
    start_date, end_date = date_range_picker(min_date, max_date, key=f"dr_{direction_label}")

    mask = (combined.index.date >= start_date) & (combined.index.date <= end_date)
    df_filtered = combined.loc[mask]

    st.subheader(f"{direction_label} ανά Χώρα")

    countries = list(combined.columns)
    selected_countries = st.multiselect(
        "Επίλεξε χώρες:", countries, default=countries, key=f"countries_{direction_label}"
    )

    if selected_countries:
        st.line_chart(df_filtered[selected_countries], use_container_width=True)
        st.dataframe(df_filtered[selected_countries], use_container_width=True, height=400)

        st.download_button(
            label=f"⬇️ Κατέβασε ({direction_label}) ως CSV",
            data=df_filtered[selected_countries].to_csv().encode("utf-8-sig"),
            file_name=f"{direction_label}_per_country.csv",
            mime="text/csv",
            key=f"dl_{direction_label}",
        )
    else:
        st.info("Επίλεξε τουλάχιστον μία χώρα.")


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
        directions_available = sorted(set(d for d, _ in country_sheet_matches.values()))

        if directions_available:
            direction_choice = st.radio(
                "Κατεύθυνση:", directions_available, horizontal=True, key="direction_choice"
            )

            relevant_sheets = [
                s for s, (d, _) in country_sheet_matches.items() if d == direction_choice
            ]

            sheets_dict = {}
            for s in relevant_sheets:
                _, country = country_sheet_matches[s]
                sheets_dict[country] = pd.read_excel(xls, sheet_name=s)

            render_country_group(direction_choice, sheets_dict)
        else:
            st.info(
                "Δεν βρέθηκαν sheets τύπου 'Export GR-XX' / 'Import GR-XX' στο αρχείο."
            )
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
