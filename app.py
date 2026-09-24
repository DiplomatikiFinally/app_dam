import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DAM Results Heatmap", layout="wide")
st.title("📊 DAM Results — Ωριαία Ανάλυση")

uploaded_file = st.file_uploader("Ανέβασε το Excel αρχείο (.xlsx)", type=["xlsx"])

COUNTRY_SHEETS = {"imports_per_country", "exports_per_country"}


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
    styled = (
        pivot.style
        .background_gradient(cmap="RdYlGn_r", subset=hour_cols, axis=None)
        .background_gradient(cmap="RdYlGn_r", subset=["SUM"], axis=None)
        .background_gradient(cmap="RdYlGn_r", subset=["AVG"], axis=None)
        .format(precision=0)
    )
    st.dataframe(styled, use_container_width=True, height=550)

    st.download_button(
        label=f"⬇️ Κατέβασε τον πίνακα ({sheet_name}) ως CSV",
        data=pivot.to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{sheet_name}",
    )


def render_country_sheet(raw_df: pd.DataFrame, sheet_name: str):
    if "delivery_ts" not in raw_df.columns:
        st.error("Το sheet αυτό δεν έχει στήλη 'delivery_ts'.")
        return

    df = raw_df.copy()
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"], errors="coerce")
    df = df.dropna(subset=["delivery_ts"])

    country_cols = [c for c in df.columns if c != "delivery_ts"]

    min_date, max_date = df["delivery_ts"].dt.date.min(), df["delivery_ts"].dt.date.max()
    start_date, end_date = date_range_picker(min_date, max_date, key=f"dr_{sheet_name}")

    mask = (df["delivery_ts"].dt.date >= start_date) & (df["delivery_ts"].dt.date <= end_date)
    df_filtered = df.loc[mask].set_index("delivery_ts")

    st.subheader(f"{sheet_name}")

    selected_countries = st.multiselect(
        "Επίλεξε χώρες:", country_cols, default=country_cols, key=f"countries_{sheet_name}"
    )

    if selected_countries:
        st.line_chart(df_filtered[selected_countries], use_container_width=True)
        st.dataframe(df_filtered[selected_countries], use_container_width=True, height=400)

        st.download_button(
            label=f"⬇️ Κατέβασε ({sheet_name}) ως CSV",
            data=df_filtered[selected_countries].to_csv().encode("utf-8-sig"),
            file_name=f"{sheet_name}.csv",
            mime="text/csv",
            key=f"dl_{sheet_name}",
        )
    else:
        st.info("Επίλεξε τουλάχιστον μία χώρα.")


if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    standard_sheets = [s for s in sheet_names if s.lower() not in COUNTRY_SHEETS]
    country_sheets = [s for s in sheet_names if s.lower() in COUNTRY_SHEETS]

    tab1, tab2 = st.tabs(["📁 Όλα τα Δεδομένα", "🌍 Imports / Exports ανά Χώρα"])

    with tab1:
        if standard_sheets:
            default_selection = [standard_sheets[0]] if standard_sheets else []
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
                render_standard_sheet(raw_df, sheet)
                st.markdown("---")
                st.markdown("---")
        else:
            st.info("Δεν βρέθηκαν sheets δεδομένων.")

    with tab2:
        if country_sheets:
            selected_country_sheet = st.selectbox(
                "Επίλεξε κατηγορία:", country_sheets, key="select_country"
            )
            raw_df = pd.read_excel(xls, sheet_name=selected_country_sheet)
            render_country_sheet(raw_df, selected_country_sheet)
        else:
            st.info("Δεν βρέθηκαν sheets 'imports_per_country' / 'exports_per_country' στο αρχείο.")
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
