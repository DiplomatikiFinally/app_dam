import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="DAM Results Heatmap", layout="wide")
st.title("📊 DAM Results — Ωριαία Ανάλυση")

uploaded_file = st.file_uploader("Ανέβασε το Excel αρχείο (.xlsx)", type=["xlsx"])


def build_hourly_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Παίρνει DataFrame με στήλες delivery_ts, value και επιστρέφει
    pivot table: γραμμές = ημερομηνία, στήλες = ώρα (1-24), + SUM/AVG."""
    df = df.copy()
    df["delivery_ts"] = pd.to_datetime(df["delivery_ts"], errors="coerce")
    df = df.dropna(subset=["delivery_ts"])

    df["date"] = df["delivery_ts"].dt.date
    df["hour"] = df["delivery_ts"].dt.hour + 1  # 1..24

    pivot = df.pivot_table(
        index="date", columns="hour", values="value", aggfunc="mean"
    )

    # Βεβαιωνόμαστε ότι υπάρχουν όλες οι στήλες 1..24, με τη σωστή σειρά
    for h in range(1, 25):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[[h for h in range(1, 25)]]

    # SUM / AVG ανά γραμμή
    pivot["SUM"] = pivot.sum(axis=1, skipna=True)
    pivot["AVG"] = pivot[range(1, 25)].mean(axis=1, skipna=True)

    # Ταξινόμηση ημερομηνιών, πιο πρόσφατη πρώτη
    pivot = pivot.sort_index(ascending=False)

    return pivot


if uploaded_file is not None:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    selected_sheet = st.selectbox("Επίλεξε κατηγορία (sheet):", sheet_names)

    raw_df = pd.read_excel(xls, sheet_name=selected_sheet)

    if not {"delivery_ts", "value"}.issubset(raw_df.columns):
        st.error("Το sheet αυτό δεν έχει στήλες 'delivery_ts' και 'value'.")
    else:
        pivot = build_hourly_pivot(raw_df)

        st.subheader(f"{selected_sheet} — Πίνακας ανά ημέρα / ώρα")

        hour_cols = list(range(1, 25))

        styled = (
            pivot.style
            .background_gradient(cmap="RdYlGn_r", subset=hour_cols, axis=None)
            .background_gradient(cmap="RdYlGn_r", subset=["SUM"], axis=None)
            .background_gradient(cmap="RdYlGn_r", subset=["AVG"], axis=None)
            .format(precision=0)
        )

        st.dataframe(styled, use_container_width=True, height=650)

        st.download_button(
            label=f"⬇️ Κατέβασε τον πίνακα ({selected_sheet}) ως CSV",
            data=pivot.to_csv().encode("utf-8-sig"),
            file_name=f"{selected_sheet}_pivot.csv",
            mime="text/csv",
        )
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
