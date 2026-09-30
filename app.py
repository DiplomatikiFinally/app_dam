import io
import re
from datetime import timedelta

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


SPACER_COL = " "  # κενή στήλη ανάμεσα στην τελευταία ώρα και το SUM/AVG


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

    pivot.insert(pivot.columns.get_loc("SUM"), SPACER_COL, "")

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

    styled = pivot.style.apply(colorize_row, subset=hour_cols, axis=1)
    for total_col in ("SUM", "AVG"):
        if total_col in pivot.columns:
            styled = styled.apply(colorize_col, subset=[total_col], axis=0)
    return styled.format(precision=0, na_rep="")


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
# Τίτλοι εμφάνισης (μόνο για το UI - τα ονόματα των sheets στο Excel μένουν ίδια)
DISPLAY_TITLES = {
    "imports": "Imports (AL, BG, MK, IT, TR)",
    "exports": "Exports (AL, BG, MK, IT, TR)",
    "implicit": "Implicit (BG, IT)",
}


def display_title(sheet_name: str) -> str:
    return DISPLAY_TITLES.get(sheet_name.strip().lower(), sheet_name)


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

    st.subheader(display_title(sheet_name))

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

    st.subheader(display_title(sheet_name))

    if pivot.empty:
        st.info("Δεν υπάρχουν δεδομένα για το επιλεγμένο διάστημα.")
        return

    # MCP: μόνο AVG · όλα τα υπόλοιπα: μόνο SUM
    pivot = pivot.drop(columns=["SUM" if "mcp" in sheet_name.lower() else "AVG"])

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
        data=pivot.drop(columns=[SPACER_COL]).to_csv().encode("utf-8-sig"),
        file_name=f"{sheet_name}_pivot.csv",
        mime="text/csv",
        key=f"dl_{key_prefix}_{sheet_name}",
    )


# ---------------------------------------------------------------------------
# Summary tab: D vs D-1 και D vs W-1
#   D-1 : Τρι–Παρ -> προηγούμενη μέρα · Δευτέρα -> προηγούμενη Παρασκευή ·
#         Σάββατο/Κυριακή -> το ίδιο (Σαβ/Κυρ) της προηγούμενης εβδομάδας
#   W-1 : καθημερινή -> μέσος όρος Δευ–Παρ της προηγούμενης εβδομάδας ·
#         Σαβ/Κυρ -> μέσος όρος Σαβ+Κυρ του προηγούμενου σαββατοκύριακου
# ---------------------------------------------------------------------------
DAY_NAMES = ["Δευ", "Τρι", "Τετ", "Πεμ", "Παρ", "Σαβ", "Κυρ"]

# key: (sheet matcher, aggregation, sign)   sign=-1 -> εμφανίζεται αρνητικό (Exports)
SUMMARY_METRICS = {
    "mcp":     (lambda s: s == "mcp",             "AVG",  1),
    "hv":      (lambda s: s == "load hv",         "SUM",  1),
    "mv":      (lambda s: s == "load mv",         "SUM",  1),
    "lv":      (lambda s: s == "load lv",         "SUM",  1),
    "losses":  (lambda s: s == "load losses",     "SUM",  1),
    "pump":    (lambda s: s == "load pump",       "SUM",  1),
    "bess_buy":  (lambda s: s == "bess_buy",      "SUM",  1),
    "bess_sell": (lambda s: s == "bess_sell",     "SUM",  1),
    "dr":      (lambda s: s == "load d-r",        "SUM",  1),
    "lignite": (lambda s: s == "lignite",         "SUM",  1),
    "gas":     (lambda s: s == "natural gas",     "SUM",  1),
    "res":     (lambda s: s == "res",             "SUM",  1),
    "hydro":   (lambda s: s == "hydro",           "SUM",  1),
    "imports": (lambda s: s == "imports",         "SUM",  1),
    "exports": (lambda s: s == "exports",         "SUM", -1),
}

BAR_COLORS = ["#10233f", "#1b4a8a", "#3b74c0", "#4f8fd6", "#8fb1e2", "#c6d6f0"]

SUMMARY_CSS = """
<style>
.dam-card{border:1px solid rgba(128,128,128,.28);border-radius:10px;overflow-x:auto;
  font-variant-numeric:tabular-nums}
.dam-card table{border-collapse:collapse;width:100%;font-size:15px}
.dam-card th{font-weight:600;font-size:13px;opacity:.75;padding:9px 10px;text-align:right;
  border-bottom:1px solid rgba(128,128,128,.35);white-space:nowrap;line-height:1.3}
.dam-card th.grp{text-align:center;opacity:1;font-size:14px;background:rgba(128,128,128,.14)}
.dam-card th:first-child,.dam-card td:first-child{text-align:left}
.dam-card td{padding:8px 10px;text-align:right;white-space:nowrap}
.dam-card tr:nth-child(even) td{background:rgba(128,128,128,.08)}
.dam-card td.lbl{font-weight:500}
.dam-card tr.sep-top td{border-top:2px solid rgba(128,128,128,.55)}
.dam-card tr.sub td{font-size:13.5px;opacity:.85}
.dam-card tr.sub td.lbl{padding-left:26px;font-weight:400}
.dam-card td.cur{font-weight:700;font-size:16px}
.dam-card tr.sub td.cur{font-size:14px;font-weight:600}
.dam-card td.ref{opacity:.75}
.dam-card td.sep,.dam-card th.sep{border-left:1px solid rgba(128,128,128,.35)}
.dam-card .u{font-size:11px;opacity:.6;margin-left:4px}
.dam-card .up{color:#2e9d5b;font-weight:600}
.dam-card .dn{color:#d64545;font-weight:600}
.dam-card .fl{opacity:.6}
</style>
"""


def _fmt_num(x, decimals=0):
    """Ελληνική μορφή: 164.605 / 57,41 — το πρόσημο μόνο όταν είναι αρνητικό."""
    if x is None or pd.isna(x):
        return "—"
    s = f"{abs(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "-" + s if round(x, decimals) < 0 else s


def _prev_same_type_day(d):
    wd = d.weekday()
    if wd == 0:            # Δευτέρα -> προηγούμενη Παρασκευή
        return d - pd.Timedelta(days=3)
    if wd <= 4:            # Τρι..Παρ -> χθες
        return d - pd.Timedelta(days=1)
    return d - pd.Timedelta(days=7)   # Σαβ/Κυρ -> ίδια μέρα προηγούμενης εβδομάδας


def _w1_days(d):
    monday = d - pd.Timedelta(days=d.weekday())
    if d.weekday() >= 5:   # προηγούμενο σαββατοκύριακο
        return [monday - pd.Timedelta(days=2), monday - pd.Timedelta(days=1)]
    return [monday - pd.Timedelta(days=7 - i) for i in range(5)]   # προηγούμενη Δευ..Παρ


def _delta_cells(cur, ref, decimals, unit):
    if cur is None or ref is None or pd.isna(cur) or pd.isna(ref):
        return '<td class="fl">—</td><td class="fl">—</td>'
    d = cur - ref
    cls = "up" if d > 0 else ("dn" if d < 0 else "fl")
    arrow = "↗" if d > 0 else ("↘" if d < 0 else "→")
    pct = f"{_fmt_num(d / abs(ref) * 100, 0)}%" if ref != 0 else "—"
    return (
        f'<td class="{cls}">{_fmt_num(d, decimals)}<span class="u">{unit}</span></td>'
        f'<td class="{cls}">{pct} {arrow}</td>'
    )


def _daily_series(raw_df, sheet_name, agg, sign, start_date, end_date):
    """Ημερήσια τιμή ανά ημέρα με την ΙΔΙΑ λογική με τους πίνακες (build_hourly_pivot)."""
    try:
        pv = build_hourly_pivot(raw_df, sheet_name=sheet_name, start_date=start_date, end_date=end_date)
        return pv[agg] * sign
    except Exception:
        return pd.Series(dtype=float)


def _dstr(x):
    return f"{DAY_NAMES[x.weekday()]} {x:%d/%m}"


def render_summary_tab(xls, sheet_names, min_date, max_date):
    st.markdown("### ⚡ Day-Ahead Market — Σύγκριση")

    d_sel = st.date_input(
        "Ημέρα D:", value=max_date, min_value=min_date, max_value=max_date, key="summary_day"
    )
    d = pd.Timestamp(d_sel)
    d1 = _prev_same_type_day(d)
    wdays = _w1_days(d)
    start = min([d1] + wdays)

    # ---- ημερήσιες τιμές ανά metric --------------------------------------
    lower = {s.lower().strip(): s for s in sheet_names}
    daily = {}
    for key, (matcher, agg, sign) in SUMMARY_METRICS.items():
        real = next((orig for low, orig in lower.items() if matcher(low)), None)
        if real is None:
            continue
        raw_df = xls[real].copy()
        daily[key] = _daily_series(raw_df, real, agg, sign, start.date(), d.date())

    have_split = all(k in daily for k in ("hv", "mv", "lv", "losses"))
    if have_split:
        daily["load_total"] = sum(daily[k] for k in ("hv", "mv", "lv", "losses"))
    if "imports" in daily and "exports" in daily:
        daily["net_imports"] = daily["imports"].add(daily["exports"], fill_value=0)

    # ---- γραμμές πίνακα ---------------------------------------------------
    rows = [("MCP", "mcp", "€/MWh", 0), ("Imports", "imports", "MWh", 0),
            ("Load + Losses", "load_total", "MWh", 0)]
    if have_split:
        rows += [("HV load", "hv", "MWh", 1), ("MV load", "mv", "MWh", 1),
                 ("LV load", "lv", "MWh", 1), ("System losses", "losses", "MWh", 1)]
    rows += [("Pump", "pump", "MWh", 0), ("D/R load", "dr", "MWh", 0),
             ("BESS buy", "bess_buy", "MWh", 0), ("Exports", "exports", "MWh", 0),
             ("RES", "res", "MWh", 0), ("Hydro", "hydro", "MWh", 0),
             ("Lignite", "lignite", "MWh", 0), ("Gas", "gas", "MWh", 0),
             ("BESS sell", "bess_sell", "MWh", 0)]

    def _at(series, day):
        v = series.get(day.date())
        return None if v is None else float(v)

    def _dcalc(key, cur, ref):
        """Exports: Δ/Δ% πάνω στα μεγέθη (|D| - |ref|), ώστε -1.049 vs -2.318 -> -1.269 (-55%)."""
        if key == "exports" and cur is not None and ref is not None and not pd.isna(ref):
            return abs(cur), abs(ref)
        return cur, ref

    body, missing = [], []
    for label, key, unit, level in rows:
        s = daily.get(key)
        if s is None:
            missing.append(label)
            continue
        dec = 2 if key == "mcp" else 0
        cur, prev = _at(s, d), _at(s, d1)
        wk = s.reindex([x.date() for x in wdays]).mean()
        body.append(
            f'<tr class="{"sub" if level else ""} {"sep-top" if key in ("imports", "exports") else ""}">'
            f'<td class="lbl">{label}</td>'
            f'<td class="cur">{_fmt_num(cur, dec)}</td>'
            f'<td class="ref">{_fmt_num(prev, dec)}</td>'
            f"{_delta_cells(*_dcalc(key, cur, prev), dec, unit)}"
            f'<td class="ref sep">{_fmt_num(wk, dec)}</td>'
            f"{_delta_cells(*_dcalc(key, cur, wk), dec, unit)}"
            "</tr>"
        )

    if not body:
        st.info("Δεν βρέθηκαν τα sheets που χρειάζονται για την περίληψη.")
        return

    wk_kind = "Σαβ+Κυρ" if d.weekday() >= 5 else "Δευ–Παρ"
    html = (
        SUMMARY_CSS
        + '<div class="dam-card"><table>'
        + '<tr><th></th><th class="grp" colspan="4">D vs D-1</th>'
        + '<th class="grp sep" colspan="3">D vs W-1</th></tr>'
        + f"<tr><th></th><th>D<br>{_dstr(d)}</th><th>D-1<br>{_dstr(d1)}</th><th>Δ</th><th>Δ%</th>"
        + f'<th class="sep">W-1 avg<br>{wk_kind} {wdays[0]:%d/%m}–{wdays[-1]:%d/%m}</th><th>Δ</th><th>Δ%</th></tr>'
        + "".join(body)
        + "</table></div>"
    )

    left, right = st.columns([5, 3])
    with left:
        st.markdown(html, unsafe_allow_html=True)

    # ---- διάγραμμα Δ ------------------------------------------------------
    with right:
        mode = st.radio("Διάγραμμα Δ (MWh)", ["vs D-1", "vs W-1"], horizontal=True, key="summary_chart_mode")

        def _delta(series):
            if series is None:
                return None
            cur = _at(series, d)
            ref = _at(series, d1) if mode == "vs D-1" else series.reindex([x.date() for x in wdays]).mean()
            return None if cur is None or pd.isna(ref) else cur - ref

        items = [
            ("Load + Losses", daily.get("load_total")),
            ("Net imports", daily.get("net_imports")),
            ("RES", daily.get("res")),
            ("Hydro", daily.get("hydro")),
            ("Lignite", daily.get("lignite")),
            ("Gas", daily.get("gas")),
        ]
        chart_rows = [
            {"name": n, "delta": _delta(s), "color": c}
            for (n, s), c in zip(items, BAR_COLORS)
            if _delta(s) is not None
        ]
        if chart_rows:
            cdf = pd.DataFrame(chart_rows)
            cdf["label"] = cdf["delta"].apply(_fmt_num)
            names, colors = cdf["name"].tolist(), cdf["color"].tolist()

            base = alt.Chart(cdf).encode(x=alt.X("name:N", sort=names, axis=None))
            bars = base.mark_bar().encode(
                y=alt.Y("delta:Q", title="Δ (MWh)", axis=alt.Axis(format=",.0f")),
                color=alt.Color(
                    "name:N", sort=names, scale=alt.Scale(domain=names, range=colors),
                    legend=alt.Legend(title=None, orient="bottom", columns=2),
                ),
                tooltip=["name", "label"],
            )
            txt_pos = base.mark_text(dy=-8, fontSize=13, fontWeight="bold").encode(
                y="delta:Q", text="label:N").transform_filter("datum.delta >= 0")
            txt_neg = base.mark_text(dy=14, fontSize=13, fontWeight="bold").encode(
                y="delta:Q", text="label:N").transform_filter("datum.delta < 0")
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
            st.altair_chart((bars + txt_pos + txt_neg + zero).properties(height=340),
                            use_container_width=True)

    st.caption(
        f"D-1 = {_dstr(d1)} (καθημερινή → προηγούμενη καθημερινή, Σαβ/Κυρ → ίδια μέρα προηγ. εβδομάδας). "
        f"W-1 = μέσος όρος {wk_kind} ({_dstr(wdays[0])} – {_dstr(wdays[-1])}). "
        "MCP: ημερήσιος μέσος όρος (€/MWh)· υπόλοιπα: ημερήσιο σύνολο (MWh). "
        "Exports εμφανίζονται αρνητικά, αλλά Δ/Δ% υπολογίζονται πάνω στο μέγεθός τους (μείωση εξαγωγών = αρνητικό Δ). "
        "Στο διάγραμμα, Net imports = Imports + Exports."
    )
    if missing:
        st.warning(
            "Δεν βρέθηκαν sheets για: " + ", ".join(missing)
            + ". Τρέξε ξανά το main_for_excel.py και ανέβασε το νέο Excel."
        )


# Sheets που ΔΕΝ εμφανίζονται στο tab 1 (lower-case ονόματα)
HIDDEN_IN_TAB1 = {"load hv", "load mv", "load lv", "load losses", "imports", "exports", "implicit"}

CMP_SLOTS = 4
CMP_COLORS = ["#1b4a8a", "#d64545", "#2e9d5b", "#e0a020"]


def render_compare_tab(xls, sheet_names, start_date, end_date, min_date, max_date):
    """Tab 4: πίνακες MCP / Total Residual / Forecast Residual + διάγραμμα σύγκρισης.
    Κάθε καμπύλη του διαγράμματος = (μέγεθος, ημέρα), άρα μπορείς π.χ. να βάλεις
    MCP σήμερα vs MCP χθες, MCP σήμερα vs Forecast Residual προχθές, κ.λπ."""
    lower = {s.lower().strip(): s for s in sheet_names}
    sources = {}  # όνομα -> (sheet, group)
    if "mcp" in lower:
        sources["MCP"] = (lower["mcp"], "mcp")
    tr = next((o for l, o in lower.items() if l.startswith("total residual")), None)
    fr = next((o for l, o in lower.items() if l.startswith("residual forecast")), None)
    if tr:
        sources["Total Residual"] = (tr, "res")
    if fr:
        sources["Forecast Residual"] = (fr, "res")

    if not sources:
        st.info("Δεν βρέθηκαν τα sheets MCP / Total Residual / Residual Forecast στο αρχείο.")
        return

    # ---- πίνακες (ίδιοι με το tab 1) -------------------------------------
    for sheet, _grp in sources.values():
        render_standard_sheet(xls[sheet].copy(), sheet, start_date, end_date, key_prefix="cmp")
        st.markdown("---")

    # ---- διάγραμμα σύγκρισης ---------------------------------------------
    st.markdown("### 🔀 Σύγκριση καμπυλών")
    st.caption(
        f"Διάλεξε έως {CMP_SLOTS} καμπύλες· κάθε μία είναι ένα μέγεθος για μία ημέρα. "
        "Άφησε «—» για να μην εμφανιστεί."
    )

    hour_cols = list(range(1, 25))
    pivots = {
        name: build_hourly_pivot(xls[sheet].copy(), sheet_name=sheet, start_date=min_date, end_date=max_date)
        for name, (sheet, _g) in sources.items()
    }

    names = list(sources)
    options = ["—"] + names
    first = "MCP" if "MCP" in sources else names[0]
    prev_day = max(min_date, max_date - timedelta(days=1))
    defaults = [(first, max_date), (first, prev_day), ("—", max_date), ("—", max_date)]

    picks = []
    for i in range(CMP_SLOTS):
        c1, c2 = st.columns([2, 1])
        with c1:
            name = st.selectbox(
                f"Καμπύλη {i + 1}", options, index=options.index(defaults[i][0]), key=f"cmp_series_{i}"
            )
        with c2:
            day = st.date_input(
                "Ημέρα", value=defaults[i][1], min_value=min_date, max_value=max_date, key=f"cmp_day_{i}"
            )
        picks.append((name, day))

    rows, seen, warns = [], set(), []
    for name, day in picks:
        if name == "—" or day is None:
            continue
        label = f"{name} · {_dstr(pd.Timestamp(day))}"
        if label in seen:
            continue
        pv = pivots[name]
        if day not in pv.index or (pv.loc[day, hour_cols] == 0).all():
            warns.append(f"Δεν υπάρχουν δεδομένα για {label}.")
            continue
        seen.add(label)
        for h in hour_cols:
            rows.append({"label": label, "hour": h, "value": float(pv.loc[day, h]), "group": sources[name][1]})

    for w in warns:
        st.warning(w)
    if not rows:
        st.info("Επίλεξε τουλάχιστον μία καμπύλη με δεδομένα.")
        return

    df = pd.DataFrame(rows)
    labels = list(dict.fromkeys(df["label"]))
    color = alt.Color(
        "label:N",
        sort=labels,
        scale=alt.Scale(domain=labels, range=CMP_COLORS[: len(labels)]),
        legend=alt.Legend(title=None, orient="bottom"),
    )

    def _line(gdf, title, orient, dash=None):
        extra = {"strokeDash": dash} if dash else {}
        return (
            alt.Chart(gdf)
            .mark_line(point=True, strokeWidth=2.5, **extra)
            .encode(
                x=alt.X("hour:O", title="Ώρα", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False),
                        axis=alt.Axis(orient=orient, format=",.0f")),
                color=color,
                tooltip=[
                    alt.Tooltip("label:N", title="Καμπύλη"),
                    alt.Tooltip("hour:O", title="Ώρα"),
                    alt.Tooltip("value:Q", title=title, format=",.2f"),
                ],
            )
        )

    g_mcp, g_res = df[df["group"] == "mcp"], df[df["group"] == "res"]
    both = not g_mcp.empty and not g_res.empty
    layers = []
    if not g_mcp.empty:
        layers.append(_line(g_mcp, "MCP (€/MWh)", "left"))
    if not g_res.empty:
        layers.append(_line(g_res, "Residual (MWh)", "right" if both else "left",
                            dash=[6, 4] if both else None))
    chart = alt.layer(*layers).resolve_scale(y="independent") if both else layers[0]
    st.altair_chart(chart.properties(height=420), use_container_width=True)
    if both:
        st.caption("MCP: αριστερός άξονας (€/MWh, συνεχής γραμμή) · Residual: δεξιός άξονας (MWh, διακεκομμένη).")


@st.cache_data(show_spinner="Φόρτωση Excel...")
def load_workbook(file_bytes: bytes) -> dict:
    """Διαβάζει ΟΛΑ τα sheets μία φορά και τα κρατάει στη μνήμη (cache).
    Χωρίς αυτό, κάθε αλλαγή σε widget (π.χ. κλείσιμο μιας κατηγορίας) ξαναδιάβαζε
    το Excel από την αρχή, γιατί το Streamlit τρέχει ολόκληρο το script σε κάθε κλικ."""
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)


if uploaded_file is not None:
    xls = load_workbook(uploaded_file.getvalue())  # dict {sheet_name: DataFrame}
    sheet_names = list(xls.keys())

    all_dates = []

    for sheet in sheet_names:
        try:
            temp_df = xls[sheet][["delivery_ts"]]

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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📁 Dam Results", "🌍 Imports / Exports", "⚡ Summary", "📈 MCP & Residual"]
    )

    with tab1:
        if sheet_names:
            # Στο tab 1 δεν εμφανίζονται μόνα τους τα HV/MV/LV/Losses ούτε τα Imports/Exports/Implicit
            # (τα τελευταία είναι στο tab "Imports / Exports" -> Net & Totals).
            standard_sheets = [s for s in standard_sheets if s.strip().lower() not in HIDDEN_IN_TAB1]
            default_selection = standard_sheets
            selected_sheets = st.multiselect(
                "Επίλεξε κατηγορίες (μπορείς πάνω από μία):",
                standard_sheets,
                default=default_selection,
                format_func=display_title,
                key="select_standard",
            )

            if not selected_sheets:
                st.info("Επίλεξε τουλάχιστον μία κατηγορία για να εμφανιστούν δεδομένα.")

            for sheet in selected_sheets:
                raw_df = xls[sheet].copy()
                render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="tab1")
                st.markdown("---")
                st.markdown("---")
        else:
            st.info("Δεν βρέθηκαν sheets δεδομένων.")

    with tab2:
        net_only = sorted([
                                    s for s in sheet_names
                                    if ("net" in s.lower() or "total" in s.lower())
                                    and "residual" not in s.lower()
                                ])
        flow_totals = [
            next(s for s in sheet_names if s.strip().lower() == name)
            for name in ("imports", "exports", "implicit")
            if any(s.strip().lower() == name for s in sheet_names)
        ]
        net_sheets = flow_totals + [s for s in net_only if s not in flow_totals]
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
                        raw_df = xls[sheet].copy()
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
                        raw_df = xls[sheet].copy()
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
                        raw_df = xls[sheet].copy()
                        render_standard_sheet(raw_df, sheet, start_date, end_date, key_prefix="exp")
                        st.markdown("---")
                else:
                    st.info("Δεν βρέθηκαν αναλυτικά sheets εξαγωγών.")

    with tab3:
        render_summary_tab(xls, sheet_names, global_min_date, global_max_date)

    with tab4:
        render_compare_tab(xls, sheet_names, start_date, end_date, global_min_date, global_max_date)
else:
    st.info("Ανέβασε ένα .xlsx αρχείο για να ξεκινήσεις.")
