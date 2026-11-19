"""
Potentiometrie-Auswertung als Streamlit-Website.

Web-Version von Tit_Analyse_max_19_09.py (Tkinter) im Aufbau von Kond_Analyse.py.
Äquivalenzpunkt = Maximum der 1. Ableitung der PCHIP- bzw. Spline-interpolierten
Titrationskurve (Savitzky-Golay-geglättet, scipy.signal.find_peaks).

Start:  streamlit run Potentio_Analyse.py
"""
from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.signal import find_peaks, savgol_filter

# --------------------------------------------------------------------------- #
# Konstanten (identisch zum Desktop-Skript)
# --------------------------------------------------------------------------- #
N_FINE = 5000            # Stützstellen der feinen Interpolation (np.linspace)
PEAK_PROMINENCE = 0.3    # find_peaks: Mindest-Prominenz [pH/mL]
PEAK_DISTANCE = 200      # find_peaks: Mindestabstand [Stützstellen]
EMPTY_ROWS = 120         # Zeilen einer leeren Tabelle
METHODS = (("PCHIP", "PCHIP"), ("SPLINE", "Spline"))

# Reihenfolge = Plot-Layout: [0,0] [0,1] / [1,0] [1,1]; label = Dateiname beim Export
PANELS = {
    "pchip_kurve": dict(row=0, col=0, method="PCHIP", deriv=False, title="PCHIP Kurve", label="PCHIP_Kurve"),
    "spline_kurve": dict(row=0, col=1, method="SPLINE", deriv=False, title="Spline Kurve", label="Spline_Kurve"),
    "pchip_abl": dict(row=1, col=0, method="PCHIP", deriv=True, title="PCHIP 1. Ableitung", label="PCHIP_1_Ableitung"),
    "spline_abl": dict(row=1, col=1, method="SPLINE", deriv=True, title="Spline 1. Ableitung", label="Spline_1_Ableitung"),
}
EXPORT_BUTTONS = (
    ("all", "Gesamtbild (2×2)"),
    ("pchip_kurve", "PCHIP Kurve"),
    ("spline_kurve", "Spline Kurve"),
    ("pchip_abl", "PCHIP 1. Ableitung"),
    ("spline_abl", "Spline 1. Ableitung"),
)


# --------------------------------------------------------------------------- #
# Hilfsfunktionen: Parsen
# --------------------------------------------------------------------------- #
def col_index_to_letter(col_idx: int) -> str:
    result = ""
    col_idx += 1
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def parse_float(x) -> float | None:
    """Zahl mit Dezimalpunkt ODER -komma; None bei leer/ungültig/nicht endlich."""
    s = cell_text(x).replace(",", ".")
    if not s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def collect_points(df: pd.DataFrame):
    """Aktive Zeilen mit gültigem (V, pH)-Paar. Gibt (v, ph, n_ignoriert) zurück."""
    v, ph, skipped = [], [], 0
    for active, vs, ps in zip(df["Aktiv"], df["V [mL]"], df["pH"]):
        if not (True if pd.isna(active) else bool(active)):
            continue
        vv, pp = parse_float(vs), parse_float(ps)
        if vv is not None and pp is not None:
            v.append(vv)
            ph.append(pp)
        elif cell_text(vs) or cell_text(ps):
            skipped += 1
    return v, ph, skipped


def fnum(x: float, nd: int = 2, comma: bool = False) -> str:
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if comma else s


# --------------------------------------------------------------------------- #
# Auswertung (1:1 aus analyze_tab / find_half_equivalence_points übernommen)
# --------------------------------------------------------------------------- #
def find_half_equivalence_points(x, y, d, peak_indices):
    half = []
    prev_v = x[0]
    for pk_idx in peak_indices:
        v_aep = x[pk_idx]
        v_half = (prev_v + v_aep) / 2.0
        idx_half = int(np.argmin(np.abs(x - v_half)))
        half.append((float(v_half), float(y[idx_half]), idx_half, float(d[idx_half])))
        prev_v = v_aep
    return half


def analyze(name, v_raw, ph_raw, v_min=None, v_max=None, smooth_window=101):
    """Gibt (Ergebnis-dict, None) oder (None, Fehlermeldung) zurück."""
    if len(v_raw) < 5:
        return None, "Mindestens 5 gültige Wertepaare nötig – Daten einfügen, importieren oder Beispieldaten laden."

    v, ph = np.asarray(v_raw, float), np.asarray(ph_raw, float)
    idx = np.argsort(v)
    v, ph = v[idx], ph[idx]

    if v_min is not None:
        keep = v >= v_min
        v, ph = v[keep], ph[keep]
    if v_max is not None:
        keep = v <= v_max
        v, ph = v[keep], ph[keep]
    if len(v) < 5:
        return None, "Im gewählten Volumenbereich (V min / V max) liegen weniger als 5 Punkte."

    dup = np.unique(v[1:][np.diff(v) == 0])
    if dup.size:
        return None, (
            "Doppelte Volumenwerte (" + ", ".join(f"{x:g}" for x in dup) + " mL): "
            "Die Interpolation braucht streng steigende Werte. Eine der Zeilen deaktivieren (Spalte „Aktiv“) oder korrigieren."
        )

    xf = np.linspace(v.min(), v.max(), N_FINE)

    wl = int(smooth_window)
    if wl >= len(xf):
        wl = len(xf) - 1 if (len(xf) % 2 == 0) else len(xf) - 2
    if wl % 2 == 0:
        wl += 1
    if wl < 3:
        wl = 3

    p_i = PchipInterpolator(v, ph)
    yp = p_i(xf)
    dp = savgol_filter(np.gradient(yp, xf), wl, 3)
    s_i = CubicSpline(v, ph)
    ys = s_i(xf)
    ds = savgol_filter(s_i.derivative()(xf), wl, 3)

    curves = {}
    for key, y, d in (("PCHIP", yp, dp), ("SPLINE", ys, ds)):
        pks, _ = find_peaks(d, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE)
        curves[key] = {
            "y": y,
            "d": d,
            "peaks": [dict(v=float(xf[p]), ph=float(y[p]), d=float(d[p])) for p in pks],
            "half": find_half_equivalence_points(xf, y, d, pks),
        }
    return dict(name=name, v=v, ph=ph, xf=xf, window=wl, step=float(xf[1] - xf[0]), curves=curves), None


# --------------------------------------------------------------------------- #
# Plotly (Anzeige) – wie Kond_Analyse.py
# --------------------------------------------------------------------------- #
def build_plotly(data, show_aep, show_haep, comma, xlabel):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[PANELS[k]["title"] for k in PANELS],
        horizontal_spacing=0.08, vertical_spacing=0.13,
    )
    legend_done = set()

    def once(group):
        show = group not in legend_done
        legend_done.add(group)
        return show

    for pn in PANELS.values():
        r, c = pn["row"] + 1, pn["col"] + 1
        cv = data["curves"][pn["method"]]
        deriv = pn["deriv"]
        y = cv["d"] if deriv else cv["y"]

        if not deriv:
            fig.add_trace(go.Scatter(
                x=data["v"], y=data["ph"], mode="markers", name="Messwerte",
                legendgroup="mess", showlegend=once("mess"),
                marker=dict(color="blue", size=7, opacity=0.8),
                hovertemplate="V = %{x:.3f} mL<br>pH = %{y:.3f}<extra>Messwert</extra>",
            ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=data["xf"], y=y, mode="lines", showlegend=False,
            name=pn["title"], line=dict(color="crimson" if deriv else "royalblue", width=2),
            hovertemplate="V = %{x:.3f} mL<br>" + ("dpH/dV = %{y:.3f}" if deriv else "pH = %{y:.3f}")
            + "<extra>" + pn["title"] + "</extra>",
        ), row=r, col=c)

        if show_aep and cv["peaks"]:
            for pk in cv["peaks"]:
                fig.add_vline(x=pk["v"], row=r, col=c, line=dict(color="gray", width=1), opacity=0.7)
            fig.add_trace(go.Scatter(
                x=[pk["v"] for pk in cv["peaks"]],
                y=[pk["d"] if deriv else pk["ph"] for pk in cv["peaks"]],
                text=[fnum(pk["v"], 2, comma) for pk in cv["peaks"]],
                mode="markers+text", textposition="middle right",
                name="Äquivalenzpunkt", legendgroup="aep", showlegend=once("aep"),
                marker=dict(color="black", size=11, symbol="cross"),
                hovertemplate="ÄP: V = %{x:.3f} mL<br>%{y:.3f}<extra></extra>",
            ), row=r, col=c)

        if show_haep and cv["half"]:
            fig.add_trace(go.Scatter(
                x=[h[0] for h in cv["half"]],
                y=[h[3] if deriv else h[1] for h in cv["half"]],
                text=[fnum(h[0], 2, comma) for h in cv["half"]],
                mode="markers+text", textposition="bottom right",
                name="Halbäquivalenzpunkt", legendgroup="haep", showlegend=once("haep"),
                marker=dict(color="black", size=9, symbol="x"),
                hovertemplate="HÄP: V = %{x:.2f} mL<br>%{y:.3f}<extra></extra>",
            ), row=r, col=c)

    fig.update_xaxes(title_text=xlabel, gridcolor="#e5e5e5", griddash="dash")
    fig.update_yaxes(gridcolor="#e5e5e5", griddash="dash")
    fig.update_yaxes(title_text="pH", row=1)
    fig.update_yaxes(title_text="dpH/dV / mL⁻¹", row=2)
    fig.update_layout(
        template="plotly_white", height=780, hovermode="closest",
        separators=",." if comma else ".,",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# --------------------------------------------------------------------------- #
# Matplotlib (Export als PDF / JPG, Look des Desktop-Skripts)
# --------------------------------------------------------------------------- #
def draw_panel(ax, key, data, opts):
    pn = PANELS[key]
    cv = data["curves"][pn["method"]]
    deriv, comma = pn["deriv"], opts["comma"]
    xf = data["xf"]

    ax.grid(True, linestyle="--", alpha=0.6)
    if deriv:
        y = cv["d"]
        ax.plot(xf, y, color="crimson")
    else:
        y = cv["y"]
        ax.scatter(data["v"], data["ph"], color="blue", alpha=0.8, s=20, zorder=5)
        ax.plot(xf, y, color="royalblue")
    ax.set_title(pn["title"])
    ax.set_xlabel(opts["xlabel"])
    ax.set_ylabel(r"dpH/dV / mL$^{-1}$" if deriv else "pH")

    if opts["show_aep"]:
        for pk in cv["peaks"]:
            yv = pk["d"] if deriv else pk["ph"]
            ax.axvline(pk["v"], color="gray", linestyle="-", linewidth=0.8, alpha=0.7, zorder=1)
            ax.plot(pk["v"], yv, marker="+", color="black", mew=1.5, ms=10, zorder=6)
            ax.text(pk["v"] + 0.2, yv, fnum(pk["v"], 2, comma), fontsize=8)

    if opts["show_haep"]:
        y_range = np.max(y) - np.min(y)
        for v_half, ph_half, _, d_half in cv["half"]:
            yv = d_half if deriv else ph_half
            ax.plot(v_half, yv, marker="x", color="black", mew=1.5, ms=8, linestyle="None", zorder=6)
            ax.text(v_half + 0.2, yv - 0.02 * y_range, fnum(v_half, 2, comma),
                    fontsize=8, color="black", ha="left", va="top")

    if comma:
        fmt = FuncFormatter(lambda val, _pos: ("%g" % val).replace(".", ","))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)


def render_figure(data, panel, fmt, opts) -> bytes:
    """panel = 'all' (2×2, 10×8 in) oder ein Schlüssel aus PANELS (6×4,5 in)."""
    if panel == "all":
        fig = Figure(figsize=(10, 8))
        axs = fig.subplots(2, 2)
        for key, pn in PANELS.items():
            draw_panel(axs[pn["row"], pn["col"]], key, data, opts)
    else:
        fig = Figure(figsize=(6, 4.5))
        draw_panel(fig.subplots(), panel, data, opts)
    fig.tight_layout()

    buf = io.BytesIO()
    if fmt == "pdf":
        fig.savefig(buf, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(buf, format="jpg", bbox_inches="tight", dpi=300, pil_kwargs={"quality": 95})
    return buf.getvalue()


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def build_zip(items, panel, fmt, opts) -> bytes:
    label = PANELS[panel]["label"]
    used, buf = set(), io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in items:
            fname, n = f"{safe_name(name)}_{label}.{fmt}", 2
            while fname in used:
                fname, n = f"{safe_name(name)}_{n}_{label}.{fmt}", n + 1
            used.add(fname)
            zf.writestr(fname, render_figure(data, panel, fmt, opts))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Beispieldaten & Datei-Import
# --------------------------------------------------------------------------- #
def simulated_titration():
    """Essigsäure (0,01275 mol/L, 100 mL) mit 0,1 mol/L NaOH, 0,5-mL-Schritte, leichtes Rauschen."""
    ka, kw, ca, cb, v0 = 1.75e-5, 1e-14, 0.01275, 0.1, 100.0
    vs = np.round(np.arange(0, 25.0001, 0.5), 2)

    def ph_at(v):
        tot = v0 + v
        na, a_tot = cb * v / tot, ca * v0 / tot
        f = lambda lh: 10 ** lh + na - a_tot * ka / (ka + 10 ** lh) - kw / 10 ** lh
        return -brentq(f, -14, 0)

    rng = np.random.default_rng(7)
    ph = np.round([ph_at(v) for v in vs] + rng.normal(0, 0.01, len(vs)), 3)
    return [f"{v:g}" for v in vs], [f"{p:g}" for p in ph]


@st.cache_data(show_spinner=False)
def excel_sheet_names(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def load_table(data: bytes, filename: str, sheet) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
        try:
            return pd.read_csv(io.StringIO(text), header=None, sep=None, engine="python")
        except Exception:
            return pd.read_csv(io.StringIO(text), header=None, sep=";", engine="python")
    return pd.read_excel(io.BytesIO(data), header=None, sheet_name=sheet)


def to_num(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):  # object- und str-Spalten (Dezimalkomma als Text)
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------- #
# Session-State: Versuche (= Tabs)
# --------------------------------------------------------------------------- #
def make_base_df(v=None, ph=None) -> pd.DataFrame:
    v, ph = [str(x) for x in (v or [])], [str(x) for x in (ph or [])]
    n = EMPTY_ROWS if not v else len(v) + 20
    pad = [""] * (n - len(v))
    return pd.DataFrame({"Aktiv": [True] * n, "V [mL]": v + pad, "pH": ph + pad})


def unique_name(name: str) -> str:
    existing = {d["name"] for d in st.session_state.ds.values()}
    if name not in existing:
        return name
    k = 2
    while f"{name} ({k})" in existing:
        k += 1
    return f"{name} ({k})"


def add_dataset(name: str, v=None, ph=None) -> int:
    ss = st.session_state
    tid = ss.next_id
    ss.next_id += 1
    ss.ds[tid] = {"name": unique_name(name), "base": make_base_df(v, ph)}
    return tid


def is_empty_dataset(tid) -> bool:
    last = st.session_state.last_edit.get(tid)
    if last is None:
        return False
    return all(not cell_text(x) for x in last["V [mL]"]) and all(not cell_text(x) for x in last["pH"])


def import_datasets(items):
    """Leere Platzhalter-Tabs entfernen, dann Versuche anlegen."""
    ss = st.session_state
    for tid in [t for t in ss.ds if is_empty_dataset(t)]:
        del ss.ds[tid]
        ss.last_edit.pop(tid, None)
    for name, v, ph in items:
        add_dataset(name, v, ph)
    ss.flash = f"{len(items)} Versuch(e) importiert."


def init_state():
    ss = st.session_state
    if "ds" not in ss:
        ss.ds, ss.next_id, ss.last_edit = {}, 1, {}
        add_dataset("V1")


# --------------------------------------------------------------------------- #
# UI-Bausteine
# --------------------------------------------------------------------------- #
def render_import(file):
    data, name = file.getvalue(), file.name
    sheet = None
    if not name.lower().endswith(".csv"):
        sheets = excel_sheet_names(data)
        sheet = sheets[0] if len(sheets) == 1 else st.selectbox("Arbeitsblatt", sheets, key=f"sheet_{name}")
    try:
        df = load_table(data, name, sheet)
    except Exception as e:  # noqa: BLE001
        st.error(f"Konnte Datei nicht lesen: {e}")
        return
    letters = [col_index_to_letter(i) for i in range(df.shape[1])]
    if len(letters) < 2:
        st.warning("Die Datei braucht mindestens zwei Spalten (V und pH).")
        return

    st.markdown("**1. Vorschau der Tabelle** (erste 15 Zeilen)")
    preview = df.head(15).astype(object).where(df.head(15).notna(), "").astype(str)
    preview.columns = letters
    st.dataframe(preview, hide_index=True, width="stretch")

    st.markdown("**2. Zu importierende Spaltenpaare wählen**")
    default = [{"Import": i == 0, "V-Spalte": letters[i], "pH-Spalte": letters[i + 1]}
               for i in range(0, len(letters) - 1, 2)]
    pairs = st.data_editor(
        pd.DataFrame(default), hide_index=True, num_rows="dynamic", width="stretch",
        key=f"pairs_{name}_{sheet}",
        column_config={
            "Import": st.column_config.CheckboxColumn("Import", default=True),
            "V-Spalte": st.column_config.SelectboxColumn("V-Spalte", options=letters, required=True, default=letters[0]),
            "pH-Spalte": st.column_config.SelectboxColumn("pH-Spalte", options=letters, required=True, default=letters[1]),
        },
    )
    st.caption("Zeilen mit „+“ hinzufügen, um weitere Paare zu importieren.")

    if st.button("Ausgewählte Paare importieren", key=f"imp_{name}_{sheet}", type="primary"):
        items = []
        for _, row in pairs.iterrows():
            if pd.isna(row["Import"]) or not bool(row["Import"]):
                continue
            lv, lp = row["V-Spalte"], row["pH-Spalte"]
            if lv not in letters or lp not in letters:
                continue
            vcol = to_num(df.iloc[:, letters.index(lv)])
            pcol = to_num(df.iloc[:, letters.index(lp)])
            ok = vcol.notna() & pcol.notna()
            if ok.any():
                items.append((f"V({lv},{lp})", vcol[ok].tolist(), pcol[ok].tolist()))
        if not items:
            st.warning("Keine gültigen Spaltenpaare zum Import ausgewählt!")
        else:
            import_datasets(items)
            st.rerun()


def log_text(data, show_aep, show_haep, comma):
    lines = ["=" * 25, f"ANALYSE {data['name']} ({datetime.now():%H:%M:%S})"]
    for key, label in METHODS:
        cv = data["curves"][key]
        lines.append(f"[{label.upper()}]")
        if show_aep:
            lines.append("  -- Äquivalenzpunkte --")
            lines += [f"  V: {fnum(p['v'], 3, comma)} ml | pH: {fnum(p['ph'], 3, comma)}" for p in cv["peaks"]]
        if show_haep:
            lines.append("  -- Halbäquivalenzpunkte --")
            lines += [f"  V: {fnum(h[0], 2, comma)} ml | pH: {fnum(h[1], 2, comma)}" for h in cv["half"]]
    lines.append("=" * 25)
    return "\n".join(lines)


def result_table(data, show_aep, show_haep, comma):
    rows = []
    for key, label in METHODS:
        cv = data["curves"][key]
        if show_aep:
            rows += [{"Methode": label, "Punkt": f"ÄP {i}", "V [mL]": fnum(p["v"], 3, comma), "pH": fnum(p["ph"], 3, comma)}
                     for i, p in enumerate(cv["peaks"], 1)]
        if show_haep:
            rows += [{"Methode": label, "Punkt": f"HÄP {i}", "V [mL]": fnum(h[0], 2, comma), "pH": fnum(h[1], 2, comma)}
                     for i, h in enumerate(cv["half"], 1)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Potentiometrie Auswertung", layout="wide")
    init_state()
    ss = st.session_state

    if "flash" in ss:
        st.toast(ss.pop("flash"))

    st.title("Analyse einer potentiometrischen Titration")
    st.markdown(
        "Äquivalenzpunkt-Bestimmung per **PCHIP**- und **Spline**-Interpolation (Maximum der 1. Ableitung). "
        "Wertepaare direkt in die Tabelle einfügen (Strg+V, Dezimalkomma erlaubt) oder Excel/CSV importieren."
    )

    # ---------------- Sidebar 1: Versuche & Datenimport ---------------- #
    st.sidebar.header("1. Versuche & Datenimport")
    b1, b2 = st.sidebar.columns(2)
    if b1.button("Versuch +", width="stretch"):
        add_dataset(f"V{len(ss.ds) + 1}")
        st.rerun()
    if b2.button("Beispieldaten", width="stretch", help="Simulierte Titration Essigsäure / NaOH laden"):
        v, ph = simulated_titration()
        import_datasets([("Beispiel", v, ph)])
        st.rerun()

    with st.sidebar.expander("Versuch umbenennen / entfernen"):
        sel = st.selectbox("Versuch", list(ss.ds), format_func=lambda t: ss.ds[t]["name"], key="manage_sel")
        new_name = st.text_input("Neuer Name", value=ss.ds[sel]["name"], key=f"rename_{sel}")
        r1, r2 = st.columns(2)
        if r1.button("Umbenennen", width="stretch") and new_name.strip():
            ss.ds[sel]["name"] = "—"  # eigenen Namen freigeben, damit unique_name nicht kollidiert
            ss.ds[sel]["name"] = unique_name(new_name.strip())
            st.rerun()
        if r2.button("Entfernen", width="stretch", disabled=len(ss.ds) <= 1):
            del ss.ds[sel]
            ss.last_edit.pop(sel, None)
            st.rerun()

    uploads = st.sidebar.file_uploader(
        "Excel / CSV importieren", type=["csv", "xlsx", "xls"], accept_multiple_files=True,
        help="Spaltenpaare (V, pH) werden anschließend im Hauptbereich ausgewählt.",
    )

    # ---------------- Sidebar 2: Einstellungen & Filter ---------------- #
    st.sidebar.header("2. Einstellungen & Filter")
    show_aep = st.sidebar.checkbox("Zeige Äquivalenzpunkte (ÄP)", value=True)
    show_haep = st.sidebar.checkbox(
        "Zeige Halbäq.-punkte (HÄP)", value=False,
        help="Mitte zwischen erstem Messpunkt (bzw. vorherigem ÄP) und ÄP; pH aus der interpolierten Kurve. "
             "Nur zur Orientierung.",
    )
    c1, c2 = st.sidebar.columns(2)
    v_min_txt = c1.text_input("V min", value="", help="Volumenbereich beschneiden (leer = alles)")
    v_max_txt = c2.text_input("V max", value="")
    v_min, v_max = parse_float(v_min_txt), parse_float(v_max_txt)
    if (v_min_txt.strip() and v_min is None) or (v_max_txt.strip() and v_max is None):
        st.sidebar.warning("V min / V max ungültig – wird ignoriert.")
    smooth = st.sidebar.slider(
        "Glättung Ableitung (Savitzky-Golay-Fenster)", 11, 301, 101, step=2,
        help="Fensterbreite in Stützstellen (von 5000), Polynomgrad 3.",
    )

    # ---------------- Sidebar 3: Darstellung ---------------- #
    st.sidebar.header("3. Darstellung")
    comma = st.sidebar.checkbox("Dezimalkomma in Plots & Export", value=False)
    xlabel = st.sidebar.text_input("Beschriftung x-Achse", value="V(NaOH) / mL")

    # ---------------- Sidebar 4: Export (Inhalt folgt nach der Auswertung) ---------------- #
    export_box = st.sidebar.container()
    with export_box:
        st.header("4. Export")
        fmt = st.radio("Format", ["pdf", "jpg"], horizontal=True, format_func=str.upper, key="export_fmt")
    opts = dict(show_aep=show_aep, show_haep=show_haep, comma=comma, xlabel=xlabel)

    # ---------------- Excel-Import (Hauptbereich) ---------------- #
    for f in uploads or []:
        with st.expander(f"Import: {f.name}", expanded=True):
            render_import(f)

    # ---------------- Versuche (Tabs) ---------------- #
    results = {}
    tids = list(ss.ds)
    for tid, tab in zip(tids, st.tabs([ss.ds[t]["name"] for t in tids])):
        with tab:
            col_tab, col_plot = st.columns([1, 3])
            with col_tab:
                edited = st.data_editor(
                    ss.ds[tid]["base"], key=f"ed_{tid}", hide_index=True, num_rows="dynamic", height=520,
                    column_config={
                        "Aktiv": st.column_config.CheckboxColumn("A/I", default=True, width="small",
                                                                 help="Aktiv / Ignorieren: Haken entfernen = Zeile wird nicht ausgewertet"),
                        "V [mL]": st.column_config.TextColumn("V [mL]", width="small"),
                        "pH": st.column_config.TextColumn("pH", width="small"),
                    },
                )
                ss.last_edit[tid] = edited
                v_raw, ph_raw, skipped = collect_points(edited)
                if skipped:
                    st.caption(f"⚠️ {skipped} Zeile(n) mit unvollständigen/ungültigen Werten werden ignoriert.")

            data, err = analyze(ss.ds[tid]["name"], v_raw, ph_raw, v_min, v_max, smooth)
            results[tid] = data

            with col_plot:
                if err:
                    st.info(err)
                    continue
                st.plotly_chart(
                    build_plotly(data, show_aep, show_haep, comma, xlabel), width="stretch", key=f"plot_{tid}",
                    config={"displaylogo": False,
                            "toImageButtonOptions": {"format": "png", "scale": 3, "filename": safe_name(data["name"])}},
                )
                st.caption(
                    f"Interpolation auf {N_FINE} Stützstellen (Δ = {data['step']:.4f} mL), "
                    f"Savitzky-Golay-Fenster {data['window']} (Grad 3), "
                    f"Peak-Suche: Prominenz ≥ {PEAK_PROMINENCE} pH/mL, Mindestabstand {PEAK_DISTANCE} Stützstellen."
                )

            if err:
                continue

            # --- Ergebnisse ---
            if show_aep:
                peaks = {k: data["curves"][k]["peaks"] for k, _ in METHODS}
                # Kennzahlen: Hauptmaximum = Peak mit der größten Steigung (bei genau einem Peak dieser)
                main = {k: max(p, key=lambda q: q["d"]) for k, p in peaks.items() if p}
                if main:
                    m = st.columns(2 * len(main))
                    for i, (k, label) in enumerate((k, l) for k, l in METHODS if k in main):
                        m[2 * i].metric(f"V_ÄP ({label}) [mL]", fnum(main[k]["v"], 3, comma))
                        m[2 * i + 1].metric(f"pH am ÄP ({label})", fnum(main[k]["ph"], 3, comma))
                for k, label in METHODS:
                    if not peaks[k]:
                        st.warning(f"{label}: Kein Äquivalenzpunkt gefunden (Prominenz < {PEAK_PROMINENCE} pH/mL).")
                    elif len(peaks[k]) > 1:
                        st.caption(f"{label}: {len(peaks[k])} Maxima gefunden – Kennzahl = Hauptmaximum (größte Steigung), "
                                   "alle Maxima in der Tabelle.")
            table = result_table(data, show_aep, show_haep, comma)
            if not table.empty:
                st.dataframe(table, hide_index=True, width="stretch")
            with st.expander("Log"):
                st.code(log_text(data, show_aep, show_haep, comma), language=None)

            # --- Export (dieser Versuch) ---
            st.markdown(f"**Export dieses Versuchs ({fmt.upper()})**")
            mime = "application/pdf" if fmt == "pdf" else "image/jpeg"
            cols = st.columns(len(EXPORT_BUTTONS))
            for col, (panel, label) in zip(cols, EXPORT_BUTTONS):
                suffix = "Alle" if panel == "all" else PANELS[panel]["label"]
                col.download_button(
                    label, data=partial(render_figure, data, panel, fmt, opts),
                    file_name=f"{safe_name(data['name'])}_{suffix}.{fmt}", mime=mime,
                    key=f"dl_{tid}_{panel}_{fmt}", on_click="ignore", width="stretch",
                )

    # ---------------- Mehrfach-Export (Sidebar) ---------------- #
    valid = {tid: d for tid, d in results.items() if d}
    with export_box:
        st.markdown("**Ausgewählte Versuche (ZIP)**")
        if not valid:
            st.caption("Noch keine auswertbaren Versuche.")
        else:
            chosen = st.multiselect(
                "1. Versuche auswählen", list(valid), default=list(valid),
                format_func=lambda t: ss.ds[t]["name"], key="batch_" + "-".join(map(str, valid)),
            )
            panel = st.radio("2. Graphentyp", list(PANELS), format_func=lambda k: PANELS[k]["title"], key="batch_panel")
            st.download_button(
                f"{len(chosen)} Graph(en) als ZIP ({fmt.upper()})",
                data=partial(build_zip, [(valid[t]["name"], valid[t]) for t in chosen], panel, fmt, opts),
                file_name="Potentiometrie_Export.zip", mime="application/zip",
                key=f"zip_{fmt}", on_click="ignore", width="stretch", disabled=not chosen,
            )


if __name__ == "__main__":
    main()
