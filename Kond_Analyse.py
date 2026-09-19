import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --- Konfiguration der UI ---
st.set_page_config(page_title="Konduktometrie Auswertung", layout="wide")
st.title("Analyse einer konduktometrischen Titration")
st.markdown("Flexibler Import für CSV & Excel mit Auswahlsystem für unterschiedliche Versuche und Spalten.")

# --- Sidebar: Parameter & Dateiupload ---
st.sidebar.header("1. Datenimport & Versuche")

# Mehrere Dateien gleichzeitig erlauben
uploaded_files = st.sidebar.file_uploader(
    "Datei(en) hochladen (.csv, .xlsx, .xls)", 
    type=["csv", "xlsx", "xls"], 
    accept_multiple_files=True
)

st.sidebar.header("2. Auswertefilter (Vorgaben)")
st.sidebar.markdown("Punkte ausschließen gemäß Auswerterichtlinien:")
drop_start = st.sidebar.number_input("Punkte am Anfang (Ast 1) ignorieren", min_value=0, value=2)
drop_end = st.sidebar.number_input("Punkte am Ende (Ast 2) ignorieren", min_value=0, value=2)
drop_center = st.sidebar.number_input("Punkte um das Minimum ignorieren (pro Seite)", min_value=0, value=2)


# --- Hilfsfunktion: Beispieldaten generieren ---
@st.cache_data
def generate_dummy_data():
    x1 = np.linspace(0, 10, 20)
    y1 = 6.0 - 0.45 * x1 + np.random.normal(0, 0.05, 20)
    x2 = np.linspace(10.5, 20, 20)
    y2 = 1.5 + 0.3 * (x2 - 10) + np.random.normal(0, 0.05, 20)
    x_center = np.array([9.5, 10.0, 10.5])
    y_center = np.array([1.9, 1.4, 1.6])
    x = np.concatenate((x1[x1 < 9], x_center, x2[x2 > 11]))
    y = np.concatenate((y1[x1 < 9], y_center, y2[x2 > 11]))
    return pd.DataFrame({"Volumen [mL]": x, "Leitfähigkeit [mS/cm]": y})


# --- Einlesen der Dateien ---
datasets = {}

if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df_temp = pd.read_csv(file)
            else:
                # Excel verarbeiten
                excel_file = pd.ExcelFile(file)
                # Falls mehrere Arbeitsblätter existieren, bieten wir diese als eigene Datensätze an
                for sheet in excel_file.sheet_names:
                    df_temp = pd.read_excel(excel_file, sheet_name=sheet)
                    dataset_key = f"{file.name} [{sheet}]" if len(excel_file.sheet_names) > 1 else file.name
                    datasets[dataset_key] = df_temp
                continue
            datasets[file.name] = df_temp
        except Exception as e:
            st.error(f"Fehler beim Lesen von {file.name}: {e}")
else:
    datasets["Beispiel-Versuch"] = generate_dummy_data()
    st.info("Keine Datei hochgeladen. Es werden simulierte Beispieldaten verwendet.")


# --- Auswertung für den gewählten Versuch ---
if datasets:
    # Reiter / Tabs für die verschiedenen Versuche erstellen
    tabs = st.tabs(list(datasets.keys()))
    
    for tab, (dataset_name, df_current) in zip(tabs, datasets.items()):
        with tab:
            st.subheader(f"Datensatz: {dataset_name}")
            
            # --- Spaltenauswahl für diesen Versuch ---
            col_select1, col_select2 = st.columns(2)
            columns_list = list(df_current.columns)
            
            # Intelligente Vorauswahl der Spalten (Standard: 0 für X, 1 für Y)
            default_x_idx = 0 if len(columns_list) > 0 else 0
            default_y_idx = 1 if len(columns_list) > 1 else 0
            
            with col_select1:
                x_col = st.selectbox(
                    "Spalte für Volumen (X-Achse)", 
                    options=columns_list, 
                    index=default_x_idx,
                    key=f"x_{dataset_name}"
                )
            with col_select2:
                y_col = st.selectbox(
                    "Spalte für Leitfähigkeit (Y-Achse)", 
                    options=columns_list, 
                    index=default_y_idx,
                    key=f"y_{dataset_name}"
                )

            # --- Datenbereinigung & Konvertierung ---
            # Nur gültige numerische Werte berücksichtigen
            df_clean = df_current[[x_col, y_col]].dropna()
            df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
            df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
            df_clean = df_clean.dropna().sort_values(by=x_col)

            x = df_clean[x_col].values.reshape(-1, 1)
            y = df_clean[y_col].values

            if len(x) < 5:
                st.warning("Zu wenige gültige numerische Datenpunkte in den gewählten Spalten.")
                continue

            # --- Titrationsauswertung ---
            min_idx = np.argmin(y)

            # Linker Ast
            left_end_idx = max(0, min_idx - drop_center)
            x_left_full = x[drop_start:left_end_idx]
            y_left_full = y[drop_start:left_end_idx]

            # Rechter Ast
            right_start_idx = min(len(x), min_idx + drop_center + 1)
            x_right_full = x[right_start_idx : max(0, len(x) - drop_end)]
            y_right_full = y[right_start_idx : max(0, len(x) - drop_end)]

            # Linear Regression
            eq_x, eq_y = None, None
            if len(x_left_full) > 1 and len(x_right_full) > 1:
                reg_left = LinearRegression().fit(x_left_full, y_left_full)
                reg_right = LinearRegression().fit(x_right_full, y_right_full)
                
                m1, b1 = reg_left.coef_[0], reg_left.intercept_
                m2, b2 = reg_right.coef_[0], reg_right.intercept_
                
                if m1 != m2:
                    eq_x = (b2 - b1) / (m1 - m2)
                    eq_y = m1 * eq_x + b1

            # --- Graphische Darstellung ---
            fig = go.Figure()

            # Alle Messwerte
            fig.add_trace(go.Scatter(
                x=df_clean[x_col], y=df_clean[y_col],
                mode='markers', name='Alle Messwerte',
                marker=dict(color='lightgray', size=8, line=dict(color='gray', width=1))
            ))

            # Verwendete Punkte Ast 1
            fig.add_trace(go.Scatter(
                x=x_left_full.flatten(), y=y_left_full,
                mode='markers', name='Auswertebereich Ast 1',
                marker=dict(color='#1f77b4', size=10)
            ))

            # Verwendete Punkte Ast 2
            fig.add_trace(go.Scatter(
                x=x_right_full.flatten(), y=y_right_full,
                mode='markers', name='Auswertebereich Ast 2',
                marker=dict(color='#2ca02c', size=10)
            ))

            # Regressionsgeraden & Äquivalenzpunkt
            if eq_x is not None:
                x_plot_left = np.array([df_clean[x_col].min(), eq_x]).reshape(-1, 1)
                x_plot_right = np.array([eq_x, df_clean[x_col].max()]).reshape(-1, 1)

                fig.add_trace(go.Scatter(
                    x=x_plot_left.flatten(), y=reg_left.predict(x_plot_left),
                    mode='lines', name=f'Ast 1 (R²={reg_left.score(x_left_full, y_left_full):.4f})',
                    line=dict(color='#1f77b4', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=x_plot_right.flatten(), y=reg_right.predict(x_plot_right),
                    mode='lines', name=f'Ast 2 (R²={reg_right.score(x_right_full, y_right_full):.4f})',
                    line=dict(color='#2ca02c', width=2, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=[eq_x], y=[eq_y],
                    mode='markers', name='Äquivalenzpunkt',
                    marker=dict(color='red', size=14, symbol='x')
                ))

            fig.update_layout(
                xaxis_title=str(x_col),
                yaxis_title=str(y_col),
                hovermode="x unified",
                template="plotly_white",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )

            st.plotly_chart(fig, width='stretch')

            # --- Ergebnisse ---
            if eq_x is not None:
                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Äquivalenzvolumen (V_eq)", f"{eq_x:.3f}")
                res_col2.metric("Leitfähigkeit am Äq. Punkt", f"{eq_y:.3f}")
            else:
                st.error("Äquivalenzpunkt konnte mit den aktuellen Auswertefiltern nicht berechnet werden.")