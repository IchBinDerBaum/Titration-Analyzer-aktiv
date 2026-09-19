import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import pandas as pd

def col_index_to_letter(col_idx):
    result = ""
    col_idx += 1
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        result = chr(65 + remainder) + result
    return result

class CustomToolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] != 'Save']


class ExcelTable(ttk.Frame):
    def __init__(self, parent, rows=120):
        super().__init__(parent)
        self.cells = []
        self.canvas = tk.Canvas(self, width=170, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Tabellenkopf mit Checkbox-Spalte für A/I (Aktiv/Ignorieren)
        ttk.Label(self.scroll_frame, text="A/I", font=('Arial', 8, 'bold')).grid(row=0, column=0, padx=1)
        ttk.Label(self.scroll_frame, text="V [ml]", font=('Arial', 8, 'bold')).grid(row=0, column=1)
        ttk.Label(self.scroll_frame, text="pH", font=('Arial', 8, 'bold')).grid(row=0, column=2)

        for _ in range(rows): 
            self.add_row()
            
        self.canvas.pack(side="left", fill="y", expand=False)
        self.scrollbar.pack(side="right", fill="y")

        self.bind_mouse_wheel(self.canvas)
        self.bind_mouse_wheel(self.scroll_frame)

    def bind_mouse_wheel(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-2, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(2, "units")

    def add_row(self):
        r = len(self.cells) + 1
        active_var = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(self.scroll_frame, variable=active_var)
        v_entry = ttk.Entry(self.scroll_frame, width=7)
        p_entry = ttk.Entry(self.scroll_frame, width=7)

        chk.grid(row=r, column=0, padx=1, pady=1)
        v_entry.grid(row=r, column=1, padx=1, pady=1)
        p_entry.grid(row=r, column=2, padx=1, pady=1)
        self.cells.append((active_var, v_entry, p_entry))

        for entry in (v_entry, p_entry):
            entry.bind("<Control-v>", self.handle_paste)
            entry.bind("<Delete>", self.handle_delete)
            entry.bind("<BackSpace>", self.handle_delete)
            self.bind_mouse_wheel(entry)

    def handle_delete(self, event):
        focused = self.focus_get()
        try:
            if focused.selection_get():
                return
        except: pass
        if isinstance(focused, ttk.Entry):
            focused.delete(0, tk.END)
            return "break"

    def handle_paste(self, event):
        try:
            clipboard = self.focus_get().selection_get(selection='CLIPBOARD')
            rows_data = clipboard.split('\n')
            focused = self.focus_get()
            start_row, start_col = 0, 0
            for i, (_, v, p) in enumerate(self.cells):
                if v == focused: start_row, start_col = i, 0; break
                if p == focused: start_row, start_col = i, 1; break
            for r_off, row_str in enumerate(rows_data):
                if not row_str.strip(): continue
                while (start_row + r_off) >= len(self.cells): self.add_row()
                cols = row_str.split('\t')
                for c_off, val in enumerate(cols):
                    t_row, t_col = start_row + r_off, start_col + c_off
                    if t_col < 2:
                        entry = self.cells[t_row][t_col + 1]
                        entry.delete(0, tk.END); entry.insert(0, val.strip())
            return "break"
        except: pass

    def load_data(self, v_data, ph_data):
        for var, entry_v, entry_ph in self.cells:
            var.set(True)
            entry_v.delete(0, tk.END)
            entry_ph.delete(0, tk.END)
            
        while len(self.cells) < len(v_data):
            self.add_row()
            
        for i, (v, ph) in enumerate(zip(v_data, ph_data)):
            self.cells[i][0].set(True)
            self.cells[i][1].insert(0, str(v))
            self.cells[i][2].insert(0, str(ph))


class MultiPairImportDialog(tk.Toplevel):
    def __init__(self, parent, df, import_callback):
        super().__init__(parent)
        self.title("Excel Spaltenpaare importieren")
        self.geometry("750x650")
        self.grab_set()

        self.df = df
        self.import_callback = import_callback
        self.pair_rows = []

        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=10)
        ttk.Button(btn_frame, text="Ausgewählte Paare importieren", command=self._on_import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Abbrechen", command=self.destroy).pack(side=tk.RIGHT)

        btn_ctrl_frame = ttk.Frame(self)
        btn_ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=2)
        ttk.Button(btn_ctrl_frame, text="+ Weitere Paare hinzufügen", command=self.add_pair_row).pack(side=tk.LEFT)

        ttk.Label(self, text="2. Zu importierende Spaltenpaare wählen:", font=('Arial', 9, 'bold')).pack(side=tk.BOTTOM, anchor="w", padx=15, pady=(10, 2))

        container_frame = ttk.Frame(self)
        container_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=15, pady=2)

        self.scroll_canvas = tk.Canvas(container_frame, highlightthickness=0)
        self.scroll_frame = ttk.Frame(self.scroll_canvas)
        scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=self.scroll_canvas.yview)
        
        self.scroll_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        excel_cols_raw = [col_index_to_letter(i) for i in range(len(df.columns))]
        for i in range(0, len(excel_cols_raw) - 1, 2):
            self.add_pair_row(i, i + 1, active=(i == 0))

        ttk.Label(self, text="1. Vorschau der Excel-Tabelle:", font=('Arial', 9, 'bold')).pack(anchor="w", padx=15, pady=(10, 2))

        preview_frame = ttk.Frame(self)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=2)

        excel_cols = [col_index_to_letter(i) for i in range(len(df.columns))]
        self.tree = ttk.Treeview(preview_frame, columns=excel_cols, show="headings", height=7)
        
        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for col_letter in excel_cols:
            self.tree.heading(col_letter, text=col_letter)
            self.tree.column(col_letter, width=60, anchor="center")

        for _, row in df.head(15).iterrows():
            row_vals = [str(val) if pd.notna(val) else "" for val in row]
            self.tree.insert("", "end", values=row_vals)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

    def add_pair_row(self, default_v=0, default_ph=1, active=True):
        row_frame = ttk.Frame(self.scroll_frame)
        row_frame.pack(fill=tk.X, pady=2)

        var_active = tk.BooleanVar(value=active)
        ttk.Checkbutton(row_frame, variable=var_active).pack(side=tk.LEFT, padx=5)

        excel_cols = [f"Spalte {col_index_to_letter(i)}" for i in range(len(self.df.columns))]

        ttk.Label(row_frame, text="V:").pack(side=tk.LEFT)
        cb_v = ttk.Combobox(row_frame, values=excel_cols, state="readonly", width=12)
        cb_v.pack(side=tk.LEFT, padx=5)
        if default_v < len(excel_cols): cb_v.current(default_v)

        ttk.Label(row_frame, text="pH:").pack(side=tk.LEFT)
        cb_ph = ttk.Combobox(row_frame, values=excel_cols, state="readonly", width=12)
        cb_ph.pack(side=tk.LEFT, padx=5)
        if default_ph < len(excel_cols): cb_ph.current(default_ph)

        self.pair_rows.append((var_active, cb_v, cb_ph))

    def _on_import(self):
        imported_pairs = []
        for var_active, cb_v, cb_ph in self.pair_rows:
            if var_active.get():
                idx_v = cb_v.current()
                idx_ph = cb_ph.current()
                if idx_v != -1 and idx_ph != -1:
                    v_col = pd.to_numeric(self.df.iloc[:, idx_v], errors='coerce')
                    ph_col = pd.to_numeric(self.df.iloc[:, idx_ph], errors='coerce')

                    valid_mask = v_col.notna() & ph_col.notna()
                    v_clean = v_col[valid_mask].tolist()
                    ph_clean = ph_col[valid_mask].tolist()

                    if v_clean and ph_clean:
                        label_v = col_index_to_letter(idx_v)
                        label_ph = col_index_to_letter(idx_ph)
                        imported_pairs.append((f"V({label_v},{label_ph})", v_clean, ph_clean))

        if not imported_pairs:
            messagebox.showwarning("Hinweis", "Keine gültigen Spaltenpaare zum Import ausgewählt!")
            return

        self.import_callback(imported_pairs)
        self.destroy()


class FormatDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Exportformat")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ttk.Label(self, text="Format wählen:", font=('Arial', 10, 'bold')).pack(pady=(16, 8), padx=20)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(0, 16), padx=20)

        ttk.Button(btn_frame, text="PDF", width=10, command=lambda: self._choose("pdf")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="JPG", width=10, command=lambda: self._choose("jpg")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Abbrechen", width=10, command=self.destroy).pack(side=tk.LEFT, padx=6)

        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")
        self.wait_window()

    def _choose(self, fmt):
        self.result = fmt
        self.destroy()


class MultiExportDialog(tk.Toplevel):
    def __init__(self, parent, tabs):
        super().__init__(parent)
        self.title("Mehrfach-Export")
        self.resizable(False, False)
        self.grab_set()

        self.selected_tabs = []
        self.selected_type = tk.IntVar(value=0)
        self.selected_fmt = tk.StringVar(value="jpg")
        self.confirmed = False

        ttk.Label(self, text="1. Versuche auswählen:", font=('Arial', 9, 'bold')).pack(anchor="w", padx=15, pady=(10, 5))
        tab_frame = ttk.Frame(self)
        tab_frame.pack(fill=tk.X, padx=15, pady=2)

        self.tab_vars = {}
        for tab_id, name in tabs:
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(tab_frame, text=name, variable=var).pack(anchor="w")
            self.tab_vars[tab_id] = (name, var)

        ttk.Label(self, text="2. Graphentyp auswählen:", font=('Arial', 9, 'bold')).pack(anchor="w", padx=15, pady=(10, 5))
        type_frame = ttk.Frame(self)
        type_frame.pack(fill=tk.X, padx=15, pady=2)
        ttk.Radiobutton(type_frame, text="PCHIP Kurve", variable=self.selected_type, value=0).pack(anchor="w")
        ttk.Radiobutton(type_frame, text="Spline Kurve", variable=self.selected_type, value=1).pack(anchor="w")
        ttk.Radiobutton(type_frame, text="PCHIP 1. Ableitung", variable=self.selected_type, value=2).pack(anchor="w")
        ttk.Radiobutton(type_frame, text="Spline 1. Ableitung", variable=self.selected_type, value=3).pack(anchor="w")

        ttk.Label(self, text="3. Format auswählen:", font=('Arial', 9, 'bold')).pack(anchor="w", padx=15, pady=(10, 5))
        fmt_frame = ttk.Frame(self)
        fmt_frame.pack(fill=tk.X, padx=15, pady=2)
        ttk.Radiobutton(fmt_frame, text="JPG", variable=self.selected_fmt, value="jpg").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(fmt_frame, text="PDF", variable=self.selected_fmt, value="pdf").pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        ttk.Button(btn_frame, text="Exportieren", command=self._on_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Abbrechen", command=self.destroy).pack(side=tk.RIGHT)

        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")
        self.wait_window()

    def _on_export(self):
        self.selected_tabs = [(tid, name) for tid, (name, var) in self.tab_vars.items() if var.get()]
        if not self.selected_tabs:
            messagebox.showwarning("Hinweis", "Bitte mindestens einen Versuch auswählen!")
            return
        self.confirmed = True
        self.destroy()


class TitrationAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titration Analyzer Pro")
        self.root.state('zoomed')
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.show_aep = tk.BooleanVar(value=True)
        self.show_haep = tk.BooleanVar(value=False)
        self.smooth_window = tk.IntVar(value=101)

        # NEU: Variablen für die manuelle Volumen-Beschneidung
        self.v_min_str = tk.StringVar(value="")
        self.v_max_str = tk.StringVar(value="")

        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=4, bg="#d9d9d9")
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.paned_window, width=230)
        self.sidebar.pack_propagate(False)
        self.paned_window.add(self.sidebar)

        self.plot_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.plot_frame)

        self.setup_sidebar()
        self.setup_plot_area()

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        os._exit(0)

    def setup_sidebar(self):
        ctrl = ttk.Frame(self.sidebar)
        ctrl.pack(fill=tk.X, pady=(5, 0), padx=5)
        ttk.Button(ctrl, text="Tab +", width=5, command=self.add_tab).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(ctrl, text="Tab -", width=5, command=self.remove_tab).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Excel Import", command=self.open_excel_import).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self.sidebar)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.notebook.bind('<Double-Button-1>', self.rename_tab)
        self.add_tab()

        set_frame = ttk.LabelFrame(self.sidebar, text="Einstellungen & Filter")
        set_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(set_frame, text="Zeige Äquivalenzpunkte (ÄP)", variable=self.show_aep, 
                        command=self.analyze_current).pack(anchor="w", padx=5, pady=2)
        ttk.Checkbutton(set_frame, text="Zeige Halbäq.-punkte (HÄP)", variable=self.show_haep, 
                        command=self.analyze_current).pack(anchor="w", padx=5, pady=2)
        
        # NEU: Eingabefelder zur Beschneidung des Volumen-Bereichs
        crop_frame = ttk.Frame(set_frame)
        crop_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(crop_frame, text="V min:").grid(row=0, column=0, sticky="w")
        ttk.Entry(crop_frame, textvariable=self.v_min_str, width=6).grid(row=0, column=1, padx=(2, 8))
        
        ttk.Label(crop_frame, text="V max:").grid(row=0, column=2, sticky="w")
        ttk.Entry(crop_frame, textvariable=self.v_max_str, width=6).grid(row=0, column=3, padx=(2, 0))

        ttk.Label(set_frame, text="Glättung Ableitung:").pack(anchor="w", padx=5, pady=(5,0))
        slider = ttk.Scale(set_frame, from_=11, to=301, variable=self.smooth_window, 
                           command=lambda _: self.analyze_current())
        slider.pack(fill=tk.X, padx=5, pady=(0,5))

        ttk.Button(self.sidebar, text="ANALYSIEREN", command=self.analyze_current).pack(fill=tk.X, pady=5, padx=5)

        self.log_text = tk.Text(self.sidebar, height=13, font=("Consolas", 8), bg="#f8f8f8")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        log_scroll = ttk.Scrollbar(self.log_text, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")

        exp_frame = ttk.LabelFrame(self.sidebar, text="Export")
        exp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_grid = ttk.Frame(exp_frame)
        btn_grid.pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(btn_grid, text="Alle", command=lambda: self.save_plot("all")).grid(row=0, column=0, sticky="ew", padx=1)
        ttk.Button(btn_grid, text="PCHIP", command=lambda: self.save_plot(0)).grid(row=0, column=1, sticky="ew", padx=1)
        ttk.Button(btn_grid, text="Spline", command=lambda: self.save_plot(1)).grid(row=0, column=2, sticky="ew", padx=1)
        btn_grid.columnconfigure(0, weight=1); btn_grid.columnconfigure(1, weight=1); btn_grid.columnconfigure(2, weight=1)
        
        ttk.Button(exp_frame, text="Ausgewählte Tabs...", command=self.export_batch).pack(fill=tk.X, pady=2, padx=3)

    def setup_plot_area(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        
        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = CustomToolbar(self.canvas_agg, toolbar_frame)
        self.toolbar.update()
        
        self.canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_tab(self, name=None):
        t_id = len(self.notebook.tabs()) + 1
        tab_title = name if name else f"V{t_id}"
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=tab_title)
        table = ExcelTable(frame)
        table.pack(fill=tk.BOTH, expand=True)
        frame.table = table
        self.notebook.select(frame)
        return frame

    def remove_tab(self):
        if len(self.notebook.tabs()) > 1:
            self.notebook.forget("current")

    def rename_tab(self, event):
        tab_id = self.notebook.select()
        old_name = self.notebook.tab(tab_id, "text")
        new_name = simpledialog.askstring("Tab umbenennen", "Neuer Name:", initialvalue=old_name, parent=self.root)
        if new_name:
            self.notebook.tab(tab_id, text=new_name)

    def open_excel_import(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv")])
        if not filepath: return
        try:
            if filepath.lower().endswith('.csv'):
                df = pd.read_csv(filepath, header=None, sep=None, engine='python')
            else:
                df = pd.read_excel(filepath, header=None)
            
            MultiPairImportDialog(self.root, df, self._handle_multi_import)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Datei nicht lesen:\n{e}")

    def _handle_multi_import(self, imported_pairs):
        for name, v_data, ph_data in imported_pairs:
            frame = self.add_tab(name=name)
            frame.table.load_data(v_data, ph_data)
        self.analyze_current()

    def find_half_equivalence_points(self, x, y, peak_indices):
        half_points = []
        prev_v = x[0]
        for pk_idx in peak_indices:
            v_aep   = x[pk_idx]
            v_half  = (prev_v + v_aep) / 2.0
            idx_half = np.argmin(np.abs(x - v_half))
            ph_half = y[idx_half]
            half_points.append((v_half, ph_half, idx_half))
            prev_v = v_aep
        return half_points

    def analyze_tab(self, tab_id):
        tab_name = self.notebook.tab(tab_id, "text")
        tab = self.notebook.nametowidget(tab_id)

        v_raw, p_raw = [], []
        for is_active, v_e, p_e in tab.table.cells:
            # NEU: Nur Punkte beachten, bei denen die Checkbox gesetzt ist
            if not is_active.get():
                continue
            vs, ps = v_e.get().replace(',', '.'), p_e.get().replace(',', '.')
            if vs and ps:
                try: v_raw.append(float(vs)); p_raw.append(float(ps))
                except: continue

        if len(v_raw) < 5: return None
        v, ph = np.array(v_raw), np.array(p_raw)
        idx = np.argsort(v); v, ph = v[idx], ph[idx]

        # NEU: Volumen-Bereich filtern, falls V min / V max angegeben sind
        try:
            v_min_val = float(self.v_min_str.get().replace(',', '.'))
            mask = v >= v_min_val
            v, ph = v[mask], ph[mask]
        except ValueError:
            pass

        try:
            v_max_val = float(self.v_max_str.get().replace(',', '.'))
            mask = v <= v_max_val
            v, ph = v[mask], ph[mask]
        except ValueError:
            pass

        if len(v) < 5: return None

        xf = np.linspace(v.min(), v.max(), 5000)

        wl = self.smooth_window.get()
        if wl >= len(xf):
            wl = len(xf) - 1 if (len(xf) % 2 == 0) else len(xf) - 2
        if wl % 2 == 0: wl += 1
        if wl < 3: wl = 3
        
        p_i = PchipInterpolator(v, ph); yp = p_i(xf); dp = savgol_filter(np.gradient(yp, xf), wl, 3)
        s_i = CubicSpline(v, ph);       ys = s_i(xf); ds = savgol_filter(s_i.derivative()(xf), wl, 3)

        return {
            "name": tab_name, "v": v, "ph": ph, "xf": xf, 
            "yp": yp, "dp": dp, "ys": ys, "ds": ds
        }

    def render_analysis(self, data):
        if not data: return
        
        tab_name = data["name"]
        v, ph, xf = data["v"], data["ph"], data["xf"]
        yp, dp, ys, ds = data["yp"], data["dp"], data["ys"], data["ds"]

        for ax in self.axs.flat:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"\n{'='*25}\n")
        self.log_text.insert(tk.END, f"ANALYSE {tab_name} ({timestamp})\n", "header")
        self.log_text.tag_config("header", font=("Consolas", 8, "bold"), foreground="blue")

        c_titr, c_deriv = 'royalblue', 'crimson'

        self.axs[0,0].scatter(v, ph, color='blue', alpha=0.8, s=20, zorder=5)
        self.axs[0,0].plot(xf, yp, color=c_titr)
        self.axs[0,0].set_title("PCHIP Kurve")
        
        self.axs[1,0].plot(xf, dp, color=c_deriv)
        self.axs[1,0].set_title("PCHIP 1. Ableitung")
        
        self.axs[0,1].scatter(v, ph, color='blue', alpha=0.8, s=20, zorder=5)
        self.axs[0,1].plot(xf, ys, color=c_titr)
        self.axs[0,1].set_title("Spline Kurve")
        
        self.axs[1,1].plot(xf, ds, color=c_deriv)
        self.axs[1,1].set_title("Spline 1. Ableitung")

        show_aep = self.show_aep.get()
        show_haep = self.show_haep.get()

        for name, x, y, d, axk, axa in [("PCHIP",  xf, yp, dp, self.axs[0,0], self.axs[1,0]),
                                         ("SPLINE", xf, ys, ds, self.axs[0,1], self.axs[1,1])]:
            pks, _ = find_peaks(d, prominence=0.3, distance=200)
            self.log_text.insert(tk.END, f"[{name}]\n")

            if show_aep:
                self.log_text.insert(tk.END, "  -- Äquivalenzpunkte --\n")
                for p in pks:
                    y_min_k, y_max_k = axk.get_ylim()
                    axk.plot([x[p], x[p]], [-1000, 1000], color='gray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)
                    axk.set_ylim((y_min_k, y_max_k)) 

                    y_min_a, y_max_a = axa.get_ylim()
                    axa.plot([x[p], x[p]], [-1000, 1000], color='gray', linestyle='-', linewidth=0.8, alpha=0.7, zorder=1)
                    axa.set_ylim((y_min_a, y_max_a))
                    
                    axk.plot(x[p], y[p], marker='+', color='black', mew=1.5, ms=10, zorder=6)
                    axa.plot(x[p], d[p], marker='+', color='black', mew=1.5, ms=10, zorder=6)
                    axk.text(x[p]+0.2, y[p], f"{x[p]:.2f}", fontsize=8)
                    axa.text(x[p]+0.2, d[p], f"{x[p]:.2f}", fontsize=8)
                    self.log_text.insert(tk.END, f"  V: {x[p]:.3f} ml | pH: {y[p]:.3f}\n")

            if show_haep:
                y_range_k = np.max(y) - np.min(y)
                y_range_a = np.max(d) - np.min(d)
                half_points = self.find_half_equivalence_points(x, y, pks)
                self.log_text.insert(tk.END, "  -- Halbäquivalenzpunkte --\n")
                for (v_half, ph_half, idx_half) in half_points:
                    axk.plot(v_half, ph_half, marker='x', color='black', mew=1.5, ms=8, linestyle='None', zorder=6)
                    axk.text(v_half + 0.2, ph_half - 0.02 * y_range_k, f"{v_half:.2f}", fontsize=8, color='black', ha='left', va='top')
                    
                    axa.plot(v_half, d[idx_half], marker='x', color='black', mew=1.5, ms=8, linestyle='None', zorder=6)
                    axa.text(v_half + 0.2, d[idx_half] - 0.02 * y_range_a, f"{v_half:.2f}", fontsize=8, color='black', ha='left', va='top')
                    
                    self.log_text.insert(tk.END, f"  V: {v_half:.2f} ml | pH: {ph_half:.2f}\n")

        self.log_text.insert(tk.END, f"{'='*25}\n")
        self.log_text.see(tk.END)
        self.fig.tight_layout(); self.canvas_agg.draw()

    def analyze_current(self):
        tab_id = self.notebook.select()
        data = self.analyze_tab(tab_id)
        if data:
            self.render_analysis(data)

    def save_plot(self, mode):
        dialog = FormatDialog(self.root)
        fmt = dialog.result
        if fmt is None:
            return

        if fmt == "pdf":
            ext = ".pdf"
            filetypes = [("PDF-Dokument", "*.pdf")]
            save_kwargs = {}
        else:
            ext = ".jpg"
            filetypes = [("JPEG-Bild", "*.jpg")]
            save_kwargs = {"dpi": 300, "quality": 95}

        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=filetypes)
        if not path:
            return

        try:
            self.canvas_agg.draw()
            if mode == "all":
                self.fig.savefig(path, bbox_inches='tight', **save_kwargs)
            else:
                ax_source = self.axs[0 if mode < 2 else 1, mode % 2]
                self._save_single_axis(ax_source, path, fmt, save_kwargs)

            messagebox.showinfo("Erfolg", f"Datei gespeichert als {fmt.upper()}!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen:\n{e}")

    def _save_single_axis(self, ax_source, path, fmt, save_kwargs):
        fig_single, ax_single = plt.subplots(figsize=(6, 4.5))
        
        for line in ax_source.get_lines():
            ax_single.plot(
                line.get_xdata(), line.get_ydata(), color=line.get_color(),
                linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                marker=line.get_marker(), markersize=line.get_markersize(),
                markeredgewidth=line.get_markeredgewidth(), markeredgecolor=line.get_markeredgecolor(),
                markerfacecolor=line.get_markerfacecolor(), alpha=line.get_alpha(), zorder=line.get_zorder()
            )
        
        for collection in ax_source.collections:
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                facecolors = collection.get_facecolors()
                ax_single.scatter(
                    offsets[:, 0], offsets[:, 1], s=collection.get_sizes(),
                    color=facecolors if len(facecolors) > 0 else 'blue',
                    alpha=collection.get_alpha(), zorder=collection.get_zorder()
                )

        for text in ax_source.texts:
            bbox = text.get_bbox_patch()
            bbox_kwargs = {"bbox": dict(facecolor=bbox.get_facecolor(), edgecolor=bbox.get_edgecolor(), alpha=bbox.get_alpha(), pad=2)} if bbox else {}
            ax_single.text(
                text.get_position()[0], text.get_position()[1], text.get_text(), 
                fontsize=text.get_fontsize(), color=text.get_color(),
                ha=text.get_horizontalalignment(), va=text.get_verticalalignment(),
                zorder=text.get_zorder(), **bbox_kwargs
            )

        ax_single.set_title(ax_source.get_title())
        ax_single.set_xlabel(ax_source.get_xlabel())
        ax_single.set_ylabel(ax_source.get_ylabel())
        ax_single.set_xlim(ax_source.get_xlim())
        ax_single.set_ylim(ax_source.get_ylim())
        ax_single.grid(True, linestyle='--', alpha=0.6)

        fig_single.tight_layout()
        fig_single.savefig(path, bbox_inches='tight', **save_kwargs)
        plt.close(fig_single)

    def export_batch(self):
        tabs = [(tid, self.notebook.tab(tid, "text")) for tid in self.notebook.tabs()]
        dialog = MultiExportDialog(self.root, tabs)
        
        if not dialog.confirmed: return

        export_dir = filedialog.askdirectory(title="Zielordner für Export wählen")
        if not export_dir: return

        target_type = dialog.selected_type.get()
        fmt = dialog.selected_fmt.get()
        
        type_mapping = {
            0: (0, 0, "PCHIP_Kurve"),
            1: (0, 1, "Spline_Kurve"),
            2: (1, 0, "PCHIP_1_Ableitung"),
            3: (1, 1, "Spline_1_Ableitung")
        }
        row, col, label = type_mapping[target_type]
        save_kwargs = {"dpi": 300, "quality": 95} if fmt == "jpg" else {}

        exported_count = 0
        for tab_id, tab_name in dialog.selected_tabs:
            data = self.analyze_tab(tab_id)
            if not data: continue
            
            self.render_analysis(data)
            ax_target = self.axs[row, col]

            safe_name = "".join([c if c.isalnum() else "_" for c in tab_name])
            filename = f"{safe_name}_{label}.{fmt}"
            full_path = os.path.join(export_dir, filename)

            self._save_single_axis(ax_target, full_path, fmt, save_kwargs)
            exported_count += 1

        if exported_count > 0:
            messagebox.showinfo("Erfolg", f"{exported_count} Graphen erfolgreich in\n{export_dir}\ngespeichert!")
        else:
            messagebox.showwarning("Fehler", "Keine gültigen Daten in den ausgewählten Tabs vorhanden.")


if __name__ == "__main__":
    root = tk.Tk(); app = TitrationAnalyzerApp(root); root.mainloop()
