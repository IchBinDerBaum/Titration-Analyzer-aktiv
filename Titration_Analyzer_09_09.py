import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime

class ExcelTable(ttk.Frame):
    def __init__(self, parent, rows=120):
        super().__init__(parent)
        self.cells = []
        self.canvas = tk.Canvas(self, width=130, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        ttk.Label(self.scroll_frame, text="V [ml]", font=('Arial', 8, 'bold')).grid(row=0, column=0)
        ttk.Label(self.scroll_frame, text="pH", font=('Arial', 8, 'bold')).grid(row=0, column=1)

        for _ in range(rows): self.add_row()
        self.canvas.pack(side="left", fill="y", expand=False)
        self.scrollbar.pack(side="right", fill="y")

    def add_row(self):
        r = len(self.cells) + 1
        v_entry = ttk.Entry(self.scroll_frame, width=7)
        p_entry = ttk.Entry(self.scroll_frame, width=7)
        v_entry.grid(row=r, column=0, padx=1, pady=1)
        p_entry.grid(row=r, column=1, padx=1, pady=1)
        self.cells.append((v_entry, p_entry))
        v_entry.bind("<Control-v>", self.handle_paste)
        p_entry.bind("<Control-v>", self.handle_paste)

    def handle_paste(self, event):
        try:
            clipboard = self.focus_get().selection_get(selection='CLIPBOARD')
            rows_data = clipboard.split('\n')
            focused = self.focus_get()
            start_row, start_col = 0, 0
            for i, (v, p) in enumerate(self.cells):
                if v == focused: start_row, start_col = i, 0; break
                if p == focused: start_row, start_col = i, 1; break
            for r_off, row_str in enumerate(rows_data):
                if not row_str.strip(): continue
                while (start_row + r_off) >= len(self.cells): self.add_row()
                cols = row_str.split('\t')
                for c_off, val in enumerate(cols):
                    t_row, t_col = start_row + r_off, start_col + c_off
                    if t_col < 2:
                        entry = self.cells[t_row][t_col]
                        entry.delete(0, tk.END); entry.insert(0, val.strip())
            return "break"
        except: pass


class FormatDialog(tk.Toplevel):
    """Kleines Dialogfenster zur Auswahl des Exportformats."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Exportformat")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ttk.Label(self, text="Format waehlen:", font=('Arial', 10, 'bold')).pack(pady=(16, 8), padx=20)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(0, 16), padx=20)

        ttk.Button(btn_frame, text="PDF", width=10,
                   command=lambda: self._choose("pdf")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="JPG", width=10,
                   command=lambda: self._choose("jpg")).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Abbrechen", width=10,
                   command=self.destroy).pack(side=tk.LEFT, padx=6)

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")
        self.wait_window()

    def _choose(self, fmt):
        self.result = fmt
        self.destroy()


class TitrationAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Titration Analyzer")
        self.root.state('zoomed')

        # --- NEU: Klick auf das X abfangen ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=4, bg="#cccccc")
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.paned_window, width=180)
        self.sidebar.pack_propagate(False)
        self.paned_window.add(self.sidebar)

        self.plot_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.plot_frame)

        self.setup_sidebar()
        self.setup_plot_area()

    def on_closing(self):
        """Beendet das Programm sauber und killt alle Hintergrundprozesse."""
        plt.close('all')      # Schließt alle unsichtbaren Matplotlib-Graphen
        self.root.quit()      # Stoppt die Tkinter-Hauptschleife
        self.root.destroy()   # Zerstört das Fenster
        sys.exit()            # Beendet den Python-Prozess komplett

    def setup_sidebar(self):
        ctrl = ttk.Frame(self.sidebar); ctrl.pack(fill=tk.X, pady=5)
        ttk.Button(ctrl, text="+", width=3, command=self.add_tab).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="-", width=3, command=self.remove_tab).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self.sidebar)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5)
        self.add_tab()

        ttk.Button(self.sidebar, text="ANALYSIEREN", command=self.analyze_current).pack(fill=tk.X, pady=5, padx=5)

        self.log_text = tk.Text(self.sidebar, height=22, font=("Consolas", 8), bg="#f8f8f8")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        log_scroll = ttk.Scrollbar(self.log_text, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")

        exp_frame = ttk.LabelFrame(self.sidebar, text="Export")
        exp_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(exp_frame, text="Alle 4 Graphen", command=lambda: self.save_plot("all")).pack(fill=tk.X)
        ttk.Button(exp_frame, text="PCHIP Kurve",    command=lambda: self.save_plot(0)).pack(fill=tk.X)
        ttk.Button(exp_frame, text="Spline Kurve",   command=lambda: self.save_plot(1)).pack(fill=tk.X)

    def setup_plot_area(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas_agg = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_agg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_tab(self):
        t_id = len(self.notebook.tabs()) + 1
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text=f"V{t_id}")
        table = ExcelTable(frame); table.pack(fill=tk.BOTH, expand=True); frame.table = table

    def remove_tab(self):
        if len(self.notebook.tabs()) > 1: self.notebook.forget("current")

    # ------------------------------------------------------------------ #
    #  NEU: Berechnung der Halbäquivalenzpunkte                          #
    # ------------------------------------------------------------------ #
    def find_half_equivalence_points(self, x, y, peak_indices):
        """
        Berechnet die Halbäquivalenzpunkte.

        Fuer jeden Aequivalenzpunkt wird das Volumen bestimmt, das genau in
        der Mitte zwischen dem vorherigen Startpunkt (Kurvenanfang bzw. letztem
        Aequivalenzpunkt) und dem aktuellen Aequivalenzpunkt liegt.
        Der zugehoerige pH-Wert wird direkt aus der interpolierten Kurve y gelesen.

        Rueckgabe: Liste von (v_half, ph_half, idx_half) Tupeln.
        """
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

    def analyze_current(self):
        tab_id = self.notebook.select()
        tab_name = self.notebook.tab(tab_id, "text")
        tab = self.notebook.nametowidget(tab_id)

        v_raw, p_raw = [], []
        for v_e, p_e in tab.table.cells:
            vs, ps = v_e.get().replace(',', '.'), p_e.get().replace(',', '.')
            if vs and ps:
                try: v_raw.append(float(vs)); p_raw.append(float(ps))
                except: continue

        if len(v_raw) < 5: return
        v, ph = np.array(v_raw), np.array(p_raw)
        idx = np.argsort(v); v, ph = v[idx], ph[idx]
        xf = np.linspace(v.min(), v.max(), 5000)

        p_i = PchipInterpolator(v, ph); yp = p_i(xf); dp = savgol_filter(np.gradient(yp, xf), 101, 3)
        s_i = CubicSpline(v, ph);       ys = s_i(xf); ds = savgol_filter(s_i.derivative()(xf), 101, 3)

        for ax in self.axs.flat:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.6)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"\n{'='*25}\n")
        self.log_text.insert(tk.END, f"ANALYSE {tab_name} ({timestamp})\n", "header")
        self.log_text.tag_config("header", font=("Consolas", 8, "bold"), foreground="blue")

        c_titr, c_deriv = 'royalblue', 'crimson'

        self.axs[0,0].scatter(v, ph, color='gray', alpha=0.3, s=10); self.axs[0,0].plot(xf, yp, color=c_titr); self.axs[0,0].set_title("PCHIP Kurve")
        self.axs[1,0].plot(xf, dp, color=c_deriv); self.axs[1,0].set_title("PCHIP 1. Ableitung")
        self.axs[0,1].scatter(v, ph, color='gray', alpha=0.3, s=10); self.axs[0,1].plot(xf, ys, color=c_titr); self.axs[0,1].set_title("Spline Kurve")
        self.axs[1,1].plot(xf, ds, color=c_deriv); self.axs[1,1].set_title("Spline 1. Ableitung")

        for name, x, y, d, axk, axa in [("PCHIP",  xf, yp, dp, self.axs[0,0], self.axs[1,0]),
                                         ("SPLINE", xf, ys, ds, self.axs[0,1], self.axs[1,1])]:
            pks, _ = find_peaks(d, prominence=0.3, distance=200)
            self.log_text.insert(tk.END, f"[{name}]\n")

            # --- Aequivalenzpunkte (unveraendert) ---
            self.log_text.insert(tk.END, "  -- Aequivalenzpunkte --\n")
            for p in pks:
                axk.plot(x[p], y[p], marker='+', color='black', mew=1.5, ms=10)
                axa.plot(x[p], d[p], marker='+', color='black', mew=1.5, ms=10)
                axk.text(x[p]+0.2, y[p], f"{x[p]:.2f}", fontsize=8)
                axa.text(x[p]+0.2, d[p], f"{x[p]:.2f}", fontsize=8)
                self.log_text.insert(tk.END, f"  V: {x[p]:.2f} ml | pH: {y[p]:.2f}\n")

            # --- Halbäquivalenzpunkte (neu) ---
            half_points = self.find_half_equivalence_points(x, y, pks)
            self.log_text.insert(tk.END, "  -- Halbaquivalenzpunkte --\n")
            for (v_half, ph_half, idx_half) in half_points:
                # Titrationskurve: orangefarbenes Dreieck
                axk.plot(v_half, ph_half, marker='x', color='black',
                         mew=1.5, ms=8, linestyle='None')
                axk.text(v_half + 0.2, ph_half, f"{v_half:.2f}", fontsize=8,
                         color='black')
                # Ableitungskurve: orangefarbenes Dreieck
                axa.plot(v_half, d[idx_half], marker='x', color='black',
                         mew=1.5, ms=8, linestyle='None')
                axa.text(v_half + 0.2, d[idx_half], f"{v_half:.2f}", fontsize=8,
                         color='black')
                self.log_text.insert(tk.END,
                    f"  V: {v_half:.2f} ml | pH: {ph_half:.2f}\n")

        self.log_text.insert(tk.END, f"{'='*25}\n")
        self.log_text.see(tk.END)
        self.fig.tight_layout(); self.canvas_agg.draw()

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
                ax = self.axs[0, mode]
                
                # --- KORREKTUR HIER ---
                # Bounding-Box abrufen und von Pixeln in Zoll (Inches) umwandeln
                bbox = ax.get_tightbbox(self.fig.canvas.get_renderer())
                extent = bbox.transformed(self.fig.dpi_scale_trans.inverted()).expanded(1.2, 1.2)
                
                self.fig.savefig(path, bbox_inches=extent, **save_kwargs)
            messagebox.showinfo("Erfolg", f"Datei gespeichert als {fmt.upper()}!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Export fehlgeschlagen:\n{e}")


if __name__ == "__main__":
    root = tk.Tk(); app = TitrationAnalyzerApp(root); root.mainloop()
