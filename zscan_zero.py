#!/usr/bin/env python3
"""
zscan_zero.py -- find the coverglass surface in an axial (Z) intensity scan and
re-reference the Z axis so that z = 0 sits at the surface.

Two cases are handled automatically:

  adsorption present   : fluorophore accumulated at the glass/water interface
                         gives a sharp spike on top of the bulk plateau.
                         z = 0 is taken as the spike position (parabolic fit
                         through the three points around the apex).

  no adsorption        : the profile is a simple sigmoid from dark (inside the
                         glass) to the bulk solution plateau.  z = 0 is taken
                         as the half-rise point, interpolated between samples.

Which case applies is decided from the peak-to-bulk ratio (PBR): the lightly
smoothed maximum divided by the solution plateau.  Above --ratio (default 1.5)
adsorption is assumed.

Input  : two-column text, whitespace- and/or comma-delimited, one header line.
Output : tab-delimited, same intensities, Z shifted so the surface is at zero,
         with the fit parameters recorded in '#' comment lines at the top.
         A batch of two or more files also produces a tab-delimited summary
         table, one row per file.

Usage
  python zscan_zero.py                            # graphical file picker
  python zscan_zero.py scan1.txt scan2.txt ...    # command line
  python zscan_zero.py *.txt -o corrected/ --plot
  python zscan_zero.py scan.txt --force half      # override the decision
  python zscan_zero.py scan.txt --z0 -27.1        # set the offset by hand
"""

import argparse
import os
import re
import sys
from datetime import datetime
from statistics import median

VERSION = "2.0"
SUMMARY_PREFIX = "zscan_summary"

# Column order of the batch summary table, and the matching result keys.
SUMMARY_COLUMNS = [
    ("file",             "file"),
    ("mode",             "mode"),
    ("z0_orig",          "z0"),
    ("bulk_glass",       "bulk_glass"),
    ("bulk_solution",    "bulk_solution"),
    ("peak",             "peak"),
    ("peak_raw",         "peak_raw"),
    ("z_peak_corr",      "z_peak_corr"),
    ("PBR",              "pbr"),
    ("PBR_bkgcorr",      "pbr_bkgcorr"),
    ("half_level",       "half_level"),
    ("solution_side",    "solution_side"),
    ("n_points",         "n_points"),
    ("z_min_corr",       "z_min_corr"),
    ("z_max_corr",       "z_max_corr"),
]

# ----------------------------------------------------------------- I/O -----


def load(path):
    """Return (z, intensity) as lists of float, sorted by ascending z."""
    z, y = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue                       # header or junk line
            z.append(a)
            y.append(b)
    if len(z) < 10:
        raise ValueError(f"{os.path.basename(path)}: fewer than 10 usable "
                         f"data points")
    order = sorted(range(len(z)), key=lambda i: z[i])
    return [z[i] for i in order], [y[i] for i in order]


def _fmt(v, nd=4):
    """Number for a header/table cell; blank for None, %g-ish for floats."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def header_lines(r):
    """The '#' provenance block that goes at the top of a corrected file."""
    return [
        f"# zscan_zero.py v{VERSION}   {r['timestamp']}",
        f"# source file          : {r['file']}",
        f"# method               : {r['mode_long']}",
        f"# z0 (orig. coords)    : {_fmt(r['z0'])}",
        f"# bulk, glass side     : {_fmt(r['bulk_glass'], 1)}",
        f"# bulk, solution side  : {_fmt(r['bulk_solution'], 1)}",
        f"# peak (smoothed)      : {_fmt(r['peak'], 1)}",
        f"# peak (raw max)       : {_fmt(r['peak_raw'], 1)}",
        f"# peak position (corr) : {_fmt(r['z_peak_corr'])}",
        f"# PBR  peak/bulk       : {_fmt(r['pbr'], 3)}",
        f"# PBR  background-corr : {_fmt(r['pbr_bkgcorr'], 3)}   "
        f"[(peak-glass)/(bulk-glass)]",
        f"# half-rise level      : {_fmt(r['half_level'], 1)}",
        f"# solution side        : {r['solution_side']}",
        f"# settings             : ratio={r['ratio']}, window={r['window']}"
        + (f", force={r['force']}" if r["force"] else "")
        + (", z0 set by hand" if r["mode"] == "manual" else ""),
    ]


def write_corrected(path, r, z, y, columns=("Z (um)", "Ch-1 Intensity")):
    z0 = r["z0"]
    with open(path, "w", newline="\n") as fh:
        for line in header_lines(r):
            fh.write(line + "\n")
        fh.write("\t".join(columns) + "\n")
        for zi, yi in zip(z, y):
            fh.write(f"{zi - z0:.4f}\t{yi:g}\n")


def write_summary(path, results, settings_note=""):
    """Tab-delimited one-row-per-file table for a batch."""
    with open(path, "w", newline="\n") as fh:
        fh.write(f"# zscan_zero.py v{VERSION} batch summary   "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if settings_note:
            fh.write(f"# {settings_note}\n")
        fh.write("# PBR = peak / bulk(solution); PBR_bkgcorr = "
                 "(peak - bulk_glass) / (bulk_solution - bulk_glass)\n")
        fh.write("# z0_orig is the surface position in the ORIGINAL z axis; "
                 "corrected files have it subtracted\n")
        fh.write("\t".join(c[0] for c in SUMMARY_COLUMNS) + "\n")
        for r in results:
            row = []
            for _, key in SUMMARY_COLUMNS:
                v = r.get(key)
                row.append(_fmt(v, 3) if isinstance(v, float) else _fmt(v))
            fh.write("\t".join(row) + "\n")

# ------------------------------------------------------------ analysis -----


def smooth(y, w=5):
    """Centred moving average, window w (odd). Used for detection only."""
    h, n, out = w // 2, len(y), []
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def find_surface(z, y, ratio=1.5, smooth_window=5, force=None):
    """Locate the glass surface.  Returns a dict of results and diagnostics.

    The returned z0 is in the ORIGINAL z coordinates of the input file.
    """
    n = len(z)
    ys = smooth(y, smooth_window)               # levels and half-rise
    yp = smooth(y, min(3, smooth_window))       # apex and PBR: a surface
    k = max(5, n // 10)                         # spike is only ~3 points wide

    # Which end of the scan is the bulk solution?  (the bright end)
    solution_right = median(ys[-k:]) >= median(ys[:k])
    if not solution_right:              # work dark-end-first in both cases
        z, y, ys, yp = z[::-1], y[::-1], ys[::-1], yp[::-1]

    glass = median(ys[:k])                      # inside the glass
    bulk = median(ys[(2 * n) // 3:])            # far solution plateau

    japex = max(range(n), key=lambda i: yp[i])
    peak = yp[japex]
    peak_raw = max(y)
    pbr = peak / bulk if bulk > 0 else float("inf")
    span = bulk - glass
    pbr_bkg = (peak - glass) / span if span > 0 else float("inf")

    mode = force or ("peak" if pbr > ratio else "half")
    half_level = glass + 0.5 * span

    if mode == "peak":
        # Parabolic interpolation through the apex and its two neighbours.
        i = min(max(japex, 1), n - 2)
        y1, y2, y3 = yp[i - 1], yp[i], yp[i + 1]
        denom = y1 - 2 * y2 + y3
        frac = 0.5 * (y1 - y3) / denom if denom != 0 else 0.0
        frac = max(-1.0, min(1.0, frac))
        step = (z[i + 1] - z[i - 1]) / 2.0
        z0 = z[i] + frac * step
        mode_long = "peak (adsorption; parabolic apex)"
    else:
        # First crossing of the half-rise level, scanning from the dark end.
        z0 = None
        for i in range(1, n):
            if ys[i - 1] < half_level <= ys[i]:
                f = (half_level - ys[i - 1]) / (ys[i] - ys[i - 1])
                z0 = z[i - 1] + f * (z[i] - z[i - 1])
                break
        if z0 is None:                   # never crosses: fall back to the max
            z0 = z[japex]
            mode_long = "half (NO CROSSING FOUND -- fell back to maximum)"
        else:
            mode_long = "half (no adsorption; half-rise of the edge)"

    return dict(
        z0=z0, mode=mode, mode_long=mode_long,
        bulk_glass=glass, bulk_solution=bulk,
        peak=peak, peak_raw=float(peak_raw), z_peak_corr=z[japex] - z0,
        pbr=pbr, pbr_bkgcorr=pbr_bkg, half_level=half_level,
        solution_side="+z" if solution_right else "-z",
        n_points=n, z_min_corr=min(z) - z0, z_max_corr=max(z) - z0,
    )

# ------------------------------------------------------------- driver -----


def make_plot(png_path, z, y, r):
    """Write a PNG of the corrected scan.  Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    z0 = r["z0"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([zi - z0 for zi in z], y, lw=1, color="#1f77b4")
    ax.axvline(0, color="crimson", lw=1, ls="--")
    if r["mode"] == "half":
        ax.axhline(r["half_level"], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("z - z0 (um)")
    ax.set_ylabel("intensity (counts)")
    if min(y) > 0:
        ax.set_yscale("log")
    ax.set_title(f"{r['stem']}   ({r['mode']}, z0 = {z0:.2f}, "
                 f"PBR = {r['pbr']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def analyze_file(path, out_dir, ratio=1.5, window=5, force=None,
                 z0_override=None, suffix="_zcorr", plot=False):
    """Analyse one file and write its corrected copy.  Returns a result dict."""
    z, y = load(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    if z0_override is not None:
        r = find_surface(z, y, ratio, window, force)
        r["z_peak_corr"] += r["z0"] - z0_override      # re-reference diagnostics
        r["z_min_corr"] += r["z0"] - z0_override
        r["z_max_corr"] += r["z0"] - z0_override
        r["z0"] = z0_override
        r["mode"] = "manual"
        r["mode_long"] = "manual (z0 supplied by the user)"
    else:
        r = find_surface(z, y, ratio, window, force)

    r.update(file=os.path.basename(path), path=os.path.abspath(path),
             stem=stem, ratio=ratio, window=window, force=force,
             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, stem + suffix + ".txt")
    write_corrected(out, r, z, y)
    r["output"] = out

    r["plot"] = None
    if plot:
        png = os.path.join(out_dir, stem + suffix + ".png")
        if make_plot(png, z, y, r):
            r["plot"] = png

    return r


def run_batch(paths, out_dir, ratio=1.5, window=5, force=None,
              z0_override=None, suffix="_zcorr", plot=False, summary=True,
              on_result=None, on_error=None):
    """Analyse every path.  Returns (results, errors, summary_path_or_None).

    on_result(r) / on_error(path, exc) are optional callbacks so a GUI can
    report progress as it goes.
    """
    results, errors = [], []
    for p in paths:
        try:
            r = analyze_file(p, out_dir, ratio, window, force,
                             z0_override, suffix, plot)
        except Exception as exc:                       # keep the batch going
            errors.append((p, exc))
            if on_error:
                on_error(p, exc)
            continue
        results.append(r)
        if on_result:
            on_result(r)

    summary_path = None
    if summary and len(results) > 1:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_path = os.path.join(out_dir, f"{SUMMARY_PREFIX}_{stamp}.txt")
        note = f"settings: ratio={ratio}, window={window}"
        if force:
            note += f", force={force}"
        write_summary(summary_path, results, note)

    return results, errors, summary_path

# ---------------------------------------------------------------- text -----

TABLE_HEAD = (f"{'file':<32}{'mode':<8}{'z0':>9}{'PBR':>8}"
              f"{'glass':>9}{'solution':>10}{'peak':>10}")


def table_row(r):
    return (f"{r['file'][:31]:<32}{r['mode']:<8}{r['z0']:>9.3f}"
            f"{r['pbr']:>8.2f}{r['bulk_glass']:>9.0f}"
            f"{r['bulk_solution']:>10.0f}{r['peak']:>10.0f}")


def main_cli(args):
    if args.z0 is not None and len(args.files) > 1:
        sys.exit("--z0 only makes sense with a single file")

    print(TABLE_HEAD)
    results, errors, summary_path = run_batch(
        args.files, args.out_dir, args.ratio, args.window, args.force,
        args.z0, args.suffix, args.plot,
        on_result=lambda r: print(table_row(r)),
        on_error=lambda p, e: print(f"  !! {os.path.basename(p)}: {e}",
                                    file=sys.stderr),
    )
    if summary_path:
        print(f"\nsummary: {summary_path}")
    if errors and not results:
        sys.exit(1)

# ----------------------------------------------------------------- GUI -----


def main_gui(initial=None):
    """Launch the Tk interface.  Imported lazily so the CLI needs no Tk."""
    try:
        import tkinter as tk                                    # noqa: F401
        from tkinter import ttk                                 # noqa: F401
    except ImportError:
        sys.exit("Tkinter is not available in this Python installation.\n"
                 "Install it (e.g. 'sudo apt install python3-tk', or use the\n"
                 "python.org / Anaconda build on Windows and macOS), or run\n"
                 "this script with file names on the command line instead.")
    app = ZScanApp(initial or [])
    app.mainloop()


class ZScanApp:
    """Thin widget layer over run_batch()."""

    def __init__(self, initial=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk

        self.root = tk.Tk()
        self.root.title(f"Z-scan zero correction  v{VERSION}")
        self.root.minsize(860, 560)

        self.paths = list(initial or [])
        self.out_dir = tk.StringVar(value="")
        self.ratio = tk.StringVar(value="1.5")
        self.window = tk.StringVar(value="5")
        self.mode = tk.StringVar(value="auto")
        self.manual_z0 = tk.StringVar(value="")
        self.use_manual = tk.BooleanVar(value=False)
        self.suffix = tk.StringVar(value="_zcorr")
        self.want_plot = tk.BooleanVar(value=False)
        self.same_dir = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Add one or more Z-scan files.")

        self._build()
        self._refresh_files()

    # -- layout ------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(3, weight=1)

        # files -----------------------------------------------------------
        bar = ttk.Frame(root, padding=(8, 8, 8, 2))
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Button(bar, text="Add files…",
                   command=self.add_files).pack(side="left")
        ttk.Button(bar, text="Remove selected",
                   command=self.remove_selected).pack(side="left", padx=4)
        ttk.Button(bar, text="Clear",
                   command=self.clear_files).pack(side="left")
        self.count_lbl = ttk.Label(bar, text="")
        self.count_lbl.pack(side="right")

        fframe = ttk.Frame(root, padding=(8, 0))
        fframe.grid(row=1, column=0, sticky="nsew")
        fframe.columnconfigure(0, weight=1)
        fframe.rowconfigure(0, weight=1)
        self.filelist = tk.Listbox(fframe, selectmode="extended",
                                   activestyle="none", height=6,
                                   exportselection=False)
        self.filelist.grid(row=0, column=0, sticky="nsew")
        fsb = ttk.Scrollbar(fframe, orient="vertical",
                            command=self.filelist.yview)
        fsb.grid(row=0, column=1, sticky="ns")
        self.filelist.configure(yscrollcommand=fsb.set)

        # options ----------------------------------------------------------
        opt = ttk.LabelFrame(root, text="Options", padding=8)
        opt.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        for c in (1, 3, 5):
            opt.columnconfigure(c, weight=0)
        opt.columnconfigure(7, weight=1)

        ttk.Label(opt, text="Surface from:").grid(row=0, column=0, sticky="w")
        modes = ttk.Frame(opt)
        modes.grid(row=0, column=1, columnspan=3, sticky="w")
        for text, val in (("auto", "auto"),
                          ("peak (adsorption)", "peak"),
                          ("half-rise (no adsorption)", "half")):
            ttk.Radiobutton(modes, text=text, value=val,
                            variable=self.mode).pack(side="left", padx=(0, 10))

        ttk.Label(opt, text="PBR threshold:").grid(row=1, column=0, sticky="w",
                                                   pady=(6, 0))
        ttk.Entry(opt, textvariable=self.ratio, width=7).grid(
            row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Smoothing window (pts):").grid(
            row=1, column=2, sticky="e", padx=(12, 4), pady=(6, 0))
        ttk.Entry(opt, textvariable=self.window, width=7).grid(
            row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Output suffix:").grid(
            row=1, column=4, sticky="e", padx=(12, 4), pady=(6, 0))
        ttk.Entry(opt, textvariable=self.suffix, width=10).grid(
            row=1, column=5, sticky="w", pady=(6, 0))

        self.manual_chk = ttk.Checkbutton(
            opt, text="Set z0 by hand (single file):",
            variable=self.use_manual, command=self._sync_manual)
        self.manual_chk.grid(row=2, column=0, columnspan=2, sticky="w",
                             pady=(6, 0))
        self.manual_entry = ttk.Entry(opt, textvariable=self.manual_z0,
                                      width=10, state="disabled")
        self.manual_entry.grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opt, text="Also save a PNG of each scan",
                        variable=self.want_plot).grid(
            row=2, column=3, columnspan=3, sticky="w", pady=(6, 0))

        outrow = ttk.Frame(opt)
        outrow.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(8, 0))
        outrow.columnconfigure(2, weight=1)
        ttk.Checkbutton(outrow, text="Write results next to the input files",
                        variable=self.same_dir,
                        command=self._sync_outdir).grid(row=0, column=0,
                                                        sticky="w")
        self.out_btn = ttk.Button(outrow, text="Output folder…",
                                  command=self.pick_outdir, state="disabled")
        self.out_btn.grid(row=0, column=1, padx=8)
        self.out_lbl = ttk.Label(outrow, textvariable=self.out_dir,
                                 foreground="#555")
        self.out_lbl.grid(row=0, column=2, sticky="w")

        # results ------------------------------------------------------------
        res = ttk.Frame(root, padding=(8, 0))
        res.grid(row=3, column=0, sticky="nsew")
        res.columnconfigure(0, weight=1)
        res.rowconfigure(0, weight=1)
        cols = ("file", "mode", "z0", "PBR", "glass", "solution", "peak")
        self.tree = self.ttk.Treeview(res, columns=cols, show="headings",
                                      height=7)
        widths = (260, 70, 80, 70, 80, 90, 90)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w,
                             anchor="w" if c == "file" else "e")
        self.tree.grid(row=0, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(res, orient="vertical", command=self.tree.yview)
        tsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tsb.set)

        # actions ------------------------------------------------------------
        act = ttk.Frame(root, padding=8)
        act.grid(row=4, column=0, sticky="ew")
        act.columnconfigure(0, weight=1)
        ttk.Label(act, textvariable=self.status).grid(row=0, column=0,
                                                      sticky="w")
        self.open_btn = ttk.Button(act, text="Open output folder",
                                   command=self.open_outdir, state="disabled")
        self.open_btn.grid(row=0, column=1, padx=(8, 4))
        ttk.Button(act, text="Analyze", command=self.run).grid(row=0, column=2)

    # -- helpers -----------------------------------------------------------

    def _sync_manual(self):
        self.manual_entry.configure(
            state="normal" if self.use_manual.get() else "disabled")

    def _sync_outdir(self):
        self.out_btn.configure(
            state="disabled" if self.same_dir.get() else "normal")
        if self.same_dir.get():
            self.out_dir.set("")

    def _refresh_files(self):
        self.filelist.delete(0, "end")
        for p in self.paths:
            self.filelist.insert("end", os.path.basename(p))
        n = len(self.paths)
        self.count_lbl.configure(
            text=f"{n} file{'s' if n != 1 else ''} selected")
        self.manual_chk.configure(state="normal" if n == 1 else "disabled")
        if n != 1 and self.use_manual.get():
            self.use_manual.set(False)
            self._sync_manual()

    def _resolved_outdir(self):
        if not self.same_dir.get() and self.out_dir.get():
            return self.out_dir.get()
        if self.paths:
            return os.path.dirname(os.path.abspath(self.paths[0]))
        return os.getcwd()

    # -- commands ----------------------------------------------------------

    def add_files(self):
        from tkinter import filedialog
        picked = filedialog.askopenfilenames(
            title="Select Z-scan files",
            filetypes=[("Text / data files", "*.txt *.dat *.csv *.tsv"),
                       ("All files", "*.*")])
        added = 0
        for p in picked:
            if p not in self.paths:
                self.paths.append(p)
                added += 1
        if added:
            self._refresh_files()
            self.status.set(f"Added {added} file{'s' if added != 1 else ''}.")

    def remove_selected(self):
        for i in sorted(self.filelist.curselection(), reverse=True):
            del self.paths[i]
        self._refresh_files()

    def clear_files(self):
        self.paths = []
        self._refresh_files()
        self.status.set("Add one or more Z-scan files.")

    def pick_outdir(self):
        from tkinter import filedialog
        d = filedialog.askdirectory(title="Output folder")
        if d:
            self.out_dir.set(d)

    def open_outdir(self):
        d = getattr(self, "last_outdir", None)
        if not d or not os.path.isdir(d):
            return
        import subprocess
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                               # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception:
            pass

    def run(self):
        from tkinter import messagebox
        if not self.paths:
            messagebox.showinfo("Nothing to do", "Add at least one file.")
            return
        try:
            ratio = float(self.ratio.get())
            window = int(float(self.window.get()))
            if window < 1:
                raise ValueError("window must be >= 1")
        except ValueError as exc:
            messagebox.showerror("Bad setting", f"Check the options: {exc}")
            return

        z0 = None
        if self.use_manual.get():
            if len(self.paths) != 1:
                messagebox.showerror(
                    "Bad setting",
                    "A manual z0 can only be applied to a single file.")
                return
            try:
                z0 = float(self.manual_z0.get())
            except ValueError:
                messagebox.showerror("Bad setting",
                                     "Enter a number for the manual z0.")
                return

        force = None if self.mode.get() == "auto" else self.mode.get()
        out_dir = self._resolved_outdir()
        suffix = self.suffix.get() or "_zcorr"

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.status.set("Working…")
        self.root.update_idletasks()

        def show(r):
            self.tree.insert("", "end", values=(
                r["file"], r["mode"], f"{r['z0']:.3f}", f"{r['pbr']:.2f}",
                f"{r['bulk_glass']:.0f}", f"{r['bulk_solution']:.0f}",
                f"{r['peak']:.0f}"))
            self.root.update_idletasks()

        try:
            results, errors, summary = run_batch(
                self.paths, out_dir, ratio, window, force, z0, suffix,
                self.want_plot.get(), on_result=show)
        except Exception as exc:
            messagebox.showerror("Failed", str(exc))
            self.status.set("Failed.")
            return

        self.last_outdir = out_dir
        self.open_btn.configure(state="normal")

        msg = (f"{len(results)} file{'s' if len(results) != 1 else ''} "
               f"written to {out_dir}")
        if summary:
            msg += f"  |  summary: {os.path.basename(summary)}"
        if errors:
            msg += f"  |  {len(errors)} failed"
        self.status.set(msg)

        if errors:
            messagebox.showwarning(
                "Some files failed",
                "\n".join(f"{os.path.basename(p)}: {e}" for p, e in errors))

    def mainloop(self):
        self.root.mainloop()

# ---------------------------------------------------------------- main -----


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help="scan files; omit them to open the graphical picker")
    ap.add_argument("-o", "--out-dir", default=".", help="output directory")
    ap.add_argument("--suffix", default="_zcorr", help="appended to the stem")
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="PBR above which adsorption is assumed (default 1.5)")
    ap.add_argument("--window", type=int, default=5,
                    help="smoothing window in points, odd (default 5)")
    ap.add_argument("--force", choices=["peak", "half"],
                    help="override the automatic choice for all files")
    ap.add_argument("--z0", type=float,
                    help="use this offset verbatim (single file only)")
    ap.add_argument("--plot", action="store_true",
                    help="also write a PNG of each corrected scan")
    ap.add_argument("--gui", action="store_true",
                    help="open the graphical picker even with files listed")
    args = ap.parse_args()

    if args.gui or not args.files:
        main_gui(args.files)
    else:
        main_cli(args)


if __name__ == "__main__":
    main()
