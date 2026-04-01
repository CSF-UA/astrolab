from pathlib import Path
import re
import math
import numpy as np
import matplotlib.pyplot as plt

# ── USER SETTINGS ─────────────────────────────────────────────
data_root   = Path("93-sector-selected")   # folder that holds TIC_*__mag.tess
output_dir  = Path("lc_grids_out")                    # where to save the grid images
tic_list_txt = None  # e.g., Path("/path/to/tic_list.txt") or None to include all files

# grid choice: e.g., 5x3 or 4x4
n_rows = 4
n_cols = 2

# each panel keeps this size; figure scales with grid
panel_width  = 6   # inches per subplot (x)
panel_height = 4   # inches per subplot (y)

dpi = 150
marker_size = 2

# ── HELPERS ──────────────────────────────────────────────────
tic_filename_re = re.compile(r"^TIC_(\d+)__mag\.tess$", re.IGNORECASE)

def extract_tic_id(path: Path) -> str | None:
    m = tic_filename_re.match(path.name)
    return m.group(1) if m else None

def load_tic_filter(txt_path: Path) -> set[str]:
    ids = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("TIC"):
                s = s.split()[-1]
            if s.isdigit():
                ids.add(s)
    return ids

# ── COLLECT FILES ────────────────────────────────────────────
output_dir.mkdir(parents=True, exist_ok=True)

# find all TIC_*__mag.tess files under data_root (recursively)
all_files = sorted(data_root.rglob("TIC_*__mag.tess"))

if tic_list_txt is not None:
    keep = load_tic_filter(Path(tic_list_txt))
    files = [p for p in all_files if (extract_tic_id(p) in keep)]
else:
    files = all_files

# sort by numeric TIC id for a clean order
files = sorted(
    files,
    key=lambda p: int(extract_tic_id(p)) if extract_tic_id(p) else 10**18
)

if not files:
    print("No matching .tess files found.")
else:
    print(f"Found {len(files)} .tess files to plot.")

# ── GLOBAL STYLE ─────────────────────────────────────────────
plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

stars_per_fig = n_rows * n_cols
n_pages = math.ceil(len(files) / stars_per_fig)

# ── PLOT PAGES ───────────────────────────────────────────────
for page in range(n_pages):
    start = page * stars_per_fig
    end   = min(start + stars_per_fig, len(files))
    batch = files[start:end]

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        squeeze=False
    )
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, path in enumerate(batch):
        ax = axes[i]

        # read data
        try:
            data = np.loadtxt(path)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center", color="red")
            ax.axis("off")
            continue

        if data.ndim != 2 or data.shape[1] < 2:
            ax.text(0.5, 0.5, f"Bad shape {data.shape}", ha="center", va="center", color="red")
            ax.axis("off")
            continue

        t, m = data[:, 0], data[:, 1]
        mask = np.isfinite(t) & np.isfinite(m)
        t, m = t[mask], m[mask]
        if t.size < 5:
            ax.text(0.5, 0.5, "Too few points", ha="center", va="center", color="red")
            ax.axis("off")
            continue

        ax.plot(t, m, ".k", markersize=marker_size)
        ax.invert_yaxis()

        tic_id = extract_tic_id(path) or "UNKNOWN"
        ax.set_title(f"TIC {tic_id}")
        ax.set_xlabel("JD - 2 457 000")
        ax.set_ylabel("Magnitude, mmag")
        ax.ticklabel_format(style="plain", axis="y")

    # turn off any unused panels on the last page
    for j in range(len(batch), len(axes)):
        axes[j].axis("off")

    outname = output_dir / f"LC_grid_{n_rows}x{n_cols}_page_{page+1:02d}.png"
    fig.savefig(outname, dpi=dpi)
    plt.close(fig)
    print(f"Saved {outname}")
