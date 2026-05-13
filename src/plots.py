"""Generate the figures used in README.md.

The figures are intentionally compact: the README carries the argument,
while each image shows one piece of evidence.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from zipfile import BadZipFile


BLUE = "#2A6F97"
GREEN = "#5B8E7D"
ORANGE = "#D97757"
DARK = "#2F3437"
MUTED = "#7B8085"
GRID = "#E6E8EA"
BG = "#FBFBFA"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "figure.facecolor": "white",
        "axes.facecolor": BG,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_npz(path: Path):
    try:
        return np.load(path)
    except (BadZipFile, OSError, ValueError):
        return None


def load_singular_values(path: Path) -> np.ndarray | None:
    blob = load_npz(path)
    if blob is None or "S" not in blob:
        return None
    return blob["S"]


def load_extra_dim(path: Path) -> int | None:
    blob = load_npz(path)
    if blob is None:
        return None
    if "extra_basis" in blob:
        return int(blob["extra_basis"].shape[0])
    return None


def annotate(ax, text: str, xy, xytext, color=DARK, ha="center") -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        ha=ha,
        va="center",
        color=color,
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9),
    )


def plot_concept_cartoon(out_path: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = (3.1, 2.45)
    ax.add_patch(Ellipse(theta, 5.2, 2.6, angle=8, facecolor=ORANGE,
                         edgecolor=ORANGE, alpha=0.10, lw=2, linestyle="--"))
    ax.add_patch(Ellipse(theta, 4.2, 2.1, angle=8, facecolor="none",
                         edgecolor=ORANGE, alpha=0.85, lw=1.8,
                         linestyle=(0, (5, 4))))
    t = np.linspace(-2.2, 2.2, 200)
    x = theta[0] + 1.35 * t
    y = theta[1] + 0.38 * np.sin(1.5 * t)
    ax.plot(x, y, color=GREEN, lw=7, alpha=0.16, solid_capstyle="round")
    ax.plot(x, y, color=GREEN, lw=3, solid_capstyle="round")
    ax.add_patch(Circle(theta, 0.23, facecolor=BLUE, edgecolor="white", lw=3, zorder=5))

    annotate(ax, "trained representative", theta, (1.25, 1.0), BLUE, ha="left")
    annotate(ax, "known architectural gauge orbit", (5.0, 2.1), (6.4, 0.85), GREEN)
    annotate(ax, "possible wider equivalence sheet", (2.2, 3.65), (2.6, 4.55), ORANGE)

    ax.plot([7.6, 7.6], [0.6, 4.4], color=GRID, lw=1.4)
    for y0 in (1.15, 2.45, 3.65):
        ax.add_patch(Circle((8.35, y0), 0.12, facecolor="#BFC4C7", edgecolor="none"))
        tt = np.linspace(-0.8, 0.8, 80)
        ax.plot(8.35 + tt, y0 + 0.18 * np.sin(2.5 * tt),
                color="#BFC4C7", lw=2, solid_capstyle="round")
    ax.text(8.35, 4.55, "other trained seeds", ha="center",
            color=MUTED, fontsize=10, fontweight="bold")

    ax.set_title("What EqNet measures: is [theta] wider than the known gauge?")
    fig.savefig(out_path)
    plt.close(fig)


def plot_jacobian_spectrum(results_dir: Path, out_path: Path) -> None:
    setup_style()
    mlp_files = sorted(results_dir.glob("tangent_A_*.npz"))
    mlp_files += sorted(results_dir.glob("tangent_mlp_MLP_*.npz"))
    cnn_files = sorted(results_dir.glob("tangent_cnn_CNN_*.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)

    for ax, files, gauge_dim, title, shade_color in [
        (axes[0], mlp_files, 64, "MLP baseline: nullity matches gauge", GREEN),
        (axes[1], cnn_files, 24, "CNN: nullity is larger than gauge", ORANGE),
    ]:
        for i, path in enumerate(files):
            s = load_singular_values(path)
            if s is None:
                continue
            s = np.clip(s, 1e-17, None)
            label = (path.stem
                     .replace("tangent_cnn_", "")
                     .replace("tangent_mlp_", "")
                     .replace("tangent_", ""))
            ax.semilogy(np.arange(1, len(s) + 1), s, lw=1.2, alpha=0.75, label=label)
        if files:
            first_s = None
            for path in files:
                first_s = load_singular_values(path)
                if first_s is not None:
                    break
            if first_s is None:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no readable spectrum files",
                        transform=ax.transAxes, ha="center", color=MUTED)
                continue
            p = len(first_s)
            ax.axvline(p - gauge_dim, color=DARK, ls="--", lw=1.1,
                       label=f"gauge cutoff P-{gauge_dim}")
            ax.axvspan(p - gauge_dim, p, color=GREEN, alpha=0.09)
            if shade_color == ORANGE:
                extra_dim = None
                for path in files:
                    extra_dim = load_extra_dim(path)
                    if extra_dim is not None:
                        break
                if extra_dim:
                    ax.axvspan(p - gauge_dim - extra_dim, p - gauge_dim,
                               color=ORANGE, alpha=0.08,
                               label="measured extra directions")
        ax.set_title(title)
        ax.set_xlabel("singular-value index")
        ax.set_ylabel("singular value of J")
        ax.legend(loc="lower left", ncol=2)

    fig.suptitle("Local tangent space from the Jacobian spectrum", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cnn_dim_breakdown(walk_cnn_json: Path, out_path: Path) -> None:
    setup_style()
    data = load_json(walk_cnn_json)
    names = [Path(r["ckpt"]).stem for r in data]
    extras = np.array([r["extra_dim"] for r in data])
    totals = np.array([r["tangent_dim"] for r in data])
    gauge = int(data[0]["gauge_dim"]) if data else 24
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.barh(y, np.full_like(extras, gauge), color=GREEN, alpha=0.85,
            label=f"known gauge = {gauge}")
    ax.barh(y, extras, left=gauge, color=ORANGE, alpha=0.9,
            label="measured extras")
    for yi, total in zip(y, totals):
        ax.text(total + 10, yi, f"{int(total)} total", va="center", ha="left",
                color=ORANGE, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("dimension")
    ax.set_title("CNN anchors have many more flat directions than the known gauge")
    ax.set_xlim(0, max(totals) * 1.18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_walk_test(results_dir: Path, out_path: Path) -> None:
    setup_style()
    panels = [
        (results_dir / "walk_test_mlp.json", "MLP: known gauge directions"),
        (results_dir / "walk_test_cnn.json", "CNN: extra directions"),
    ]
    colors = {"gauge": GREEN, "extra": BLUE, "curved": ORANGE, "random": MUTED}
    labels = {
        "gauge": "architectural gauge",
        "extra": "extra subspace",
        "curved": "high-curvature directions",
        "random": "random directions",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)

    for ax, (path, title) in zip(axes, panels):
        if not path.exists():
            ax.set_axis_off()
            continue
        data = load_json(path)
        if not data:
            ax.set_axis_off()
            continue
        eps = np.array(data[0]["walk_eps_grid"])
        for key in ("gauge", "extra", "curved", "random"):
            series = []
            for record in data:
                result = record["walk_results"].get(key, {"available": False})
                if not result.get("available"):
                    continue
                losses = np.array(result["losses"])
                series.append(losses.mean(axis=1) - record["L0"])
            if not series:
                continue
            arr = np.stack(series)
            mean = arr.mean(axis=0)
            ax.plot(eps, mean, marker="o", ms=4, lw=1.8,
                    color=colors[key], label=labels[key])
        ax.axhline(0, color=DARK, lw=0.8)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("step size epsilon")
        ax.set_title(title)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("change in training loss")
    fig.suptitle("Walking the candidate subspace stays flat", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tangent_dim_scaling(json_path: Path, out_path: Path, gauge_dim: int = 24) -> None:
    setup_style()
    data = load_json(json_path)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    final_values = []
    for i, anchor in enumerate(data):
        rows = [r for r in anchor["records"] if "tangent_dim" in r]
        ns = np.array([r["n_samples"] for r in rows])
        tangent_dims = np.array([r["tangent_dim"] for r in rows])
        final_values.append(tangent_dims[-1])
        ax.plot(ns, tangent_dims, marker=markers[i % len(markers)], ms=5,
                lw=1.8, label=anchor["anchor"])
    ax.axhline(gauge_dim, color=GREEN, lw=2, ls="--",
               label=f"architectural gauge = {gauge_dim}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Jacobian sample count")
    ax.set_ylabel("measured output-preserving dimension")
    ax.set_title("Measured flat dimension shrinks, then stays above the gauge")
    ax.legend(loc="upper right", ncol=2)
    if final_values:
        lo, hi = int(min(final_values)), int(max(final_values))
        ax.text(0.97, 0.12, f"final plateau: {lo}-{hi} directions",
                transform=ax.transAxes, ha="right", color=ORANGE,
                fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_orbit_vs_projector(out_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    titles = [
        "What we want: fields tangent to the orbit",
        "What we can build: projected fields",
    ]
    subtitles = [
        "commutators would test group structure",
        "the fixed direction misses the turning sheet",
    ]

    for ax, title, subtitle in zip(axes, titles, subtitles):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        t = np.linspace(-2, 2, 160)
        x = 3 + 2.2 * np.sin(0.8 * t)
        y = 2 + 0.55 * t
        ax.plot(x, y, color=GREEN, lw=8, alpha=0.15, solid_capstyle="round")
        ax.plot(x, y, color=GREEN, lw=3, solid_capstyle="round")
        ax.set_title(title, fontsize=11)
        ax.text(3, 0.25, subtitle, ha="center", color=MUTED, fontsize=9)

    for t0 in [-1.3, -0.3, 0.8]:
        x0 = 3 + 2.2 * np.sin(0.8 * t0)
        y0 = 2 + 0.55 * t0
        dx = 2.2 * 0.8 * np.cos(0.8 * t0)
        dy = 0.55
        norm = np.hypot(dx, dy)
        axes[0].annotate("", xy=(x0 + dx / norm * 0.65, y0 + dy / norm * 0.65),
                         xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
        axes[1].annotate("", xy=(x0 + 0.65, y0), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))

    axes[1].annotate("not tangent\nafter the sheet turns",
                     xy=(4.55, 3.05), xytext=(4.9, 3.65),
                     ha="center", color=DARK, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

    fig.suptitle("Why the Lie-bracket attempt is inconclusive", y=1.02,
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    results = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    figs = Path("figs")
    figs.mkdir(parents=True, exist_ok=True)
    plot_concept_cartoon(figs / "00_concept.png")
    plot_jacobian_spectrum(results, figs / "01_jacobian_spectrum.png")
    plot_cnn_dim_breakdown(results / "walk_test_cnn.json", figs / "02_cnn_dim_breakdown.png")
    plot_walk_test(results, figs / "03_walk_test.png")
    plot_tangent_dim_scaling(results / "tangent_dim_scaling.json",
                             figs / "04_tangent_dim_scaling.png")
    plot_orbit_vs_projector(figs / "05_orbit_vs_projector.png")
    print(f"wrote figures to {figs}")


if __name__ == "__main__":
    main()
