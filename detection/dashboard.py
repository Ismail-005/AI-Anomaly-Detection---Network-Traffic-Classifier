"""
dashboard.py
────────────────────────────────────────────────────────────────
Reads the alert_log.csv produced by realtime_detector.py and
renders a 6-panel dashboard customised to the deez.pkt topology.

Panels:
  1 — Traffic class distribution (bar chart)
  2 — Confidence score distribution per class (violin)
  3 — Anomaly rate over time (rolling window)
  4 — Top targeted destination ports
  5 — Attack flow heatmap (src_vlan → dst_vlan)
  6 — Most targeted devices (horizontal bar)

Usage:
  python detection/dashboard.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
LOG_PATH = BASE / "reports" / "alert_log.csv"
OUT_PNG  = BASE / "reports" / "dashboard.png"

# ── colour palette ────────────────────────────────────────────────────
BG      = "#0d1117"
CARD_BG = "#161b22"
ACCENT  = "#00d4ff"
PALETTE = {
    "Normal":      "#6bcb77",
    "DDoS":        "#ff6b6b",
    "Port Scan":   "#ffd93d",
    "Brute Force": "#c77dff",
    "Data Exfil":  "#ff9f43",
}
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "DejaVu Sans",
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.6,
})

# ── load data ─────────────────────────────────────────────────────────
def load_log():
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Alert log not found: {LOG_PATH}\n"
            "Run realtime_detector.py first."
        )
    df = pd.read_csv(LOG_PATH)
    return df

# ══════════════════════════════════════════════════════════════════════
#  DASHBOARD PANELS
# ══════════════════════════════════════════════════════════════════════

def panel_class_distribution(ax, df):
    """Panel 1: Traffic class distribution bar chart."""
    counts  = df["rf_label"].value_counts()
    colours = [PALETTE.get(l, "#888") for l in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colours,
                  edgecolor=BG, linewidth=0.8, zorder=3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    ax.set_title("Traffic Class Distribution", fontsize=11, color=ACCENT, pad=10)
    ax.set_xlabel("Class")
    ax.set_ylabel("Flow Count")
    ax.grid(axis="y", zorder=0)
    ax.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=8)


def panel_confidence_violin(ax, df):
    """Panel 2: Confidence score distribution per class."""
    labels  = sorted(df["rf_label"].unique())
    data    = [df.loc[df["rf_label"] == l, "confidence"].values for l in labels]
    colours = [PALETTE.get(l, "#888") for l in labels]

    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colours)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color(ACCENT)
    parts["cbars"].set_color("#555")
    parts["cmaxes"].set_color("#555")
    parts["cmins"].set_color("#555")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Confidence Score Distribution", fontsize=11, color=ACCENT, pad=10)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")


def panel_anomaly_rate(ax, df):
    """Panel 3: Anomaly rate over time (rolling window)."""
    window = min(50, len(df) // 3) if len(df) > 10 else 5
    df_copy = df.copy()
    df_copy["is_anomaly"] = (df_copy["rf_label"] != "Normal").astype(int)
    rolling = df_copy["is_anomaly"].rolling(window, min_periods=1).mean() * 100

    ax.fill_between(range(len(rolling)), rolling, alpha=0.25, color="#ff6b6b")
    ax.plot(rolling, color="#ff6b6b", linewidth=1.5, label=f"Rolling {window}-flow avg")
    ax.axhline(rolling.mean(), linestyle="--", color=ACCENT, linewidth=1,
               label=f"Overall avg: {rolling.mean():.1f}%")
    ax.set_title("Anomaly Rate Over Time", fontsize=11, color=ACCENT, pad=10)
    ax.set_xlabel("Flow Index")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid()


def panel_top_ports(ax, df):
    """Panel 4: Top targeted destination ports."""
    top = df["dst_port"].value_counts().head(10)
    suspicious_ports = {22, 3389, 4444, 1337, 9001, 6667, 8080}
    colours = [
        "#ff6b6b" if p in suspicious_ports else "#6bcb77"
        for p in top.index
    ]
    ax.barh([str(p) for p in top.index[::-1]],
            top.values[::-1],
            color=colours[::-1],
            edgecolor=BG, linewidth=0.6)
    ax.set_title("Top 10 Targeted Dst Ports", fontsize=11, color=ACCENT, pad=10)
    ax.set_xlabel("Flow Count")
    ax.grid(axis="x")
    legend = [Patch(color="#ff6b6b", label="Suspicious"),
              Patch(color="#6bcb77", label="Normal")]
    ax.legend(handles=legend, fontsize=7, loc="lower right")


def panel_vlan_heatmap(ax, df):
    """Panel 5: Attack flow heatmap (src_vlan → dst_vlan)."""
    if "src_vlan" not in df.columns or "dst_vlan" not in df.columns:
        ax.text(0.5, 0.5, "No VLAN data\navailable",
                ha="center", va="center", fontsize=12, color="#666")
        ax.set_title("Attack Flow by Zone", fontsize=11, color=ACCENT, pad=10)
        return

    attacks = df[df["rf_label"] != "Normal"]
    if attacks.empty:
        ax.text(0.5, 0.5, "No attacks detected",
                ha="center", va="center", fontsize=12, color="#6bcb77")
        ax.set_title("Attack Flow by Zone", fontsize=11, color=ACCENT, pad=10)
        return

    cross = pd.crosstab(attacks["src_vlan"], attacks["dst_vlan"])

    im = ax.imshow(cross.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(cross.columns)))
    ax.set_xticklabels(cross.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(cross.index)))
    ax.set_yticklabels(cross.index, fontsize=7)

    # Annotate cells
    for i in range(len(cross.index)):
        for j in range(len(cross.columns)):
            val = cross.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color="white" if val > cross.values.max()*0.5 else "#ddd")

    ax.set_title("Attack Flow Heatmap (Src → Dst Zone)", fontsize=11, color=ACCENT, pad=10)
    ax.set_xlabel("Destination Zone")
    ax.set_ylabel("Source Zone")


def panel_top_targets(ax, df):
    """Panel 6: Most targeted devices by attack traffic."""
    if "dst_device" not in df.columns:
        ax.text(0.5, 0.5, "No device data\navailable",
                ha="center", va="center", fontsize=12, color="#666")
        ax.set_title("Most Targeted Devices", fontsize=11, color=ACCENT, pad=10)
        return

    attacks = df[df["rf_label"] != "Normal"]
    if attacks.empty:
        ax.text(0.5, 0.5, "No attacks detected",
                ha="center", va="center", fontsize=12, color="#6bcb77")
        ax.set_title("Most Targeted Devices", fontsize=11, color=ACCENT, pad=10)
        return

    top_dev = attacks["dst_device"].value_counts().head(8)

    # Color by VLAN
    vlan_colors = {
        "HR": "#ff6b6b",
        "IT": "#ffd93d",
        "Finance": "#c77dff",
        "ASA": "#ff9f43",
        "NMS": "#00d4ff",
        "Internet": "#6bcb77",
    }
    colors = []
    for dev in top_dev.index:
        c = "#888"
        for key, col in vlan_colors.items():
            if key in str(dev):
                c = col
                break
        colors.append(c)

    ax.barh([str(d) for d in top_dev.index[::-1]],
            top_dev.values[::-1],
            color=colors[::-1],
            edgecolor=BG, linewidth=0.6)
    ax.set_title("Most Targeted Devices", fontsize=11, color=ACCENT, pad=10)
    ax.set_xlabel("Attack Flow Count")
    ax.grid(axis="x")


# ══════════════════════════════════════════════════════════════════════
#  RENDER DASHBOARD
# ══════════════════════════════════════════════════════════════════════
def render_dashboard(df, save=True):
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        "🔍  AI Network Anomaly Detection — deez.pkt Dashboard",
        fontsize=16, color=ACCENT, fontweight="bold", y=0.98
    )

    # Subtitle stats
    total     = len(df)
    anomalies = (df["rf_label"] != "Normal").sum()
    fig.text(
        0.5, 0.955,
        f"Network: Internet → R1 → ASA 5505 → DS1 → AS1/AS2/AS3  │  "
        f"Flows: {total}  │  Anomalies: {anomalies} ({anomalies/max(total,1):.1%})"
        f"  │  Classes: {df['rf_label'].nunique()}",
        ha="center", fontsize=9, color="#8b949e"
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.06)

    panel_class_distribution(fig.add_subplot(gs[0, 0]), df)
    panel_confidence_violin(fig.add_subplot(gs[0, 1]),  df)
    panel_anomaly_rate(fig.add_subplot(gs[0, 2]),       df)
    panel_top_ports(fig.add_subplot(gs[1, 0]),          df)
    panel_vlan_heatmap(fig.add_subplot(gs[1, 1]),       df)
    panel_top_targets(fig.add_subplot(gs[1, 2]),        df)

    if save:
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        print(f"[OK] Dashboard saved -> {OUT_PNG}")
    return fig


def main(live=False, interval=5):
    if live:
        print(f"[live mode] refreshing every {interval}s -- Ctrl+C to stop")
        while True:
            try:
                df  = load_log()
                fig = render_dashboard(df, save=True)
                plt.close(fig)
                time.sleep(interval)
            except KeyboardInterrupt:
                print("[!] Live mode stopped.")
                break
    else:
        df  = load_log()
        fig = render_dashboard(df, save=True)
        plt.close(fig)
        print("[OK] Dashboard rendered. Open reports/dashboard.png to view.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Dashboard — deez.pkt")
    parser.add_argument("--live",     action="store_true",
                        help="Auto-refresh dashboard")
    parser.add_argument("--interval", type=int, default=5,
                        help="Refresh interval in seconds (default: 5)")
    args = parser.parse_args()
    main(live=args.live, interval=args.interval)
