import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "performance_analysis.png")
CSV_PATH = None

def load_or_generate_data(csv_path=None, n_records=250, seed=42):
    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        return df

    rng = np.random.default_rng(seed)

    hours_studied   = rng.uniform(0, 12, n_records).round(1)
    attendance      = rng.integers(0, 101, n_records)
    previous_grade  = rng.uniform(20, 100, n_records).round(1)
    sleep_hours     = rng.uniform(3, 10, n_records).round(1)
    extracurricular = rng.integers(0, 2, n_records)

    final_grade = (
        0.30 * (hours_studied / 12) * 100
        + 0.25 * attendance
        + 0.25 * previous_grade
        + 0.05 * (sleep_hours / 10) * 100
        + 2.0 * extracurricular
        + rng.normal(0, 5, n_records)
    )

    final_grade = np.clip(final_grade, 0, 100).round(1)

    df = pd.DataFrame({
        "hours_studied":   hours_studied,
        "attendance":      attendance,
        "previous_grade":  previous_grade,
        "sleep_hours":     sleep_hours,
        "extracurricular": extracurricular,
        "final_grade":     final_grade,
    })

    idx_nan = rng.choice(n_records, size=5, replace=False)
    df.loc[idx_nan[:2], "hours_studied"]  = np.nan
    df.loc[idx_nan[2:4], "sleep_hours"]   = np.nan
    df.loc[idx_nan[4], "attendance"]      = np.nan

    df.loc[0, "final_grade"]  = 110.0
    df.loc[1, "final_grade"]  = -5.0

    return df

def clean_data(df):
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print("Missing values per column:\n", missing.to_string())
    
    if total_missing > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    df = df[(df["final_grade"] >= 0) & (df["final_grade"] <= 100)].copy()
    print("\nSummary statistics:\n", df.describe().round(2).to_string())
    return df

def correlation_analysis(df):
    corr_matrix = df.corr(numeric_only=True)
    print("\nPearson Correlation Matrix:\n", corr_matrix.round(3).to_string())

    target_corr = (
        corr_matrix["final_grade"]
        .drop("final_grade")
        .sort_values(ascending=False)
    )

    print("\nCorrelations with final_grade:\n", target_corr.to_string())

    top3 = target_corr.head(3)
    print("\nTop 3 factors influencing final_grade:")
    for rank, (feat, val) in enumerate(top3.items(), 1):
        print(f"{rank}. {feat} (Correlation: {val:.3f})")

    return target_corr

def create_visualizations(df, save_path=None):
    plt.rcParams.update({
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor":   "#1a1a2e",
        "axes.edgecolor":   "#444466",
        "axes.labelcolor":  "#ccccdd",
        "xtick.color":      "#aaaacc",
        "ytick.color":      "#aaaacc",
        "text.color":       "#eeeeff",
        "grid.color":       "#2a2a44",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "font.family":      "sans-serif",
        "font.size":        11,
    })

    C_CYAN   = "#00e5ff"
    C_PINK   = "#ff4081"
    C_GOLD   = "#ffd740"
    C_GREEN  = "#69f0ae"
    C_PURPLE = "#b388ff"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    grouped = df.groupby("extracurricular")["final_grade"]
    means   = grouped.mean()
    stds    = grouped.std()
    labels  = ["No Extracurricular\n(0)", "Extracurricular\n(1)"]
    colors  = [C_PINK, C_CYAN]

    bars = ax1.bar(labels, means, yerr=stds, capsize=6,
                   color=colors, edgecolor="#ffffff22", linewidth=1.2,
                   width=0.5, error_kw={"ecolor": "#ffffffaa", "elinewidth": 1.2})

    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                 f"{m:.1f}", ha="center", va="bottom", fontweight="bold",
                 fontsize=13, color=C_GOLD)

    ax1.set_title("Average Final Grade by Extracurricular Participation", fontweight="bold", pad=12)
    ax1.set_ylabel("Average Final Grade")
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax1.grid(axis="y")
    fig1.tight_layout()
    plt.show(block=False)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 13, 1)
    df["hours_bin"] = pd.cut(df["hours_studied"], bins=bins, right=False)
    bin_stats = df.groupby("hours_bin", observed=True)["final_grade"].agg(["mean", "std", "count"])
    bin_centers = [interval.left + 0.5 for interval in bin_stats.index]

    ax2.fill_between(bin_centers,
                     bin_stats["mean"] - bin_stats["std"],
                     bin_stats["mean"] + bin_stats["std"],
                     alpha=0.15, color=C_CYAN)

    ax2.plot(bin_centers, bin_stats["mean"], marker="o", markersize=7,
             linewidth=2.5, color=C_CYAN, markeredgecolor="#fff",
             markeredgewidth=1.2, zorder=5, label="Mean grade")

    ax2.set_title("Average Final Grade vs. Hours Studied", fontweight="bold", pad=12)
    ax2.set_xlabel("Hours Studied (1-hour bins)")
    ax2.set_ylabel("Average Final Grade")
    ax2.set_xlim(-0.2, 12.2)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax2.legend(loc="lower right", framealpha=0.6)
    ax2.grid(True)
    fig2.tight_layout()
    plt.show(block=False)
    
    df.drop(columns=["hours_bin"], inplace=True, errors="ignore")

    fig3 = plt.figure(figsize=(10, 6))
    gs   = fig3.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.04)
    ax3  = fig3.add_subplot(gs[0])
    ax3h = fig3.add_subplot(gs[1], sharey=ax3)

    ax3.scatter(df["attendance"], df["final_grade"],
                s=28, alpha=0.55, color=C_GREEN, edgecolors="#ffffff33",
                linewidth=0.5, zorder=3)

    z = np.polyfit(df["attendance"], df["final_grade"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["attendance"].min(), df["attendance"].max(), 200)
    ax3.plot(xs, p(xs), linewidth=2.5, color=C_PINK, linestyle="--",
             label=f"Trend: y = {z[0]:.2f}x + {z[1]:.1f}", zorder=4)

    ax3.set_title("Attendance vs. Final Grade (with Trend Line)", fontweight="bold", pad=12)
    ax3.set_xlabel("Attendance (%)")
    ax3.set_ylabel("Final Grade")
    ax3.legend(loc="upper left", framealpha=0.6, fontsize=10)
    ax3.grid(True)

    ax3h.hist(df["final_grade"], bins=20, orientation="horizontal",
              color=C_PURPLE, alpha=0.7, edgecolor="#ffffff22")
    ax3h.set_xlabel("Count")
    ax3h.tick_params(labelleft=False)
    ax3h.grid(axis="x")

    fig3.tight_layout()

    if save_path:
        fig3.savefig(save_path, dpi=180, bbox_inches="tight",
                     facecolor=fig3.get_facecolor())

    plt.show()

def main():
    df = load_or_generate_data(csv_path=CSV_PATH)
    df = clean_data(df)
    correlation_analysis(df)
    create_visualizations(df, save_path=OUTPUT_PNG)

if __name__ == "__main__":
    main()
