# %%
# Cross-dataset figures. Reads the result CSVs that memorisation_analysis.py writes to ./figs and,
# for the AUROC figures, the run directories under LOG_ROOT (default ./logs).
# Run as a script or cell by cell ('# %%') in VSCode/Jupyter.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, os
import logging
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

from src.utils import get_dataset_color_hex, get_dataset_str
from src.plotting import esf_plot


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force jax to use CPU only
FONT_SIZE = 6
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2.0, 2.0),
        "font.family": "sans-serif",
        "lines.linewidth": 1,
        "font.sans-serif": "Helvetica",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
    }
)
# silence fontTools' INFO chatter when embedding Type 42 fonts into PDFs
logging.getLogger("fontTools").setLevel(logging.ERROR)

# %%
fig_dir = Path("./figs")
# root of the run directories (layout in README.md)
LOG_ROOT = Path(os.environ.get("LOG_ROOT", "./logs"))

mimic_ecg_data = pd.read_csv(fig_dir / "mimic-ecg/files/mimic-ecg_mem_data.csv")
mimic_cxr_data = pd.read_csv(fig_dir / "mimic-cxr/files/mimic-cxr_mem_data.csv")
#mimic_ivdl_data = pd.read_csv(fig_dir / "mimic-iv-ed/files/mimic-iv-ed_mem_data.csv")
mimic_iv_data = pd.read_csv(fig_dir / "mimic-iv-ed_rf/files/mimic-iv-ed_rf_mem_data.csv")
heedb_data = pd.read_csv(fig_dir / "heedb/files/heedb_mem_data.csv")

# One HEEDB record reports an acquisition interval of 1016 months (~85 years), far beyond the
# next-largest interval of 489 months; it is a labelling error rather than a real follow-up.
# Drop it at load: every figure below reads heedb_data, so excluding it once keeps them consistent.
HEEDB_MAX_PLAUSIBLE_INTERVAL_MONTHS = 600
_heedb_implausible = heedb_data["time_deltas"] > HEEDB_MAX_PLAUSIBLE_INTERVAL_MONTHS
heedb_excluded_ids = set(heedb_data.loc[_heedb_implausible, "id"])
heedb_data = heedb_data[~_heedb_implausible].reset_index(drop=True)
print(f"HEEDB: dropped {len(heedb_excluded_ids)} implausible record(s), {len(heedb_data):,} left")

mimic_ecg_data_train = pd.read_csv(fig_dir / "mimic-ecg/files/mimic-ecg_mem_data_train.csv")
mimic_cxr_data_train = pd.read_csv(fig_dir / "mimic-cxr/files/mimic-cxr_mem_data_train.csv")
#mimic_iv_datadl_train = pd.read_csv(fig_dir / "mimic-iv-ed/files/mimic-iv-ed_mem_data_train.csv")
mimic_iv_data_train = pd.read_csv(fig_dir / "mimic-iv-ed_rf/files/mimic-iv-ed_rf_mem_data_train.csv")
heedb_data_train = pd.read_csv(fig_dir / "heedb/files/heedb_mem_data_train.csv")

# %%
# results for random partioning (sanity check as comparison)
mimic_ecg_data_random = pd.read_csv(fig_dir / "mimic-ecg/files/mimic-ecg_mem_data_le_sanity.csv")
mimic_cxr_data_random = pd.read_csv(fig_dir / "mimic-cxr/files/mimic-cxr_mem_data_le_sanity.csv")
mimic_iv_data_random = pd.read_csv(fig_dir / "mimic-iv-ed_rf/files/mimic-iv-ed_rf_mem_data_le_sanity.csv")
heedb_data_random = pd.read_csv(fig_dir / "heedb/files/heedb_mem_data_le_sanity.csv")
# the random-partitioning baseline is row-aligned with heedb_data (same records, same order) and
# is read positionally against it in effect_size_tail_survival, so drop the same record here to
# keep the two frames the same length
heedb_data_random = heedb_data_random[
    ~heedb_data_random["id"].isin(heedb_excluded_ids)
].reset_index(drop=True)

mimic_ecg_data_random_train = pd.read_csv(fig_dir / "mimic-ecg/files/mimic-ecg_mem_data_train_sanity.csv")
mimic_cxr_data_random_train = pd.read_csv(fig_dir / "mimic-cxr/files/mimic-cxr_mem_data_train_sanity.csv")
mimic_iv_data_random_train = pd.read_csv(fig_dir / "mimic-iv-ed_rf/files/mimic-iv-ed_rf_mem_data_train_sanity.csv")
heedb_data_random_train = pd.read_csv(fig_dir / "heedb/files/heedb_mem_data_train_sanity.csv")

# %%
mimic_cxr_data

# %%
mimic_cxr_data_train

# %%
mem_rates_future = {
    "mimic-ecg": mimic_ecg_data["rejected"].mean()*100,
    "mimic-cxr": mimic_cxr_data["rejected"].mean()*100,
    "mimic-iv-ed": mimic_iv_data["rejected"].mean()*100,
    "heedb": heedb_data["rejected"].mean()*100,
}
mem_rates_random_future = {
    "mimic-ecg": mimic_ecg_data_random["rejected"].mean()*100,
    "mimic-cxr": mimic_cxr_data_random["rejected"].mean()*100,
    "mimic-iv-ed": mimic_iv_data_random["rejected"].mean()*100,
    "heedb": heedb_data_random["rejected"].mean()*100,
}
rejected_counts_future = {
    "mimic-ecg": mimic_ecg_data["rejected"].sum(),
    "mimic-cxr": mimic_cxr_data["rejected"].sum(),
    "mimic-iv-ed": mimic_iv_data["rejected"].sum(),
    "heedb": heedb_data["rejected"].sum(),
}
total_counts_future = {
    "mimic-ecg": len(mimic_ecg_data),
    "mimic-cxr": len(mimic_cxr_data),
    "mimic-iv-ed": len(mimic_iv_data),
    "heedb": len(heedb_data),
}

def fmt_count(x):
    # commas for readable counts, scientific notation (x 10^Y) above 100k to save space
    if x <= 100_000:
        return f"{x:,}"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def fit_ylim_to_labels(axes, pad=0.08):
    # expand the top y-limit so rotated text labels stay within the axes
    axes = np.atleast_1d(axes)
    fig = axes.flat[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = 0.0
    for ax in axes.flat:
        for txt in ax.texts:
            y_top = ax.transData.inverted().transform((0, txt.get_window_extent(renderer).y1))[1]
            top = max(top, y_top)
    for ax in axes.flat:
        ax.set_ylim(0, top * (1 + pad))

# plot
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0), width_ratios=[0.7, 1], layout="constrained")
bars = axs[0].bar(
    mem_rates_future.keys(),
    mem_rates_future.values(),
    color=[get_dataset_color_hex(dataset) for dataset in mem_rates_future.keys()],
)
# Add N=X labels above bars
for bar, dataset in zip(bars, mem_rates_future.keys()):
    axs[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height()+0.1 if dataset in ["mimic-ecg", "heedb"] else bar.get_height()-0.05,
        f"{fmt_count(rejected_counts_future[dataset])} of {fmt_count(total_counts_future[dataset])}",
        ha="center",
        va="bottom" if dataset in ["mimic-ecg", "heedb"] else "top",
        fontsize=FONT_SIZE,
        rotation=90,
    )
axs[0].set_ylabel("Future records with significant\n change in prediction (%)")
axs[0].set_xticklabels([])
axs[0].set_xlabel("Dataset")
axs[0].spines[["right", "top"]].set_visible(False)
esf_plot(
    data_arr=mimic_ecg_data["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-ecg"),
)
esf_plot(
    data_arr=mimic_cxr_data["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-cxr"),
)
esf_plot(
    data_arr=mimic_iv_data["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-iv-ed"),
)
esf_plot(
    data_arr=heedb_data["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("heedb"),
)
# random partitioning (sanity check)
esf_plot(
    data_arr=mimic_ecg_data_random["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-ecg"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=mimic_cxr_data_random["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-cxr"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=mimic_iv_data_random["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-iv-ed"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=heedb_data_random["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    line_style="--",
    color=get_dataset_color_hex("heedb"),
)
axs[1].set_yscale("log")
axs[1].set_xlim(0, 90)
axs[1].set_xlabel("Change in predicted probability (%)", fontsize=FONT_SIZE)
axs[1].set_ylabel("1 - cumulative probability", fontsize=FONT_SIZE)
axs[1].yaxis.set_minor_locator(ticker.NullLocator())
#max_n_long_eval = max(len(mimic_ecg_data), len(mimic_cxr_data), len(mimic_iv_data), len(heedb_data))
handles = [Line2D([0], [0], color="black", linestyle="--", label="Random partitioning")]
axs[1].legend(handles=handles, fontsize=FONT_SIZE, loc="upper right", frameon=False)
axs[1].set_ylim(1e-7, 1)
fit_ylim_to_labels(axs[0])
axs[0].set_ylim((0,5))
plt.savefig(fig_dir / "mem_analysis_future.pdf", dpi=300)

# %%
mem_rates_train_historical = {
    "mimic-ecg": mimic_ecg_data_train["rejected"].mean()*100,
    "mimic-cxr": mimic_cxr_data_train["rejected"].mean()*100,
    "mimic-iv-ed": mimic_iv_data_train["rejected"].mean()*100,
    "heedb": heedb_data_train["rejected"].mean()*100,
}
mem_rates_random_train_historical = {
    "mimic-ecg": mimic_ecg_data_random_train["rejected"].mean()*100,
    "mimic-cxr": mimic_cxr_data_random_train["rejected"].mean()*100,
    "mimic-iv-ed": mimic_iv_data_random_train["rejected"].mean()*100,
    "heedb": heedb_data_random_train["rejected"].mean()*100,
}
rejected_counts_historical = {
    "mimic-ecg": mimic_ecg_data_train["rejected"].sum(),
    "mimic-cxr": mimic_cxr_data_train["rejected"].sum(),
    "mimic-iv-ed": mimic_iv_data_train["rejected"].sum(),
    "heedb": heedb_data_train["rejected"].sum(),
}
total_counts_historical = {
    "mimic-ecg": len(mimic_ecg_data_train),
    "mimic-cxr": len(mimic_cxr_data_train),
    "mimic-iv-ed": len(mimic_iv_data_train),
    "heedb": len(heedb_data_train),
}
def fmt_count(x):
    # commas for readable counts, scientific notation (x 10^Y) above 100k to save space
    if x <= 100_000:
        return f"{x:,}"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def fit_ylim_to_labels(axes, pad=0.08):
    # expand the top y-limit so rotated text labels stay within the axes
    axes = np.atleast_1d(axes)
    fig = axes.flat[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = 0.0
    for ax in axes.flat:
        for txt in ax.texts:
            y_top = ax.transData.inverted().transform((0, txt.get_window_extent(renderer).y1))[1]
            top = max(top, y_top)
    for ax in axes.flat:
        ax.set_ylim(0, top * (1 + pad))

# plot for training data
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.0), width_ratios=[0.7, 1], layout="constrained")
bars = axs[0].bar(
    mem_rates_train_historical.keys(),
    mem_rates_train_historical.values(),
    color=[get_dataset_color_hex(dataset) for dataset in mem_rates_train_historical.keys()],
)
# Add N=X labels above bars
for bar, dataset in zip(bars, mem_rates_train_historical.keys()):
    axs[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 2.0 if dataset in ["mimic-ecg", "heedb"] else bar.get_height() - 2.0,
        f"{fmt_count(rejected_counts_historical[dataset])} of {fmt_count(total_counts_historical[dataset])}",
        ha="center",
        va="bottom" if dataset in ["mimic-ecg", "heedb"] else "top",
        fontsize=FONT_SIZE,
        rotation=90,
    )
axs[0].set_ylabel("Historical records with significant\n change in prediction (%)")
axs[0].set_xticklabels([])
axs[0].set_xlabel("Dataset")
axs[0].spines[["right", "top"]].set_visible(False)
esf_plot(
    data_arr=mimic_ecg_data_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-ecg"),
)
esf_plot(
    data_arr=mimic_cxr_data_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-cxr"),
)
esf_plot(
    data_arr=mimic_iv_data_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-iv-ed"),
)
esf_plot(
    data_arr=heedb_data_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("heedb"),
)
# random partitioning (sanity check)
esf_plot(
    data_arr=mimic_ecg_data_random_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-ecg"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=mimic_cxr_data_random_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-cxr"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=mimic_iv_data_random_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    color=get_dataset_color_hex("mimic-iv-ed"),
    line_style="--",
    alpha=0.75,
)
esf_plot(
    data_arr=heedb_data_random_train["effect_size"]*100,
    xlabel="",
    draw_conf=False,
    ax=axs[1],
    line_style="--",
    color=get_dataset_color_hex("heedb"),
)
axs[1].set_yscale("log")
axs[1].set_xlim(0, 100)
axs[1].set_xlabel("Change in predicted probability (%)", fontsize=FONT_SIZE)
axs[1].set_ylabel("1 - Cumulative Probability", fontsize=FONT_SIZE)
axs[1].yaxis.set_minor_locator(ticker.NullLocator())
handles = [Line2D([0], [0], color="black", linestyle="--", label="Random partitioning")]
axs[1].legend(handles=handles, fontsize=FONT_SIZE, loc="upper right", frameon=False)
axs[1].set_ylim(1e-7, 1)
fit_ylim_to_labels(axs[0])
axs[0].set_ylim((0,100))
plt.savefig(fig_dir / "mem_analysis_historical.pdf", dpi=300)

# %%
# manhattan plot of time deltas vs. -log10(pvals)
fig, axs = plt.subplots(1, 4, figsize=(7.05, 2.0), sharey=True)
ALPHA=0.3
sc0 = axs[0].scatter(
    mimic_ecg_data["time_deltas"],
    np.clip(-np.log10(mimic_ecg_data["pvals(corrected)"]), a_min=0, a_max=20.0),
    color=get_dataset_color_hex("mimic-ecg"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-ecg')}",
    zorder=0
)
sc1 = axs[1].scatter(
    mimic_cxr_data["time_deltas"],
    np.clip(-np.log10(mimic_cxr_data["pvals(corrected)"]), a_min=0, a_max=20.0),
    color=get_dataset_color_hex("mimic-cxr"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-cxr')}",
    zorder=0
)
sc2= axs[2].scatter(
    mimic_iv_data["time_deltas"],
    np.clip(-np.log10(mimic_iv_data["pvals(corrected)"]), a_min=0, a_max=20.0),
    color=get_dataset_color_hex("mimic-iv-ed"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-iv-ed')}",
    zorder=0
)
sc3 = axs[3].scatter(
    heedb_data["time_deltas"],
    np.clip(-np.log10(heedb_data["pvals(corrected)"]), a_min=0, a_max=20.0),
    color=get_dataset_color_hex("heedb"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('heedb')}",
    zorder=0
)
for sc in [sc0, sc1, sc2, sc3]:
    sc.set_rasterized(True) # rasterize for better pdf viewing performance and smaller file size
for ax in axs.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(
    y=-np.log10(0.05),
    color="black",
    linestyle="--",
    alpha=0.7,
    label=f"Significance Threshold)",
    zorder=2
    )
    #ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_ylim(0, 20.05)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_xlabel("Acquisition interval (months)", fontsize=FONT_SIZE)
axs[0].set_ylabel(r"$-\log_{10}(p)$", fontsize=FONT_SIZE)
axs[3].set_xlim(0,500)
handles = [Line2D([0], [0], color="black", linestyle="--", label="Significance Threshold")]
axs[3].legend(
    handles=handles,
    loc="upper right",
    ncol=1,
    fontsize=FONT_SIZE,
    #bbox_to_anchor=(0.5, 1.125),
    frameon=False,
    framealpha=0.0,
)
# for ax, datadf in zip(axs.flatten(), [mimic_ecg_data, mimic_cxr_data, mimic_iv_data, heedb_data]):
#     ax.text(
#         0.95,
#         0.5,
#         f"N={len(datadf):,}",
#         ha="right",
#         va="top",
#         fontsize=6,
#         transform=ax.transAxes,
#     )
plt.savefig(fig_dir / "manhattan_plot.pdf", dpi=300)

# %%
# eSF of the acquisition intervals of future records, one panel per dataset.
# Dashed = all future records, solid = the subset with a significant change in prediction.
# Same dataset order and panel geometry as the manhattan row above, so the two rows stack.
interval_dfs = {
    "mimic-ecg": mimic_ecg_data,
    "mimic-cxr": mimic_cxr_data,
    "mimic-iv-ed": mimic_iv_data,
    "heedb": heedb_data,
}
# x tick spacing copied from the manhattan row: this row is narrower, so the auto locator
# thins the ticks out and the two rows stop lining up when stacked
XTICK_STEP = {"mimic-ecg": 25, "mimic-cxr": 20, "mimic-iv-ed": 20, "heedb": 100}
fig, axs = plt.subplots(1, 4, figsize=(4.8, 1.9), sharey=True)
for ax, (ds, df) in zip(axs.flatten(), interval_dfs.items()):
    memorised = df[df["rejected"]]
    esf_plot(  # all future records
        data_arr=df["time_deltas"].values,
        xlabel="Acquisition interval (months)",
        draw_conf=False,
        color=get_dataset_color_hex(ds),
        line_style="--",
        alpha=0.75,
        ax=ax,
    )
    esf_plot(  # only the records where memorisation was detected
        data_arr=memorised["time_deltas"].values,
        xlabel="Acquisition interval (months)",
        draw_conf=False,
        color=get_dataset_color_hex(ds),
        ax=ax,
    )
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(XTICK_STEP[ds]))
fig.supxlabel("Acquisition interval (months)", fontsize=FONT_SIZE)
# heedb clipped to the manhattan row's range so the two rows stack; the longest retained interval
# is 489 months, so this only trims empty space
axs[3].set_xlim(0, 500)
# one legend for the whole row: the partitioning is the same in every panel, and the datasets
# are identified by colour and position as in the manhattan row
handles = [
    Line2D([0], [0], color="black", linestyle="--", label="All records"),
    Line2D([0], [0], color="black", linestyle="-", label="Memorised records"),
]
axs[3].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False)
# joint y-axis (sharey): the survival function floors at 1/N, so scale it to the largest cohort
axs[0].set_ylim(0.5 / max(len(df) for df in interval_dfs.values()), 1)
axs[0].set_ylabel("1 - cumulative probability", fontsize=FONT_SIZE)
fig.savefig(fig_dir / "acquisition_interval_esf.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# for name, df in zip(["mimic-ecg", "mimic-cxr", "mimic-iv-ed", "heedb"], [mimic_ecg_data, mimic_cxr_data, mimic_iv_data, heedb_data]):
#     df["color"] = df["rejected"]
#     c = get_dataset_color_hex(name)
#     df['color'] = df['color'].map({True: '#A51C30', False: c})
#     print("computed colors")

# %%
heedb_data.loc[heedb_data.rejected, "effect_size"]

# %%
# time deltas vs. effect size for records with a significant change in predicted probability (i.e. rejected records)
fig, axs = plt.subplots(1, 4, figsize=(7.05, 1.7), sharey=True)
ALPHA=0.3

sc0 = axs[0].scatter(
    mimic_ecg_data.loc[mimic_ecg_data.rejected, "time_deltas"],
    mimic_ecg_data.loc[mimic_ecg_data.rejected, "effect_size"]*100,
    color=get_dataset_color_hex("mimic-ecg"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-ecg')}",
)
sc1 = axs[1].scatter(
    mimic_cxr_data.loc[mimic_cxr_data.rejected,"time_deltas"],
    mimic_cxr_data.loc[mimic_cxr_data.rejected, "effect_size"]*100,
    color=get_dataset_color_hex("mimic-cxr"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-cxr')}",
)
sc2= axs[2].scatter(
    mimic_iv_data.loc[mimic_iv_data.rejected,"time_deltas"],
    mimic_iv_data.loc[mimic_iv_data.rejected, "effect_size"]*100,
    color=get_dataset_color_hex("mimic-iv-ed"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('mimic-iv-ed')}",
)
sc3 = axs[3].scatter(
    heedb_data.loc[heedb_data.rejected, "time_deltas"],
    heedb_data.loc[heedb_data.rejected, "effect_size"]*100,
    color=get_dataset_color_hex("heedb"),
    s=1,
    alpha=ALPHA,
    label=f"{get_dataset_str('heedb')}",
)
for sc in [sc0, sc1, sc2, sc3]:
    sc.set_rasterized(True) # rasterize for better pdf viewing performance and smaller file size
for ax in axs.flatten():
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_xlabel("Acquisition interval (months)", fontsize=FONT_SIZE)
axs[0].set_ylabel("Change in predicted probability \n(%)", fontsize=FONT_SIZE)
axs[3].set_xlim(0,500)
# display N count in to right corner
for ax, dataset in zip(axs.flatten(), ["mimic-ecg", "mimic-cxr", "mimic-iv-ed", "heedb"]):
    ax.text(
        0.95,
        0.95,
        f"N={rejected_counts_future[dataset]:,}",
        ha="right",
        va="top",
        fontsize=6,
        transform=ax.transAxes,
    )
plt.savefig(fig_dir / "elapsed_time_vs_effect_size.pdf", dpi=300, bbox_inches="tight")

# %%
from typing import Tuple, List

def calculate_binned_memorisation_rates(logdata:pd.DataFrame, bins:List[Tuple[int, int]]):
    max_time_delta = int(logdata["time_deltas"].max())
    print(f"Max time delta: {max_time_delta} months")
    print(f"Using bins: {bins}")
    bin_labels = []
    mem_rates = []
    counts = []
    rejected_counts = []

    for i, (start, end) in enumerate(bins):
        # Filter data within the bin range. Bins are half-open [start, end) so neighbouring bins
        # never double count, except the last one, which closes at [start, end] so records sitting
        # exactly on the final edge are counted instead of silently dropped.
        is_last = i == len(bins) - 1
        within_upper = (
            logdata["time_deltas"] <= end if is_last else logdata["time_deltas"] < end
        )
        mask = (logdata["time_deltas"] >= start) & within_upper
        bin_df = logdata[mask]

        if len(bin_df) > 0:
            mem_rate = bin_df["rejected"].mean()*100
            count = len(bin_df)
            rejected_count = bin_df["rejected"].sum()
        else:
            mem_rate = 0
            count = 0
            rejected_count = 0

        # Create label
        label = f"{start}-{end}"

        bin_labels.append(label)
        mem_rates.append(mem_rate)
        counts.append(count)
        rejected_counts.append(rejected_count)
    # Each bar is labelled with its own "n of N", and those N's are read as a breakdown of the
    # dataset total shown in mem_analysis_future.pdf, so the bins must span every record. Fail
    # loudly rather than let uncovered records make the two figures disagree.
    if sum(counts) != len(logdata):
        raise ValueError(
            f"bins {bins} cover {sum(counts):,} of {len(logdata):,} records; "
            f"{len(logdata) - sum(counts):,} fall outside "
            f"(time deltas span {int(logdata['time_deltas'].min())}-{max_time_delta} months)"
        )
    results_dict = {
        "bin_labels": bin_labels,
        "mem_rates": mem_rates,
        "counts": counts,
        "rejected_counts": rejected_counts,
    }
    return results_dict

# %%
# bin widths are non-decreasing with acquisition interval, so a wider bar never sits to the left of
# a narrower one. 24-48 and 48-60 are merged into 24-60 for that reason.
# The final bin is closed, so its upper edge must reach each dataset's longest interval (129, 60,
# 97 and 489 months) or calculate_binned_memorisation_rates raises. HEEDB ends at 500 to match the
# 500-month x-range its panels use elsewhere.
mimic_ecg_binned_mem_rates = calculate_binned_memorisation_rates(mimic_ecg_data, bins=[(0, 6), (6, 12), (12, 18), (18, 24), (24, 60), (60, 130)])
mimic_cxr_binned_mem_rates = calculate_binned_memorisation_rates(mimic_cxr_data, bins=[(0, 6), (6, 12), (12, 18), (18, 24), (24, 60)])
mimiv_iv_binned_mem_rates = calculate_binned_memorisation_rates(mimic_iv_data, bins=[(0, 6), (6, 12), (12, 18), (18, 24), (24, 60), (60, 97)])
heedb_binned_mem_rates = calculate_binned_memorisation_rates(heedb_data, bins=[(0, 12), (12, 24), (24, 60), (60, 120), (120, 240), (240, 500)])

fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.0), sharey=True)

def fmt_count(x):
    # commas for readable counts, scientific notation (x 10^Y) above 100k to save space
    if x <= 100_000:
        return f"{x:,}"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def fit_ylim_to_labels(axes, pad=0.08):
    # expand the top y-limit so rotated text labels stay within the axes
    axes = np.atleast_1d(axes)
    fig = axes.flat[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = 0.0
    for ax in axes.flat:
        for txt in ax.texts:
            y_top = ax.transData.inverted().transform((0, txt.get_window_extent(renderer).y1))[1]
            top = max(top, y_top)
    for ax in axes.flat:
        ax.set_ylim(0, top * (1 + pad))

def add_n_labels(ax, mem_rates, rejected_counts, counts, fontsize=6, below_threshold=4.5):
    bars = ax.patches
    for bar, rate, n, total in zip(bars, mem_rates, rejected_counts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height()+0.15 if rate <= below_threshold else bar.get_height()-0.15,
            f"{fmt_count(n)} of {fmt_count(total)}",
            ha="center",
            va="bottom" if rate <= below_threshold else "top",
            fontsize=fontsize,
            rotation=90,
        )

axs[0].bar(
    mimic_ecg_binned_mem_rates["bin_labels"],
    mimic_ecg_binned_mem_rates["mem_rates"],
    color=get_dataset_color_hex("mimic-ecg"),
)
add_n_labels(axs[0], mimic_ecg_binned_mem_rates["mem_rates"], mimic_ecg_binned_mem_rates["rejected_counts"], mimic_ecg_binned_mem_rates["counts"])

axs[1].bar(
    mimic_cxr_binned_mem_rates["bin_labels"],
    mimic_cxr_binned_mem_rates["mem_rates"],
    color=get_dataset_color_hex("mimic-cxr"),
)
add_n_labels(axs[1], mimic_cxr_binned_mem_rates["mem_rates"], mimic_cxr_binned_mem_rates["rejected_counts"], mimic_cxr_binned_mem_rates["counts"])

axs[2].bar(
    mimiv_iv_binned_mem_rates["bin_labels"],
    mimiv_iv_binned_mem_rates["mem_rates"],
    color=get_dataset_color_hex("mimic-iv-ed"),
)
add_n_labels(axs[2], mimiv_iv_binned_mem_rates["mem_rates"], mimiv_iv_binned_mem_rates["rejected_counts"], mimiv_iv_binned_mem_rates["counts"])
axs[3].bar(
    heedb_binned_mem_rates["bin_labels"],
    heedb_binned_mem_rates["mem_rates"],
    color=get_dataset_color_hex("heedb"),
)
add_n_labels(axs[3], heedb_binned_mem_rates["mem_rates"], heedb_binned_mem_rates["rejected_counts"], heedb_binned_mem_rates["counts"])

axs[0].set_ylabel("Future records with significant\n change in prediction (%)")
for ax in axs.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Acquisition interval (months)", fontsize=FONT_SIZE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
fit_ylim_to_labels(axs)
plt.savefig(fig_dir / "binned_memorisation_rates.pdf", dpi=300, bbox_inches="tight")

# %%
plt.figure()
handles = [mpatches.Patch(color=get_dataset_color_hex(dataset), label=get_dataset_str(dataset)) for dataset in ["mimic-ecg", "mimic-cxr", "mimic-iv-ed", "heedb"]]
plt.legend(
    handles=handles,
    loc="upper right",
    ncol=1,
    fontsize=FONT_SIZE,
    #bbox_to_anchor=(0.5, 1.125),
    frameon=False,
    framealpha=0.0,
)
plt.savefig(fig_dir / "legend.pdf", dpi=300, bbox_inches="tight")

# %%
# --- Per-model test-set macro AUROC: random variation across random subset models ---
# The held-out test set is drawn from patients disjoint from the historical/future
# datasets, so every random-subset model is an OUT model w.r.t. the test set. This
# therefore quantifies the overall random variation in test-set performance that
# arises purely from training each model on a different random subset.
auroc_logdirs = {
    "mimic-ecg":   LOG_ROOT / "mimic-ecg/vit_s",
    "mimic-cxr":   LOG_ROOT / "mimic-cxr",
    "mimic-iv-ed": LOG_ROOT / "mimic-iv-ed-rf",
    "heedb":       LOG_ROOT / "heedb/vit_s",
}

def collect_test_auroc(root: Path) -> np.ndarray:
    """Collect per-model test-set macro AUROC (%) from each run's info.json."""
    scores = []
    for run_dir in sorted(root.iterdir()):
        info_path = run_dir / "info.json"
        if not run_dir.is_dir() or not info_path.exists():
            continue
        with open(info_path) as f:
            tm = json.load(f).get("test_metrics", {})
        # field name differs across datasets and model types: the keras/jax runs prefix the
        # metric with an underscore (_test_macro_auroc / _test_macro_auc), the sklearn ones don't
        auroc = next(
            (
                tm[k]
                for k in ("_test_macro_auroc", "_test_macro_auc", "test_macro_auroc", "test_macro_auc")
                if k in tm
            ),
            None,
        )
        if auroc is not None:
            scores.append(float(auroc))
    return np.array(scores) * 100

test_auroc = {ds: collect_test_auroc(root) for ds, root in auroc_logdirs.items()}
for ds, v in test_auroc.items():
    print(f"{ds:12s} n={len(v):4d}  mean={v.mean():.3f}%  std={v.std():.3f}%")

# %%
# Random variation in test-set macro AUROC across all random subset models (single panel)
np.random.seed(0)
datasets = ["mimic-ecg", "mimic-cxr", "mimic-iv-ed", "heedb"]
fig, ax = plt.subplots(figsize=(2.0, 1.9))
for i, ds in enumerate(datasets):
    vals = test_auroc[ds]
    color = get_dataset_color_hex(ds)
    jitter = np.random.uniform(-0.06, 0.06, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, s=2, alpha=0.35, zorder=1, rasterized=True)
    m, s = vals.mean(), vals.std()
    ax.errorbar(
        i, m, yerr=s, fmt="o", color="white", markeredgecolor="black",
        markeredgewidth=0.7, markersize=4, ecolor="black", capsize=3,
        elinewidth=0.7, zorder=3,
    )
    ax.text(
        i + 0.15, m, f"{m:.2f}$\\pm${s:.2f}",
        ha="left", va="center", fontsize=FONT_SIZE,
    )
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels([get_dataset_str(ds) for ds in datasets], rotation=45, ha="center")
ax.set_xlim(-0.5, len(datasets) - 0.5)
ax.set_ylabel("Diagnostic performance (%)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(fig_dir / "test_auroc_random_variation.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Record-level test statistics vs. months since last data contribution
teststat_dfs = {
    "mimic-ecg": mimic_ecg_data,
    "mimic-cxr": mimic_cxr_data,
    "mimic-iv-ed": mimic_iv_data,
    "heedb": heedb_data,
}
teststat_random_dfs = {
    "mimic-ecg": mimic_ecg_data_random,
    "mimic-cxr": mimic_cxr_data_random,
    "mimic-iv-ed": mimic_iv_data_random,
    "heedb": heedb_data_random,
}
available = {ds: df for ds, df in teststat_dfs.items() if "test_statistic" in df.columns}
for ds in teststat_dfs:
    if ds not in available:
        print(f"WARNING: {ds} mem_data CSV has no 'test_statistic' column, skipping (re-run memorisation_analysis.py)")

fig, axs = plt.subplots(1, len(teststat_dfs), figsize=(4.7, 1.9), sharey=True)
ALPHA = 0.3
for ax, (ds, df) in zip(axs, teststat_dfs.items()):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Acquisition interval (months)")
    if ds not in available:
        continue
    sc = ax.scatter(
        df["time_deltas"],
        df["test_statistic"],
        color=get_dataset_color_hex(ds),
        s=1,
        alpha=ALPHA,
        label=f"{get_dataset_str(ds)}",
    )
    sc.set_rasterized(True)  # rasterize for better pdf viewing performance and smaller file size
    #ax.legend(loc="upper right", fontsize=FONT_SIZE, frameon=False)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    # random-partitioning baseline: max test statistic over all randomly partitioned records
    df_rand = teststat_random_dfs[ds]
    # if "test_statistic" in df_rand.columns:
    #     ax.axhline(df_rand["test_statistic"].max(), color="black", lw=1, linestyle="--", alpha=0.8)
axs[0].set_ylabel("Energy test statistic")
axs[3].set_xlim(0, 500)  # match HEEDB x-range of the p-value plot
#handles = [Line2D([0], [0], color="black", lw=1, linestyle="--", alpha=0.8, label="Random-partitioning maximum")]
#axs[3].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False)
fig.savefig(fig_dir / "test_statistic_vs_time_delta.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Empirical survival functions of the record-level test statistics.
# Left = future records, right = historical (training) records; shared (joint) y-axis, independent x-axes.
# solid = memorisation partitioning, dashed = random-partitioning baseline (sanity check)
teststat_random_dfs = {
    "mimic-ecg": mimic_ecg_data_random,
    "mimic-cxr": mimic_cxr_data_random,
    "mimic-iv-ed": mimic_iv_data_random,
    "heedb": heedb_data_random,
}
teststat_train_dfs = {
    "mimic-ecg": mimic_ecg_data_train,
    "mimic-cxr": mimic_cxr_data_train,
    "mimic-iv-ed": mimic_iv_data_train,
    "heedb": heedb_data_train,
}
teststat_random_train_dfs = {
    "mimic-ecg": mimic_ecg_data_random_train,
    "mimic-cxr": mimic_cxr_data_random_train,
    "mimic-iv-ed": mimic_iv_data_random_train,
    "heedb": heedb_data_random_train,
}
has_ts = lambda d: {ds: df for ds, df in d.items() if "test_statistic" in df.columns}
available_random = has_ts(teststat_random_dfs)
available_train = has_ts(teststat_train_dfs)
available_random_train = has_ts(teststat_random_train_dfs)

# (title, solid dfs, available-solid, random dfs, available-random, xlim); right x-limit None => autoscale
panels = [
    ("Future records", teststat_dfs, available, teststat_random_dfs, available_random, (0, 1.0)),
    ("Historical records", teststat_train_dfs, available_train, teststat_random_train_dfs, available_random_train, (0, None)),
]

fig, axs = plt.subplots(1, 2, figsize=(4.0, 1.9), sharey=True)
for ax, (title, solid_dfs, avail_solid, rand_dfs, avail_rand, xlim) in zip(axs, panels):
    for ds in solid_dfs:
        if ds not in avail_solid:
            continue
        esf_plot(
            data_arr=solid_dfs[ds]["test_statistic"].values,
            xlabel="Energy test statistic",
            color=get_dataset_color_hex(ds),
            label=get_dataset_str(ds),
            ax=ax,
        )
        # random-partitioning baseline (dashed)
        if ds in avail_rand:
            esf_plot(
                data_arr=rand_dfs[ds]["test_statistic"].values,
                xlabel="Energy test statistic",
                color=get_dataset_color_hex(ds),
                line_style="--",
                alpha=0.75,
                ax=ax,
            )
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Energy test statistic")
    ax.set_xlim(xlim[0], xlim[1])  # independent x-axis per panel
    #ax.set_title(title, fontsize=FONT_SIZE)

# joint y-axis: survival function floors at 1/N over all shown records (both panels)
n_counts = [len(df) for avail in (available, available_train) for df in avail.values()]
if n_counts:
    axs[0].set_ylim(0.5 / max(n_counts), 1)
axs[0].set_ylabel("1 - cumulative probability", fontsize=FONT_SIZE)

# legend (datasets + random-partitioning baseline) on the future panel (cleaner upper-right)
legend_ds = list({**available, **available_train}.keys())
if legend_ds:
    handles = [Line2D([0], [0], color=get_dataset_color_hex(ds), label=get_dataset_str(ds)) for ds in legend_ds]
    handles.append(Line2D([0], [0], color="black", linestyle="--", label="Random partitioning"))
    axs[0].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False, ncol=1)
fig.savefig(fig_dir / "test_statistic_esf.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Survival-analysis view over records that "survive" to a minimum acquisition interval X.
# Top row: effect size vs. acquisition interval for records with a significant change (rejected).
# Lower rows = 99th pct, 99.9th pct, and maximum change in predicted probability.
# Columns = datasets.
# Solid = memorisation partitioning, dashed = random-partitioning baseline (sanity check).
survival_dfs = {
    "mimic-ecg": (mimic_ecg_data, mimic_ecg_data_random),
    "mimic-cxr": (mimic_cxr_data, mimic_cxr_data_random),
    "mimic-iv-ed": (mimic_iv_data, mimic_iv_data_random),
    "heedb": (heedb_data, heedb_data_random),
}
MIN_SURVIVORS = 500  # stop each curve once fewer than this many records remain (avoids noisy tail)
ALPHA = 0.3

def survivor_sweep(intervals, effect, X):
    p99 = np.array([np.percentile(effect[intervals >= x], 99) for x in X])
    p999 = np.array([np.percentile(effect[intervals >= x], 99.9) for x in X])
    mx = np.array([effect[intervals >= x].max() for x in X])
    return p99, p999, mx

fig, axs = plt.subplots(4, 4, figsize=(7.05, 6.69))
for j, (ds, (df, df_rand)) in enumerate(survival_dfs.items()):
    intervals = df["time_deltas"].values
    effect = df["effect_size"].values * 100
    rejected = df["rejected"].values  # records with a significant change in predicted probability
    # random baseline shares the same longitudinal-eval records (id-aligned), so reuse the intervals
    effect_rand = df_rand["effect_size"].values * 100
    color = get_dataset_color_hex(ds)
    # largest X that still leaves >= MIN_SURVIVORS records with interval >= X
    x_max = np.sort(intervals)[-MIN_SURVIVORS] if len(intervals) >= MIN_SURVIVORS else intervals.max()
    X = np.linspace(0, x_max, 150)
    stats = survivor_sweep(intervals, effect, X)
    stats_r = survivor_sweep(intervals, effect_rand, X)

    # top row: effect size vs. acquisition interval (significant / rejected records only)
    ax = axs[0, j]
    sc = ax.scatter(intervals[rejected], effect[rejected], color=color, s=1, alpha=ALPHA)
    sc.set_rasterized(True)  # rasterize for smaller/faster pdf
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticklabels([])
    ax.set_title(get_dataset_str(ds), fontsize=FONT_SIZE)

    for row in range(3):
        ax = axs[row + 1, j]
        ax.plot(X, stats[row], color=color, lw=1)
        ax.plot(X, stats_r[row], color=color, lw=1, linestyle="--", alpha=0.75)
        ax.set_xlim(0, x_max)
        #ax.set_ylim(0, ax.get_ylim()[1])
        ax.spines[["top", "right"]].set_visible(False)
        if row < 2:
            ax.set_xticklabels([])
fig.supxlabel("Acquisition interval (months)", fontsize=FONT_SIZE)
axs[0, 0].set_ylabel("Change in\npredicted probability\n(%)", fontsize=FONT_SIZE)
axs[1, 0].set_ylabel("99th percentile change in\npredicted probability\n(%)", fontsize=FONT_SIZE)
axs[2, 0].set_ylabel("99.9th percentile change in\npredicted probability\n(%)", fontsize=FONT_SIZE)
axs[3, 0].set_ylabel("Maximum change in\npredicted probability\n(%)", fontsize=FONT_SIZE)
handles = [Line2D([0], [0], color="black", lw=1, linestyle="--", alpha=0.75, label="Random partitioning")]
axs[1, 3].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False)
axs[2, 3].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False)
axs[3, 3].legend(handles=handles, loc="upper right", fontsize=FONT_SIZE, frameon=False)
fig.savefig(fig_dir / "effect_size_tail_survival.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# --- Model architecture comparison (MIMIC-IV-ED) ---
# All three architectures were trained on MIMIC-IV-ED with the identical random-subset
# protocol (same splits, same n_runs), so differences below are attributable to the
# model architecture rather than to the data or the evaluation procedure.
ARCH_ORDER = ["lr", "rf", "tabresnet"]
ARCH_DIRS = {"lr": "mimic-iv-ed_lr", "rf": "mimic-iv-ed_rf", "tabresnet": "mimic-iv-ed_tabresnet"}
ARCH_STR = {"lr": "LR", "rf": "RF", "tabresnet": "DNN"}
# colourblind-safe categorical palette (Okabe-Ito), assigned per architecture in fixed order
ARCH_COLOR = {"lr": "#1D609E", "rf": "#A23B72", "tabresnet": "#F18F01"}


def load_arch_data(arch, suffix=""):
    d = ARCH_DIRS[arch]
    return pd.read_csv(fig_dir / f"{d}/files/{d}_mem_data{suffix}.csv")


# future (longitudinal eval) and historical (training) records, plus the
# random-partitioning sanity check for each
arch_future = {a: load_arch_data(a) for a in ARCH_ORDER}
arch_historical = {a: load_arch_data(a, "_train") for a in ARCH_ORDER}
arch_future_random = {a: load_arch_data(a, "_le_sanity") for a in ARCH_ORDER}
arch_historical_random = {a: load_arch_data(a, "_train_sanity") for a in ARCH_ORDER}

for a in ARCH_ORDER:
    print(
        f"{ARCH_STR[a]:20s} future: {arch_future[a]['rejected'].sum():6d}/{len(arch_future[a]):7d} "
        f"({arch_future[a]['rejected'].mean()*100:5.2f}%)   "
        f"historical: {arch_historical[a]['rejected'].sum():6d}/{len(arch_historical[a]):7d} "
        f"({arch_historical[a]['rejected'].mean()*100:5.2f}%)"
    )

# %%
# Memorised records by architecture: future (left) vs. historical (right) records.
# Bars = absolute number of records with a significant change under memorisation
# partitioning, black tick = random-partitioning baseline (sanity check).
# Log y-axis, shared across both panels: the counts span ~5 orders of magnitude, so on a
# linear axis everything but the random forest collapses onto zero.
# NOTE: the two panels have different denominators (N in the titles), so counts are not
# directly comparable across panels -- the historical panel carries the ~4.9x larger
# dataset on top of any difference in memorisation. Rates are in the printout of the
# cell above.
count_panels = [
    ("Future records", arch_future, arch_future_random),
    ("Historical records", arch_historical, arch_historical_random),
]
floor = 0.5  # half a record: the smallest count either panel can resolve is a single record
max_count = max(dfs[a]["rejected"].sum() for _, dfs, _ in count_panels for a in ARCH_ORDER)
top = 10.0 ** np.ceil(np.log10(2 * max_count))  # next decade, with headroom for the labels
fig, axs = plt.subplots(1, 2, figsize=(2.0, 1.9), sharey=True)
drew_baseline = False
for ax, (title, dfs, rand_dfs) in zip(axs, count_panels):
    n_records = {len(dfs[a]) for a in ARCH_ORDER}
    assert len(n_records) == 1, f"architectures cover different record counts: {n_records}"
    x = np.arange(len(ARCH_ORDER))
    counts = [dfs[a]["rejected"].sum() for a in ARCH_ORDER]
    ax.bar(x, counts, color=[ARCH_COLOR[a] for a in ARCH_ORDER], width=0.7)
    ax.set_yscale("log")
    ax.set_ylim(floor, top)
    # random-partitioning baseline, where non-zero (zero counts cannot be shown on a log axis
    # and are printed below instead)
    for xi, a in zip(x, ARCH_ORDER):
        rand_count = rand_dfs[a]["rejected"].sum()
        if rand_count > 0:
            ax.hlines(
                rand_count, xi - 0.35, xi + 0.35,
                color="black", linestyle="--", lw=0.8, zorder=3,
            )
            drew_baseline = True
    # n rejected per bar, inside tall bars and above short ones. Architectures that memorise
    # nothing get an explicit n=0 at the axis floor, since a zero bar cannot be drawn on a log
    # axis and would otherwise be indistinguishable from missing data.
    span = np.log10(top) - np.log10(floor)
    for xi, a, count in zip(x, ARCH_ORDER, counts):
        frac = 0.0 if count <= 0 else (np.log10(count) - np.log10(floor)) / span
        inside = frac > 0.5
        ax.text(
            xi,
            count * 0.85 if inside else max(count, floor) * 1.3,
            f"n={count:,}",
            ha="center",
            va="top" if inside else "bottom",
            color="white" if inside else "black",
            fontsize=FONT_SIZE, rotation=90,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_title(f"{title}\n(N = {n_records.pop():,})", fontsize=FONT_SIZE)
    ax.spines[["right", "top"]].set_visible(False)
axs[0].set_ylabel("Records with significant change\n in prediction (count)")
handles = [mpatches.Patch(color=ARCH_COLOR[a], label=ARCH_STR[a]) for a in ARCH_ORDER]
if drew_baseline:  # only advertise the baseline in the legend if any was actually drawn
    handles.append(Line2D([0], [0], color="black", linestyle="--", label="Random partitioning"))
fig.legend(handles=handles, fontsize=FONT_SIZE, loc="outside lower center", ncol=3, frameon=False)
fig.savefig(fig_dir / "arch_comparison_mem_counts.pdf", dpi=300, bbox_inches="tight")
plt.show()

# random-partitioning baselines, for the figure caption (all zero => not drawable above)
for panel, (title, _, rand_dfs) in zip(["future", "historical"], count_panels):
    for a in ARCH_ORDER:
        r = rand_dfs[a]["rejected"]
        print(f"random partitioning, {panel:11s} {ARCH_STR[a]:20s}: {r.sum():6d}/{len(r):7d} ({r.mean()*100:.4f}%)")

# %%
# Empirical survival functions by architecture (single row of four panels).
# Left pair = effect size (change in predicted probability), right pair = energy test statistic;
# within each pair: future data / historical (training) data.
# Each pair sits in its own subfigure so it can carry one shared x-axis label.
# Solid = memorisation partitioning, dashed = random-partitioning baseline (sanity check).
esf_metrics = [
    ("effect_size", "Change in predicted probability (%)", 100.0, (0, None)),
    ("test_statistic", "Energy test statistic", 1.0, (0, None)),
]
esf_sets = [
    ("Future records", arch_future, arch_future_random),
    ("Historical records", arch_historical, arch_historical_random),
]
fig = plt.figure(figsize=(4.6, 2.1), layout="constrained")
subfigs = fig.subfigures(1, len(esf_metrics), wspace=0.06)
axs, esf_pairs = [], []
for sf, (col_name, xlabel, scale, xlim) in zip(subfigs, esf_metrics):
    pair = sf.subplots(1, len(esf_sets), sharey=True)
    for ax, (title, dfs, rand_dfs) in zip(pair, esf_sets):
        for a in ARCH_ORDER:
            esf_plot(
                data_arr=dfs[a][col_name].values * scale,
                xlabel=xlabel,
                draw_conf=False,
                color=ARCH_COLOR[a],
                label=ARCH_STR[a],
                ax=ax,
            )
            # random-partitioning baseline (dashed)
            esf_plot(
                data_arr=rand_dfs[a][col_name].values * scale,
                xlabel=xlabel,
                draw_conf=False,
                color=ARCH_COLOR[a],
                line_style="--",
                alpha=0.75,
                ax=ax,
            )
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("")  # replaced by the shared per-pair label below
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_title(title, fontsize=FONT_SIZE)
    sf.supxlabel(xlabel, fontsize=FONT_SIZE)
    axs.extend(pair)
    esf_pairs.append(pair)

# joint y-axis floors at 1/N over all shown records (set on every panel: sharey only
# links the two axes inside a subfigure, not across the two subfigures)
n_max = max(len(df) for dfs in (arch_future, arch_historical) for df in dfs.values())
for ax in axs:
    ax.set_ylim(0.5 / n_max, 1)
for ax in axs[1:]:
    ax.tick_params(labelleft=False)
axs[0].set_ylabel("1 - cumulative probability", fontsize=FONT_SIZE)
handles = [Line2D([0], [0], color=ARCH_COLOR[a], label=ARCH_STR[a]) for a in ARCH_ORDER]
handles.append(Line2D([0], [0], color="black", linestyle="--", label="Random partitioning"))
fig.legend(
    handles=handles,
    fontsize=FONT_SIZE,
    loc="outside lower center",
    ncol=len(ARCH_ORDER) + 1,
    frameon=False,
)

# supxlabel centres on its subfigure, but the left subfigure also holds the y-axis label
# and tick labels, so its axes sit right of centre. Lay out once, then re-place each
# label on the measured centre of its own pair of axes and freeze the layout.
fig.draw_without_rendering()
for sf, pair, (_, xlabel, _, _) in zip(subfigs, esf_pairs, esf_metrics):
    x0, x1 = pair[0].get_position().x0, pair[-1].get_position().x1
    sf.supxlabel(xlabel, fontsize=FONT_SIZE, x=0.53 * (x0 + x1))
fig.set_layout_engine("none")

fig.savefig(fig_dir / "arch_comparison_esf.pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Training- and test-set diagnostic performance by architecture (MIMIC-IV-ED).
# The held-out test set is drawn from patients disjoint from the historical/future datasets,
# so every random-subset model is an OUT model w.r.t. it: the spread below is the random
# variation induced purely by training each model on a different random subset.
# Companion to the memorisation comparison above -- it separates how much an architecture
# memorises from how well it actually performs. The two panels are scaled independently,
# so the generalisation gap is not readable across them; it is printed out below instead.
ARCH_LOGDIRS = {
    "lr": LOG_ROOT / "mimic-iv-ed-lr",
    "rf": LOG_ROOT / "mimic-iv-ed-rf",
    "tabresnet": LOG_ROOT / "mimic-iv-ed-tabresnet",
}
# the keras/jax runs prefix the metric with an underscore, the sklearn ones do not
TEST_AUROC_KEYS = ("_test_macro_auroc", "_test_macro_auc", "test_macro_auroc", "test_macro_auc")
TRAIN_AUROC_KEYS = ("_macro_auroc", "_macro_auc", "train_macro_auroc", "train_macro_auc")


def collect_macro_auroc(root: Path, section: str, keys) -> np.ndarray:
    """Collect per-model macro AUROC (%) from each run's info.json."""
    scores = []
    for run_dir in sorted(root.iterdir()):
        info_path = run_dir / "info.json"
        if not run_dir.is_dir() or not info_path.exists():
            continue
        with open(info_path) as f:
            metrics = json.load(f).get(section, {})
        auroc = next((metrics[k] for k in keys if k in metrics), None)
        if auroc is not None:
            scores.append(float(auroc))
    return np.array(scores) * 100


arch_test_auroc = {a: collect_macro_auroc(ARCH_LOGDIRS[a], "test_metrics", TEST_AUROC_KEYS) for a in ARCH_ORDER}
arch_train_auroc = {a: collect_macro_auroc(ARCH_LOGDIRS[a], "train_metrics", TRAIN_AUROC_KEYS) for a in ARCH_ORDER}

np.random.seed(0)
# separate panels with independent y-axes: the random forest's training score sits ~12 points
# above its test score, so a shared axis would flatten the differences within each panel
perf_panels = [("Training set", arch_train_auroc), ("Test set", arch_test_auroc)]
fig, axs = plt.subplots(1, 2, figsize=(2.5, 1.8))
for ax, (title, data) in zip(axs, perf_panels):
    for i, a in enumerate(ARCH_ORDER):
        vals = data[a]
        jitter = np.random.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color=ARCH_COLOR[a], s=2, alpha=0.35, zorder=1, rasterized=True,
        )
        m, s = vals.mean(), vals.std()
        ax.errorbar(
            i, m, yerr=s, fmt="o", color="white", markeredgecolor="black",
            markeredgewidth=0.7, markersize=4, ecolor="black", capsize=3,
            elinewidth=0.7, zorder=3,
        )
        # label above the error bar; a label beside the marker would collide with its
        # neighbour, and on one line "85.07+-0.08" is wider than the space per group,
        # so mean and std are stacked to halve the label width
        ax.annotate(
            f"{m:.2f}\n$\\pm${s:.2f}",
            xy=(i, m + 2*s), xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=FONT_SIZE, linespacing=0.95,
        )
    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels([ARCH_STR[a] for a in ARCH_ORDER])#, rotation=90, ha="center")
    ax.set_xlim(-0.6, len(ARCH_ORDER) - 0.4)  # keep the edge labels off the spines
    ax.margins(y=0.16)  # headroom for the labels above the error bars
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Diagnostic performance (%)")

# margins(y=...) reserves headroom in data units, which cannot know how tall the labels
# render, so lay out once and then grow each panel's top limit until its tallest label
# fits inside the axes (otherwise the topmost one runs into the title)
fig.draw_without_rendering()
for ax in axs:
    inv = ax.transData.inverted()
    label_top = max(t.get_window_extent().transformed(inv).y1 for t in ax.texts)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, max(y1, label_top + 0.04 * (y1 - y0)))

fig.savefig(fig_dir / "arch_comparison_test_auroc.pdf", dpi=300, bbox_inches="tight")
plt.show()

# train/test gap alongside, since it is what the memorisation differences track
for a in ARCH_ORDER:
    te, tr = arch_test_auroc[a], arch_train_auroc[a]
    print(
        f"{ARCH_STR[a]:20s} n={len(te):3d}  test {te.mean():6.3f}+-{te.std():.3f}  "
        f"train {tr.mean():6.3f}+-{tr.std():.3f}  gap {tr.mean() - te.mean():+.3f}"
    )
