"""Render the manuscript exhibits from their public CSV inputs."""

import csv
from html import escape
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASE, FULL, HOTEL, SEARCH = "#71655f", "#21818c", "#24577b", "#b65e37"
METHOD_COLORS = ["#66844d", "#6591b5", "#c49a49", "#a381a9", FULL]


def read_csv(filename):
    with (RESULTS / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def table_html(rows):
    """Keep numeric precision and show each panel with its own columns."""
    panels = dict.fromkeys(row.get("Panel") for row in rows)
    parts = []
    for panel in panels:
        subset = [row for row in rows if row.get("Panel") == panel]
        columns = [
            name for name in subset[0]
            if name != "Panel" and any(row[name] for row in subset)
        ]
        caption = f"<caption>{escape(panel)}</caption>" if panel else ""
        head = "".join(f'<th scope="col">{escape(col)}</th>' for col in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(row[col])}</td>" for col in columns) + "</tr>"
            for row in subset
        )
        parts.append(f"<table>{caption}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
    return "".join(parts)


def groups(rows, key):
    by_value = {}
    for row in rows:
        by_value.setdefault(row[key], []).append(row)
    return by_value


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def setting_color(setting):
    return HOTEL if setting.startswith("Expedia") else SEARCH


def compact(setting):
    return setting.replace("Expedia ", "E. ").replace("KuaiSearch Overall", "Overall")


def capture(rows):
    fig, ax = plt.subplots(figsize=(8.5, 4.1), layout="constrained")
    shares = values(rows, "capture_pct")
    names = [row["setting"] for row in rows]
    bars = ax.barh(names, shares, color=[setting_color(name) for name in names])
    ax.bar_label(bars, labels=[f"{share:.1f}%" for share in shares], padding=4)
    ax.invert_yaxis()
    ax.set(xlim=(0, 103), xlabel="Stronger single-interface gain / Full S/R gain (%)")
    ax.axvline(100, color="#677780", linestyle="--", linewidth=1)
    ax.grid(axis="x")
    return fig


def ranges(rows):
    fig, ax = plt.subplots(figsize=(8.8, 4.3), layout="constrained")
    for y, row in enumerate(rows):
        low, high, ref = (float(row[key]) for key in (
            "min_uplift_pct", "max_uplift_pct", "reference_uplift_pct"
        ))
        color = setting_color(row["setting"])
        ax.hlines(y, low, high, linewidth=3, color=color, alpha=.55)
        ax.plot([low, high], [y, y], linestyle="none", marker="|", markersize=10, color=color)
        ax.scatter(ref, y, color=color, edgecolor="white", zorder=3)
    ax.set(yticks=range(len(rows)), yticklabels=[r["setting"] for r in rows],
           xlabel="Full − Base uplift (%)")
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_title("Range across tested capacities; dots mark reference capacity", loc="left")
    return fig


def deployment_map(rows):
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    ref = values(rows, "reference_uplift_pct")
    span = values(rows, "range_pp")
    offsets = {
        "Expedia30": (7, 8), "Expedia Head": (-8, 9), "Expedia Regular": (7, 8),
        "Expedia Cold": (7, 8), "KuaiSearch Overall": (7, 8), "Q-Head": (9, 7),
        "Q-Regular": (7, 7), "Q-Cold": (7, -7), "Q-Unseen": (9, -8),
    }
    for domain, color in (("Expedia", HOTEL), ("KuaiSearch", SEARCH)):
        indices = [i for i, r in enumerate(rows) if setting_color(r["setting"]) == color]
        ax.scatter(ref[indices], span[indices], s=67, color=color, edgecolor="white",
                   linewidth=.8, zorder=3, label=domain)
    for x, y, row in zip(ref, span, rows):
        dx, dy = offsets[row["setting"]]
        name = compact(row["setting"])
        ax.annotate(f"{name}  ·  min {row['min_uplift_pct']}%", (x, y), (dx, dy),
                    textcoords="offset points", ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top", fontsize=8.5)
    ax.axvline(median(ref), color="#8a979e", linestyle="--")
    ax.axhline(median(span), color="#8a979e", linestyle="--")
    ax.set(xlim=(-.8, 19.4), ylim=(-.2, 20.3), xlabel="Reference-capacity uplift (%)",
           ylabel="Resource sensitivity (uplift range, percentage points)")
    ax.grid()
    ax.legend(frameon=False, loc="upper right")
    return fig


def interval_series(rows, left, right, key, right_key=None):
    x = [float(row[k]) for row in rows for k in (left, right)]
    y = [float(row[k]) for row in rows for k in (key, right_key or key)]
    return x, y


def adoption(rows):
    fig, (stock, value, boundary) = plt.subplots(
        1, 3, figsize=(11.7, 3.6), layout="constrained"
    )
    for policy, color, dash, label in (("base", BASE, "--", "Base"), ("full", FULL, "-", "Full S/R")):
        stock.plot(*interval_series(rows, "charge_left", "charge_right", f"{policy}_q"),
                   color=color, linestyle=dash, label=label)
        value.plot(*interval_series(rows, "charge_left", "charge_right", f"{policy}_net_left",
                                   f"{policy}_net_right"), color=color, linestyle=dash, label=label)
    x, gain = interval_series(rows, "charge_left", "charge_right",
                              "break_even_sr_charge_left", "break_even_sr_charge_right")
    boundary.plot(x, gain, color=FULL)
    boundary.fill_between(x, 0, gain, color=FULL, alpha=.19, label="Full S/R preferred")
    stock.set(title="(a.1) Optimal capacity", ylabel="Initial units per hotel", yticks=[1, 2, 3, 4])
    value.set(title="(a.2) Net daily value", ylabel="Revenue less inventory cost / day")
    boundary.set(title="(b) Adoption boundary", ylabel="Incremental S/R cost / day", ylim=(0, 1100))
    for ax in (stock, value, boundary):
        ax.set(xlabel="Cost per initial unit / day", xlim=(min(x), max(x)))
        ax.grid()
    stock.legend(frameon=False)
    boundary.legend(frameon=False, loc="upper right")
    return fig


def thresholds(rows):
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.4), layout="constrained")
    for ax, (setting, batch) in zip(axes.flat, groups(rows, "setting").items()):
        for key, label, color, dash in (
            ("base_capacity", "Base", BASE, "--"), ("full_capacity", "Full S/R", FULL, "-")
        ):
            ax.plot(*interval_series(batch, "target_left_pct", "target_right_pct", key),
                    label=label, color=color, linestyle=dash)
        for row in batch:
            lower, upper = (float(row[key]) for key in ("full_capacity", "base_capacity"))
            if lower < upper:
                ax.fill_between([float(row["target_left_pct"]), float(row["target_right_pct"])],
                                lower, upper, color=FULL, alpha=.16)
        ylabel = "Units per hotel" if batch[0]["capacity_type"] == "inventory_per_hotel" else "Exposure multiplier"
        ax.set(title=setting, ylabel=ylabel, xlim=(float(batch[0]["target_left_pct"]), 100))
        ax.set_yticks([1, 2, 3, 4] if ylabel == "Units per hotel" else [.5, .75, 1, 1.25, 1.5])
        ax.grid()
    fig.supxlabel("Revenue target (% of maximum Base revenue on the tested grid)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return fig


def gains(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained", sharey=True)
    methods = list(groups(rows, "method"))
    first, last = list(groups(rows, "setting"))[:5], list(groups(rows, "setting"))[5:]
    width = .76 / len(methods)
    for ax, settings, title in zip(axes, (first, last), ("Hotel and Overall", "Query-support tiers")):
        for j, method in enumerate(methods):
            data = {r["setting"]: float(r["gain_pct"]) for r in rows if r["method"] == method}
            bars = [data[s] for s in settings]
            colors = ([FULL if item >= 0 else "#b65e54" for item in bars]
                      if len(methods) == 1 else METHOD_COLORS[j])
            ax.bar(np.arange(len(settings)) + (j - (len(methods) - 1) / 2) * width,
                   bars, width, label=method, color=colors)
        ax.set(xticks=range(len(settings)), xticklabels=[compact(s) for s in settings], title=title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.axhline(0, color="#555555", linewidth=.7)
        ax.grid(axis="y")
    axes[0].set_ylabel("Gain over comparator (%)")
    if len(methods) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    return fig


def markets(rows):
    fig, (level, gain) = plt.subplots(1, 2, figsize=(8.8, 3.6), layout="constrained")
    settings = list(groups(rows, "setting"))
    x = np.arange(len(settings))
    refs = list(groups(rows, "reference").values())
    level.plot(x, values(refs[0], "mean"), marker="s", color=FULL, label="Qwen S/R")
    for subset, color in zip(refs, (BASE, "#b18d47")):
        label = subset[0]["reference"]
        level.plot(x, values(subset, "reference_mean"), marker="o", label=label, color=color)
        gain.plot(x, values(subset, "gain_pct"), marker="o", label=f"Over {label}", color=color)
    level.set(title="Revenue levels", ylabel="Revenue per search")
    gain.set(title="Qwen S/R uplift", ylabel="Relative gain (%)")
    for ax in (level, gain):
        ax.set(xticks=x, xticklabels=[s.removeprefix("Expedia ") for s in settings])
        ax.grid()
        ax.legend(frameon=False)
    return fig


def heatmap(rows):
    settings = list(groups(rows, "setting"))
    methods = list(groups(rows, "method"))
    lookup = {(row["setting"], row["method"]): float(row["gain_pct"]) for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.65), layout="constrained",
                             gridspec_kw={"width_ratios": [5, 4]}, sharey=True)
    for ax, subset, title in zip(axes, (settings[:5], settings[5:]),
                                 ("Hotel and Overall", "Query-support tiers")):
        mat = np.array([[lookup[s, method] for s in subset] for method in methods])
        image = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(lookup.values()))
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if val > 14 else "#172f41")
        ax.set(xticks=range(len(subset)), xticklabels=[compact(s) for s in subset],
               yticks=range(len(methods)), yticklabels=[m.removesuffix(" + S/R") for m in methods],
               title=title)
        ax.tick_params(axis="x", labelrotation=25)
    fig.colorbar(image, ax=axes, label="Gain over A2C Transformer (%)", shrink=.8)
    return fig


def capacity(rows):
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.6), layout="constrained")
    for ax, (setting, batch) in zip(axes.flat, groups(rows, "setting").items()):
        x = values(batch, "capacity")
        for key, label, color, marker, style in (
            ("base", "Base", BASE, "o", "--"), ("full", "Full S/R", FULL, "s", "-")
        ):
            ax.errorbar(x, values(batch, key + "_mean"), values(batch, key + "_sd"),
                        label=label, color=color, marker=marker, linestyle=style, capsize=2)
        inventory = batch[0]["capacity_type"] == "inventory_per_hotel"
        ylabel = "Daily revenue" if setting == "Expedia30" else (
            "Revenue per search" if inventory else "GMV / 1,000 requests"
        )
        ax.set(title=setting, xticks=x, ylabel=ylabel,
               xlabel="Initial inventory per hotel" if inventory else "Exposure multiplier")
        ax.grid()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return fig


DRAW = {
    "capture": capture, "ranges": ranges, "map": deployment_map, "adoption": adoption,
    "thresholds": thresholds, "gains": gains, "markets": markets, "heatmap": heatmap,
    "capacity": capacity,
}


def render_figure(chart, rows):
    with plt.rc_context({
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "grid.color": "#dde4e9", "grid.linewidth": .65, "font.size": 10,
        "legend.fontsize": 9, "figure.dpi": 115,
    }):
        return DRAW[chart](rows)
