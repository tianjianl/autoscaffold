"""Autoscaffold progress plot, inspired by karpathy/autoresearch/analysis.ipynb"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.tsv", sep="\t")

# Parse accuracy "19/33" -> fraction
def parse_acc(s):
    num, den = str(s).split("/")
    return int(num) / int(den)

def parse_correct(s):
    return int(str(s).split("/")[0])

df["acc"] = df["accuracy"].apply(parse_acc)
df["correct"] = df["accuracy"].apply(parse_correct)

# Filter out crashes (none here, but for safety)
valid = df[df["status"] != "crash"].copy().reset_index(drop=True)

baseline_acc = valid.loc[0, "acc"]
best_acc = valid.loc[valid["status"] == "keep", "acc"].max()

# --------------- Main progress plot ---------------
fig, ax = plt.subplots(figsize=(16, 8))

# Discarded: faint gray dots
disc = valid[valid["status"] == "discard"]
ax.scatter(disc.index, disc["acc"], c="#cccccc", s=40, alpha=0.5,
           zorder=2, label="Discarded")

# Kept: prominent green dots
kept = valid[valid["status"] == "keep"]
ax.scatter(kept.index, kept["acc"], c="#2ecc71", s=80, zorder=4,
           label="Kept", edgecolors="black", linewidths=0.5)

# Running maximum step line (higher is better)
kept_mask = valid["status"] == "keep"
kept_idx = valid.index[kept_mask]
kept_acc = valid.loc[kept_mask, "acc"]
running_max = kept_acc.cummax()
ax.step(kept_idx, running_max, where="post", color="#27ae60",
        linewidth=2, alpha=0.7, zorder=3, label="Running best")

# Label each kept experiment, stagger overlapping labels
label_offsets = {}  # track y positions to avoid overlap
for idx, acc_val in zip(kept_idx, kept_acc):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 50:
        desc = desc[:47] + "..."

    # Stagger labels at same accuracy level
    key = f"{acc_val:.3f}"
    count = label_offsets.get(key, 0)
    label_offsets[key] = count + 1
    y_offset = 6 + count * 14  # stack labels vertically

    ax.annotate(desc, (idx, acc_val),
                textcoords="offset points", xytext=(6, y_offset),
                fontsize=7.5, color="#1a7a3a", alpha=0.9,
                rotation=25, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#1a7a3a", alpha=0.3, lw=0.5)
                if count > 0 else None)

n_total = len(df)
n_kept = len(df[df["status"] == "keep"])
n_discard = len(df[df["status"] == "discard"])

ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Accuracy on HMMT Feb 2026 (higher is better)", fontsize=12)
ax.set_title(
    f"Autoscaffold Progress: {n_total} Experiments, {n_kept} Kept Improvements "
    f"({parse_correct(df.loc[0, 'accuracy'])}/33 \u2192 {int(best_acc*33)}/33)",
    fontsize=14,
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)

# Y-axis: zoom into interesting range
margin = (best_acc - baseline_acc) * 0.15
ax.set_ylim(baseline_acc - margin, best_acc + margin)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

# Add secondary y-axis showing X/33
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_ylabel("Correct / 33", fontsize=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*33:.0f}/33"))

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print(f"Saved progress.png")

# --------------- Summary stats ---------------
print(f"\n{'='*50}")
print(f"Total experiments:  {n_total}")
print(f"  Kept:             {n_kept}")
print(f"  Discarded:        {n_discard}")
print(f"  Keep rate:        {n_kept/(n_kept+n_discard):.0%}")
print(f"Baseline accuracy:  {baseline_acc:.1%} ({int(baseline_acc*33)}/33)")
print(f"Best accuracy:      {best_acc:.1%} ({int(best_acc*33)}/33)")
print(f"Improvement:        +{best_acc - baseline_acc:.1%} (+{int(best_acc*33) - int(baseline_acc*33)} problems)")
print(f"{'='*50}")

# Top kept experiments by marginal improvement
print(f"\nKept experiments (in order):")
prev_best = 0
for _, row in valid[valid["status"] == "keep"].iterrows():
    delta = row["acc"] - prev_best if row["acc"] > prev_best else 0
    marker = " *NEW BEST*" if delta > 0 else ""
    prev_best = max(prev_best, row["acc"])
    tokens_k = row["tokens_used"] / 1000
    print(f"  {row['accuracy']:>5s}  {tokens_k:6.0f}K tokens  {row['description']}{marker}")
