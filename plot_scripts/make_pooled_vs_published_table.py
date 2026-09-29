import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"

chems = ["Li-ion", "CALB", "Zn-ion", "Na-ion", "Macro avg"]
# (mean, std or None)
mape = {
    "pub_mlp": [(0.179, .003), (0.140, .009), (0.558, .034), (0.274, .026), (0.288, None)],
    "pub_tf":  [(0.184, .003), (0.149, .005), (0.515, .067), (0.255, .036), (0.276, None)],
    "pooled":  [(0.203, .006), (0.117, .039), (0.512, .025), (0.271, .054), (0.276, None)],
    "cand":    [(0.234, .013), (0.136, .032), (0.402, .051), (0.337, .085), (0.278, None)],
    "reg":     [(0.177, .005), (0.183, .055), (0.533, .051), (0.359, .065), (0.313, None)],
    "a1":      [(0.192, .007), (0.129, .068), (0.465, .067), (0.330, .097), (0.279, None)],
}
acc = {
    "pub_mlp": [(62.0, 0.4), (70.4, 5.3), (29.7, 8.4), (33.7, 3.8), (49.0, None)],
    "pub_tf":  [(57.3, 1.6), (67.2, 10.7), (20.2, 8.4), (40.6, 8.4), (46.3, None)],
    "pooled":  [(57.1, 0.6), (84.4, 10.7), (19.3, 8.9), (34.7, 7.4), (48.9, None)],
    "cand":    [(56.2, 2.3), (83.0, 3.7), (28.5, 13.8), (29.7, 13.8), (49.4, None)],
    "reg":     [(61.4, 1.1), (58.5, 16.0), (22.4, 9.4), (24.6, 8.7), (41.7, None)],
    "a1":      [(60.6, 0.4), (79.9, 26.4), (30.6, 11.5), (24.7, 14.1), (49.0, None)],
}

INK, INK2, MUTED = "#1f2328", "#57606a", "#8c959f"
GRID = "#d8dee4"
WIN, SPLIT, LOSE = "#dcf2e3", "#fbf0d4", "#fbe1df"
WIN_T, SPLIT_T, LOSE_T = "#1a7f37", "#9a6700", "#cf222e"

POOL = "one model, all 4 chemistries"
heads = [
    ("Published CPMLP",        ["BatteryLife paper", "separate model per chemistry", "regression", "reported mean ± std"]),
    ("Published CPTransformer", ["BatteryLife paper", "separate model per chemistry", "regression", "reported mean ± std"]),
    ("Ours: pooled CPMLP",     [POOL, "regression · wd 0 · dropout 0", "no chemistry conditioning", "3 seeds"]),
    ("Ours: pooled CPMLP",     [POOL, "geo_bins · wd 1e-3 · dropout 0", "no chemistry conditioning", "3 seeds"]),
    ("Ours: pooled CPMLP",     [POOL, "geo_bins · wd 1e-3 · dropout 0", "+ chem. input embedding (E=16)", "3 seeds"]),
    ("Ours: pooled CPMLP",     [POOL, "geo_bins · wd 1e-3 · dropout 0", "+ loss weighting α = 1.0", "3 seeds"]),
]
keys = ["pub_mlp", "pub_tf", "reg", "pooled", "cand", "a1"]

fig = plt.figure(figsize=(16.8, 10.25), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 16.8); ax.set_ylim(1.15, 11.4)

ax.text(0.4, 11.0, "Pooled CPMLP vs. published BatteryLife baselines",
        fontsize=17, weight="bold", color=INK, va="center")
ax.text(0.4, 10.62, "Our models: one CPMLP trained on Li-ion + CALB + Zn-ion + Na-ion together, evaluated per chemistry on the paper's test cells. Single checkpoint per seed.",
        fontsize=10.5, color=INK2, va="center")

x_chem = 0.4
col_x = [2.9, 5.15, 7.4, 9.75, 12.1, 14.45]
x_share = 1.55
# share of pooled training samples (approx.): Li 50.3k, CALB 1.69k, Zn 5.95k, Na 2.0k; train cells 515/17/60/20
_tr = [50300, 1690, 5950, 2000]
share = [f"{100*n/sum(_tr):.1f}%" for n in _tr] + [""]
share_cells = ["515 cells", "17 cells", "60 cells", "20 cells", ""]
col_w = 2.1

def fmt(v, s, is_mape):
    if is_mape:
        m = f"{v:.3f}"; sd = f" ± {s:.3f}" if s is not None else ""
    else:
        m = f"{v:.1f}%"; sd = f" ± {s:.1f}" if s is not None else ""
    return m, sd

def panel(y_top, title, data, is_mape):
    better = (lambda a, b: a < b - 1e-9) if is_mape else (lambda a, b: a > b + 1e-9)
    ax.text(x_chem, y_top, title, fontsize=12.5, weight="bold", color=INK, va="center")
    htop = y_top - 0.38
    for x, (t, lines) in zip(col_x, heads):
        ax.text(x + col_w / 2, htop, t, fontsize=9.5, color=INK, ha="center", va="top", weight="bold")
        ax.text(x + col_w / 2, htop - 0.2, "\n".join(lines), fontsize=8.3, color=INK2, ha="center", va="top", linespacing=1.35)
    ax.text(x_share + 0.55, htop, "Share of\ntrain data", fontsize=9.5, color=INK, ha="center", va="top", weight="bold")
    hy = y_top - 0.9
    ax.plot([x_chem, 16.6], [hy - 0.33, hy - 0.33], color=INK2, lw=1)
    rh = 0.42
    for i, ch in enumerate(chems):
        y = hy - 0.62 - i * rh
        macro = ch.startswith("Macro")
        if macro:
            ax.plot([x_chem, 16.6], [y + rh / 2, y + rh / 2], color=GRID, lw=1)
        ax.text(x_chem, y, ch, fontsize=10.5, color=INK, va="center", weight="bold" if macro else "normal")
        if share[i]:
            t = ax.text(x_share + 0.2, y, share[i], fontsize=10.5, color=INK, va="center")
            ax.annotate("  " + share_cells[i], xycoords=t, xy=(1, 0.5), va="center", fontsize=8.5, color=MUTED)
        vals = [data[k][i][0] for k in keys]
        best = min(vals) if is_mape else max(vals)
        for j, (x, k) in enumerate(zip(col_x, keys)):
            v, s = data[k][i]
            if k in ("reg", "pooled", "cand", "a1"):
                w = sum(better(v, data[p][i][0]) for p in ("pub_mlp", "pub_tf"))
                bg, tc, mark = [(LOSE, LOSE_T, "✗ beats neither"), (SPLIT, SPLIT_T, "◐ beats one"), (WIN, WIN_T, "✓ beats both")][w]
                ax.add_patch(FancyBboxPatch((x + 0.06, y - rh / 2 + 0.04), col_w - 0.12, rh - 0.08,
                             boxstyle="round,pad=0,rounding_size=0.06", fc=bg, ec="none"))
                ax.text(x + col_w - 0.14, y, mark.split(" ")[0], fontsize=10, color=tc, ha="right", va="center", weight="bold")
            m, sd = fmt(v, s, is_mape)
            isbest = abs(v - best) < 1e-9
            t = ax.text(x + 0.2, y, m, fontsize=10.5, color=INK, va="center", weight="bold" if isbest else "normal")
            if sd:
                ax.annotate(sd, xycoords=t, xy=(1, 0.5), va="center", fontsize=8.5, color=MUTED)
    return hy - 0.62 - len(chems) * rh

y_end = panel(10.1, "Test MAPE  (lower is better)", mape, True)
panel(y_end - 0.25, "Test 15%-accuracy  (higher is better)", acc, False)

# legend + notes
ly = 2.5
items = [(WIN, WIN_T, "✓", "beats both published"), (SPLIT, SPLIT_T, "◐", "beats one"), (LOSE, LOSE_T, "✗", "beats neither")]
lx = 0.4
for bg, tc, sym, lab in items:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.13), 0.34, 0.26, boxstyle="round,pad=0,rounding_size=0.05", fc=bg, ec="none"))
    ax.text(lx + 0.17, ly, sym, fontsize=10, color=tc, ha="center", va="center", weight="bold")
    ax.text(lx + 0.45, ly, lab, fontsize=9.5, color=INK2, va="center")
    lx += 2.35
ax.text(lx + 0.1, ly, "Bold = best in row", fontsize=9.5, color=INK2, va="center", weight="bold")
ax.text(0.4, 1.8,
        "± = std across seeds (published: as reported in the paper; ours: seeds 42/2021/2024; seed also sets the CALB/Zn-ion/Na-ion split). geo_bins = classification over ~±15% geometric bins.\n"
        "Share of train data = fraction of pooled training samples (~59.9k total); cell counts are training cells per chemistry.\n"
        "Chem. input embedding = 16-dim learned chemistry embedding concatenated to the input (early_concat). Loss weighting α: each sample weighted by N_chem^-α (α = 1: every chemistry counts equally).\n"
        "CALB and Na-ion have only 5 test cells each; each cell contributes ~100 samples (prefix lengths 1-100 cycles).",
        fontsize=8.5, color=MUTED, va="center", linespacing=1.5)

fig.savefig("figures/pooled_cpmlp_vs_published.png", dpi=200, facecolor="white")
