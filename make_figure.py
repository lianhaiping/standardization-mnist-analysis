import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RED, BLUE, INK, MUT = '#e34948', '#2a78d6', '#1a1a19', '#6b6a66'
rows = [
    ("Logistic regression", "lbfgs",     "0.9165 ± 0.0002", "0.9037 ± 0.0008", -1.28, 0.06, 0),
    ("MLP (128)",           "adam",      "0.9737 ± 0.0004", "0.9688 ± 0.0002", -0.49, 0.04, 0),
    ("SGDClassifier",       "plain SGD", "0.9021 ± 0.0020", "0.9100 ± 0.0006", +0.79, 0.22, 1),
    ("MLP (128)",           "plain SGD", "0.9256 ± 0.0005", "0.9483 ± 0.0002", +2.27, 0.05, 1),
    ("MLP (128)",           "adam",      "0.9170 ± 0.0011", "0.9690 ± 0.0001", +5.20, 0.33, 1),
    ("Logistic regression", "lbfgs",     "0.8869 ± 0.0028", "0.9035 ± 0.0008", +1.67, 0.15, 1),
]
SPLIT = 4          # rows before this index share units
labels = [f"{r[0]}\n{r[1]}" for r in rows]
labels[SPLIT]   = "MLP (128)  adam\n100 cols x1000"
labels[SPLIT+1] = "Logistic regression  lbfgs\n100 cols x1000"
rows_tbl = rows
vals   = [r[4] for r in rows]
errs   = [r[5] for r in rows]
cols   = [BLUE if r[6] else RED for r in rows]

fig = plt.figure(figsize=(10.8, 13.6), dpi=120, facecolor='white')
gs = GridSpec(3, 1, height_ratios=[1.45, 5.0, 5.0], hspace=0.30,
              left=0.085, right=0.955, top=0.965, bottom=0.055)

# ---- title block
ax0 = fig.add_subplot(gs[0]); ax0.axis('off')
ax0.text(0, .96, "Always Standardize Your ML Features?", fontsize=26, color=INK,
         va='top', weight='medium')
for i, line in enumerate([
        "Preprocessing, model choice, and optimization algorithm aren't three",
        "independent decisions you make in sequence. They're one system,",
        "and they constrain each other."]):
    ax0.text(0, .52 - i*.245, line, fontsize=15, color=MUT, va='top')

# ---- chart
ax = fig.add_subplot(gs[1])
y = range(len(rows))
ax.barh(y, vals, color=cols, height=.55, zorder=3)
ax.errorbar(vals, y, xerr=errs, fmt='none', ecolor='#44443f',
            elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
for i, v in enumerate(vals):
    off = 0.30 if v > 0 else -0.30
    ax.text(v + off + (errs[i] if v > 0 else -errs[i]), i,
            f"{v:+.2f}", va='center', ha='left' if v > 0 else 'right',
            fontsize=13, color=INK, weight='medium', zorder=5)
ax.axvline(0, color='#b8b7b0', lw=1.1, zorder=2)
ax.axhline(SPLIT-0.5, color='#c9c8c1', lw=1.0, ls=(0,(4,3)), zorder=2)
ax.text(-2.5, -0.60, "shared units  [0,1]", ha='left', fontsize=11,
        color=MUT, style='italic')
ax.text(-2.5, SPLIT-0.40, "mismatched units", ha='left', fontsize=11,
        color=MUT, style='italic')
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=13, color=INK)
ax.invert_yaxis()
ax.set_xlim(-2.6, 7.2)
_tk=[-2,-1,0,1,2,3,4,5,6,7]
ax.set_xticks(_tk)
ax.set_xticklabels([f"{t:+g}" if t else "0" for t in _tk], fontsize=11, color=MUT)
ax.set_xlabel("change in accuracy from standardizing  (percentage points)",
              fontsize=12, color=MUT, labelpad=10)
ax.xaxis.grid(True, color='#e6e5df', lw=.8, zorder=0)
ax.set_axisbelow(True)
for s in ('top','right','left','bottom'): ax.spines[s].set_visible(False)
ax.tick_params(length=0)

h = [plt.Rectangle((0,0),1,1,color=RED), plt.Rectangle((0,0),1,1,color=BLUE)]
ax.legend(h, ["scaling hurt","scaling helped"], loc='lower right',
          frameon=False, fontsize=12, ncol=2, bbox_to_anchor=(1.0, 1.055),
          handlelength=.9, handleheight=.9, columnspacing=1.4, labelcolor=MUT)

# ---- table
axt = fig.add_subplot(gs[2]); axt.axis('off')
xs = [0.0, 0.255, 0.405, 0.617, 0.842]
hd = ["model", "optimizer", "raw [0,1]", "standardized", "Δ points"]
top = 0.90
axt.text(0, 1.03, "3-fold CV · 50,000 MNIST images, pixels in [0,1] · mean ± sd over seeds 0/1/2",
         fontsize=12, color=MUT, va='bottom')
for x, t in zip(xs, hd):
    axt.text(x, top, t, fontsize=12.5, color=MUT, va='center')
axt.plot([0,1],[top-.055]*2, color='#c9c8c1', lw=1.1)
for i, r in enumerate(rows):
    yy = top - 0.150 - i*0.142
    win_raw = r[6] == 0
    if i >= SPLIT:
        axt.text(xs[0], yy+.024, r[0], fontsize=13.2, color=INK, va='center')
        axt.text(xs[0], yy-.036, "100 cols \u00d7 1000", fontsize=11,
                 color=BLUE, va='center')
    else:
        axt.text(xs[0], yy, r[0], fontsize=13.5, color=INK, va='center')
    axt.text(xs[1], yy, r[1], fontsize=13.5, color=INK, va='center')
    axt.text(xs[2], yy, r[2], fontsize=13.5, va='center',
             color=INK, weight='bold' if win_raw else 'normal')
    axt.text(xs[3], yy, r[3], fontsize=13.5, va='center',
             color=INK, weight='normal' if win_raw else 'bold')
    axt.text(xs[4], yy, f"{r[4]:+.2f} ± {r[5]:.2f}", fontsize=13.5, va='center',
             color=RED if win_raw else BLUE, weight='medium')
    axt.plot([0,1],[yy-.075]*2,
             color='#c9c8c1' if i == SPLIT-1 else '#ecebe5',
             lw=1.1 if i == SPLIT-1 else .9)
_cap = top - 0.150 - 5*0.142
axt.text(0, _cap-0.150,
         "Rows 2 and 4: identical architecture and data \u2014 only  solver  differs.",
         fontsize=12, color=MUT, va='center')
axt.text(0, _cap-0.228,
         "Rows 5-6: rows 2 and 1 repeated with 100 of the 784 columns multiplied by 1000.",
         fontsize=12, color=MUT, va='center')
axt.set_xlim(0,1); axt.set_ylim(-0.34,1.06)

fig.savefig('standardization_summary.png',
            dpi=120, facecolor='white', bbox_inches='tight', pad_inches=0.35)
print("saved")
