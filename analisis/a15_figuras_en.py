"""A15. English versions of the article figures (for the English manuscript, obs. 4).

Same data as the Spanish figures:
- fig_v4_v5_modelos_en.png: Table 5 values (4 classes, without healthy leaf).
- fig_v4_v5_clases_en.png: metricas/ap_clase_ultralytics.csv (test180).
- fig_exposicion_en.png: exposicion/exposicion_por_imagen.csv and rendimiento_por_tercil.csv.
- fig_estres_v6_en.png: Resultados v6 estres_v6.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W = Path(__file__).parent
OUT = Path(r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final")
NAR, AZU, GRI = "#EB6835", "#2B79D6", "#4a4a4a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6"})

# 1. v4 vs v5 per architecture (Table 5, 4 classes)
mods = ["YOLOv8n", "YOLO11n", "YOLOv10n"]
datos = {"Fixed test (97 images, May)": ([17.6, 18.4, 21.3], [2.9, 1.0, 0.9], 19.2),
         "New images (83, July)": ([39.5, 32.1, 44.4], [7.4, 11.7, 6.4], 25.8)}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for a, (t, (m, s, v4)) in zip(ax, datos.items()):
    x = np.arange(3)
    a.bar(x, m, 0.55, color=NAR, label="v5: mean ± SD of 3 seeds")
    a.errorbar(x, m, s, fmt="none", ecolor=GRI, capsize=3)
    for xi, mi, si in zip(x, m, s):
        a.text(xi, mi + si + 0.8, f"{mi:.1f}", ha="center", fontsize=10)
    a.axhline(v4, color=AZU, ls="--", lw=2.5, label="v4: YOLO11n seed 1")
    a.text(-0.42, v4 - 2.5, f"{v4:.1f}", color=AZU, fontsize=10)
    a.set_xticks(x, mods); a.set_title(t); a.set_ylim(0, 54)
ax[0].set_ylabel("mAP@0.5 without healthy leaf (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig_v4_v5_modelos_en.png", dpi=200); plt.close(fig)

# 2. AP per class, full test (180)
ap = pd.read_csv(W / "metricas" / "ap_clase_ultralytics.csv")
ap = ap[ap.subconjunto == "test180"]
orden = ["polilla", "mosca_blanca", "mildiu", "oidio", "sana"]
nom = {"polilla": "tomato leafminer", "mosca_blanca": "whitefly", "mildiu": "late blight",
       "oidio": "powdery mildew", "sana": "healthy leaf"}
v4 = ap[ap.modelo.str.startswith("v4")].set_index("clase").AP50 * 100
v5 = ap[ap.modelo.str.startswith("v5")].groupby("clase").AP50.agg(["mean", "std"]) * 100
fig, a = plt.subplots(figsize=(11, 4))
x = np.arange(len(orden))
a.bar(x - 0.19, [v4[c] for c in orden], 0.36, color=AZU, label="v4 (YOLO11n s1)")
a.bar(x + 0.19, [v5.loc[c, "mean"] for c in orden], 0.36, color=NAR, label="v5 (9 runs)")
a.errorbar(x + 0.19, [v5.loc[c, "mean"] for c in orden], [v5.loc[c, "std"] for c in orden], fmt="none", ecolor=GRI, capsize=3)
for i, c in enumerate(orden):
    a.text(i - 0.19, v4[c] + 1, f"{v4[c]:.0f}", ha="center", fontsize=10)
    a.text(i + 0.19, v5.loc[c, "mean"] + v5.loc[c, "std"] + 1, f"{v5.loc[c, 'mean']:.0f}", ha="center", fontsize=10)
a.set_xticks(x, [nom[c] for c in orden]); a.set_ylabel("AP@0.5 (%)"); a.legend(frameon=False, loc="upper center")
fig.tight_layout(); fig.savefig(OUT / "fig_v4_v5_clases_en.png", dpi=200); plt.close(fig)

# 3. Exposure by time band and accuracy by saturation tercile
ex = pd.read_csv(W / "exposicion" / "exposicion_por_imagen.csv")
ter = pd.read_csv(W / "exposicion" / "rendimiento_por_tercil.csv")
ter = ter[ter.variable == "saturados"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
fr = [("manana", "morning"), ("mediodia", "midday"), ("tarde", "afternoon")]
vals = [ex[ex.franja == f].saturados for f, _ in fr]
ax[0].boxplot(vals, whis=1.5, showfliers=False, widths=0.3,
              medianprops=dict(color=NAR, lw=3), boxprops=dict(color=GRI, lw=1.5),
              whiskerprops=dict(color=GRI, lw=1.5), capprops=dict(color=GRI, lw=1.5))
ax[0].set_xticks([1, 2, 3], [f"{e}\n(n={len(v)})" for (_, e), v in zip(fr, vals)])
ax[0].set_ylabel("Saturated pixels (%)"); ax[0].set_title("Overexposure by time band (1966 images)")
nt = {"bajo": "low", "medio": "middle", "alto": "high"}
for i, (_, r) in enumerate(ter.iterrows()):
    lo, hi = map(float, str(r.acierto_ic).split("-"))
    m = r.acierto * 100
    ax[1].errorbar(i, m, [[m - lo * 100], [hi * 100 - m]], fmt="o", color=NAR, ecolor=GRI, capsize=5, ms=9, lw=2)
    ax[1].text(i + 0.08, m, f"{m:.0f}%", va="center", fontsize=10)
ax[1].set_xticks(range(len(ter)), [f"{nt[t]}\n{rg}%" for t, rg in zip(ter.tercil, ter.rango)])
ax[1].set_ylim(30, 90); ax[1].set_ylabel("Main-class accuracy (%)"); ax[1].set_title("Test: accuracy by saturation tercile")
fig.tight_layout(); fig.savefig(OUT / "fig_exposicion_en.png", dpi=200); plt.close(fig)

# 4. Stress test v6
e = pd.read_csv(r"D:/ProyectoInvestigacion/Resultados de los entrenamientos/v6-4clases/extraido/estres_v6.csv")
for k in ["mAP50", "falso_sano", "AP50_mosca_blanca", "AP50_polilla"]:
    e[k] *= 100
g = e.drop(columns=["modelo"]).groupby(["experimento", "ev"]).agg(["mean", "std"])
nombres = {"base": "Baseline", "fotometrico": "Photometric augmentation", "clahe": "CLAHE"}
col = {"base": GRI, "fotometrico": "#d9822b", "clahe": "#2b7bb9"}
pan = [("mAP50", "mAP@0.5 (4 classes, %)"), ("AP50_mosca_blanca", "AP@0.5 whitefly (%)"),
       ("AP50_polilla", "AP@0.5 tomato leafminer (%)"), ("falso_sano", "False healthy (% of infested images)")]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
for a, (k, t) in zip(ax, pan):
    for exn in ["base", "fotometrico", "clahe"]:
        s = g.loc[exn][k]; x = s.index.values
        a.plot(x, s["mean"], "-o", color=col[exn], label=nombres[exn], ms=4)
        a.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=col[exn], alpha=.12)
    a.set_title(t, fontsize=10); a.set_xlabel("Added exposure steps (EV)"); a.grid(alpha=.3, axis="both")
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_estres_v6_en.png", dpi=200); plt.close(fig)
print("ok")
