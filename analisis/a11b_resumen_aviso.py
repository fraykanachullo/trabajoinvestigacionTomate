import pandas as pd, numpy as np
W = r"D:/ProyectoInvestigacion/analisis_revision/v6/"
d = pd.read_csv(W + "aviso_luz.csv")
out = []
p = lambda *a: out.append(" ".join(str(x) for x in a))
pl = d[d.con_plaga]
p("== Por nivel (imagenes con plaga n=%d por nivel) ==" % (len(pl) // 6))
t = pl.groupby("ev").agg(aviso=("aviso", "mean"), acierto=("acierto", "mean"), falso_sano=("falso_sano", "mean"),
                         sobre240_med=("pct_sobre240", "median"), media_med=("media_lum", "median"))
t[["aviso", "acierto", "falso_sano"]] *= 100
p(t.round(1).to_string())
# falla inducida por la luz: acierta en ev0 y falla en ev>0
b0 = pl[pl.ev == 0].set_index("imagen")["acierto"]
s = pl[pl.ev > 0].copy(); s["ok0"] = s.imagen.map(b0)
s = s[s.ok0]                                  # solo las que aciertan con la exposicion original
s["falla"] = ~s.acierto
p("\n== Fallas inducidas por la luz (imagenes que aciertan en EV 0; pares imagen-nivel EV>0: %d, fallas %d) ==" % (len(s), s.falla.sum()))
def met(sub, col="aviso"):
    tp = (sub[col] & sub.falla).sum(); fn = (~sub[col] & sub.falla).sum()
    fp = (sub[col] & ~sub.falla).sum(); tn = (~sub[col] & ~sub.falla).sum()
    sens = tp / (tp + fn) if tp + fn else np.nan; esp = tn / (tn + fp) if tn + fp else np.nan
    vpp = tp / (tp + fp) if tp + fp else np.nan
    fr_a = sub[sub[col]].falla.mean(); fr_n = sub[~sub[col]].falla.mean()
    return sens, esp, vpp, fr_a, fr_n
se, es, vpp, fa, fn_ = met(s)
p(f"Aviso actual (media>200 o >15%>240): sensibilidad {se:.2f}, especificidad {es:.2f}, VPP {vpp:.2f}; tasa de falla con aviso {fa:.2f} vs sin aviso {fn_:.2f}")
# falsas alarmas en las fotos reales (EV 0)
r0 = d[d.ev == 0]
p(f"Fotos reales (EV 0, 180): aviso en {r0.aviso.mean()*100:.1f}% ; pct>240 mediana {r0.pct_sobre240.median():.2f}, p90 {r0.pct_sobre240.quantile(.9):.2f}, max {r0.pct_sobre240.max():.2f}")
# barrido de umbral en % de pixeles > 240 (sin criterio de media)
p("\n== Barrido: aviso si % pixeles lum>240 > umbral ==")
p("umbral  sens  esp  VPP  aviso_EV0(%)")
best = None
for u in [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30]:
    s["a2"] = s.pct_sobre240 > u
    se, es, vpp, _, _ = met(s, "a2")
    a0 = (r0.pct_sobre240 > u).mean() * 100
    p(f"{u:6}  {se:.2f}  {es:.2f}  {vpp:.2f}  {a0:.1f}")
    j = se + es - 1
    if best is None or j > best[0]: best = (j, u)
p(f"Mejor Youden: umbral {best[1]}% (J={best[0]:.2f})")
txt = "\n".join(out); print(txt); open(W + "aviso_luz_resumen.txt", "w", encoding="utf-8").write(txt)
