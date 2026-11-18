"""Análisis de resultados v6 (4 clases): base vs fotometrico vs clahe, por corrida."""
import sys, pandas as pd, numpy as np
from scipy import stats
D = r"D:/ProyectoInvestigacion/Resultados de los entrenamientos/v6-4clases/extraido/"
OUT = r"D:/ProyectoInvestigacion/analisis_revision/v6/"
c = pd.read_csv(D+"comparacion_v6.csv"); e = pd.read_csv(D+"estres_v6.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
m = ["valid_mAP50","test_mAP50","prueba_fija_mAP50","nuevas_mAP50","test_falso_sano",
     "test_AP50_mildiu","test_AP50_mosca_blanca","test_AP50_oidio","test_AP50_polilla"]
c[m] = c[m]*100
print(c.groupby("experimento")[m].agg(["mean","std"]).round(1).T)
# diferencias pareadas por (modelo, semilla) frente a base
b = c[c.experimento=="base"].set_index(["modelo","semilla"])
for ex in ["fotometrico","clahe"]:
    x = c[c.experimento==ex].set_index(["modelo","semilla"])
    print(f"\n== {ex} - base (9 pares modelo/semilla) ==")
    for k in m:
        d = (x[k]-b[k]).dropna()
        w = stats.wilcoxon(d).pvalue if (d!=0).any() else 1
        print(f"{k:26s} dif {d.mean():+6.1f} (DE {d.std():4.1f})  pares mejores {int((d>0).sum())}/9  Wilcoxon p={w:.3f}")
# robustez: caida relativa con +1 y +1.5 respecto a ev 0
e2 = e.copy(); cols=["mAP50","falso_sano","AP50_mildiu","AP50_mosca_blanca","AP50_oidio","AP50_polilla"]
e2[cols]*=100
piv = e2.pivot_table(index=["experimento","modelo","semilla"], columns="ev", values="mAP50")
mb = e2.pivot_table(index=["experimento","modelo","semilla"], columns="ev", values="AP50_mosca_blanca")
fs = e2.pivot_table(index=["experimento","modelo","semilla"], columns="ev", values="falso_sano")
r = pd.DataFrame({"mAP_0":piv[0.0],"mAP_+1":piv[1.0],"ret_mAP_+1":piv[1.0]/piv[0.0]*100,
                  "ret_mAP_+1.5":piv[1.5]/piv[0.0]*100,"ret_mAP_+2":piv[2.0]/piv[0.0]*100,
                  "MB_0":mb[0.0],"MB_+1":mb[1.0],"ret_MB_+1":mb[1.0]/mb[0.0]*100,
                  "FS_0":fs[0.0],"FS_+1":fs[1.0],"dFS_+1":fs[1.0]-fs[0.0]}).reset_index()
print("\n== Robustez (retencion % = metrica con sobreexposicion / metrica original) ==")
print(r.drop(columns=["modelo","semilla"]).groupby("experimento").agg(["mean","std"]).round(1).T)
rb = r[r.experimento=="base"].set_index(["modelo","semilla"])
for ex in ["fotometrico","clahe"]:
    x = r[r.experimento==ex].set_index(["modelo","semilla"])
    print(f"\n-- {ex} - base robustez --")
    for k in ["mAP_+1","ret_mAP_+1","ret_mAP_+1.5","ret_mAP_+2","MB_+1","ret_MB_+1","dFS_+1"]:
        d=(x[k]-rb[k]); print(f"{k:14s} dif {d.mean():+6.1f} (DE {d.std():4.1f}) mejores {(d>0).sum()}/9 p={stats.wilcoxon(d).pvalue:.3f}")
print("\n== Robustez por arquitectura ==")
print(r.groupby(["experimento","modelo"])[["mAP_0","mAP_+1","ret_mAP_+1","ret_mAP_+2","ret_MB_+1"]].mean().round(1))
r.round(2).to_csv(OUT+"robustez_v6.csv", index=False)
g = e2.groupby(["experimento","ev"])[cols].agg(["mean","std"]).round(1); g.to_csv(OUT+"estres_v6_resumen.csv")
print("\n== estres por clase (media) ==")
print(e2.groupby(["experimento","ev"])[cols].mean().round(1))
