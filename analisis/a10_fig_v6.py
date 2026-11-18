import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = r"D:/ProyectoInvestigacion/Resultados de los entrenamientos/v6-4clases/extraido/"
e = pd.read_csv(D+"estres_v6.csv"); 
for k in ["mAP50","falso_sano","AP50_mosca_blanca","AP50_polilla"]: e[k]*=100
g = e.drop(columns=["modelo"]).groupby(["experimento","ev"]).agg(["mean","std"])
nombres = {"base":"Base","fotometrico":"Aumento fotométrico","clahe":"CLAHE"}
col = {"base":"#4a4a4a","fotometrico":"#d9822b","clahe":"#2b7bb9"}
pan = [("mAP50","mAP@0.5 (4 clases, %)"),("AP50_mosca_blanca","AP@0.5 mosca blanca (%)"),
       ("AP50_polilla","AP@0.5 polilla (%)"),("falso_sano","Falso sano (% de imágenes con plaga)")]
fig, ax = plt.subplots(1,4, figsize=(15,3.8))
for a,(k,t) in zip(ax,pan):
    for ex in ["base","fotometrico","clahe"]:
        s = g.loc[ex][k]; x = s.index.values
        a.plot(x, s["mean"], "-o", color=col[ex], label=nombres[ex], ms=4)
        a.fill_between(x, s["mean"]-s["std"], s["mean"]+s["std"], color=col[ex], alpha=.12)
    a.set_title(t, fontsize=10); a.set_xlabel("Pasos de exposición añadidos (EV)"); a.grid(alpha=.3)
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout()
for p in [r"D:/ProyectoInvestigacion/analisis_revision/v6/fig_estres_v6.png",
          r"D:/ProyectoInvestigacion/redaccionarticulo/sensorpeaper/avance_final/fig_estres_v6.png",
          r"D:/ProyectoInvestigacion/app/tareas/fig_estres_v6.png"]:
    fig.savefig(p, dpi=200)
print("ok")
