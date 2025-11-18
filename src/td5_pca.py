# td5_pca.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========== 0) Lecture des données ===========

CSV_PATH = "vgsales.csv"   # Modifier le chemin si nécessaire
df = pd.read_csv(CSV_PATH)

# Normaliser les noms de colonnes (cohérent avec TD1/TD2/TD3, utilisé pour la tolérance aux erreurs)
df = df.rename(columns={
    "Other_Sale": "Other_Sales",
    "Global_Sale": "Global_Sales",
    "NA_Sale": "NA_Sales",
    "EU_Sale": "EU_Sales",
    "JP_Sale": "JP_Sales",
})

cols_regions = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

# Ne garder que les ventes par région + Genre (pour la couleur), et supprimer les valeurs manquantes
for c in cols_regions + ["Genre"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c != "Genre" else df[c]
df_pca = df[cols_regions + ["Genre"]].dropna().copy()

# =========== 1) Standardisation & ACP ===========

X = df_pca[cols_regions].to_numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Conserver 4 composantes principales (au maximum = nombre de variables)
pca = PCA(n_components=4)
Z = pca.fit_transform(X_std)      # Z : (n, 4) coordonnées projetées

expl_var = pca.explained_variance_ratio_
cum_expl_var = np.cumsum(expl_var)

print("=== Variance expliquée par chaque composante ===")
for i, v in enumerate(expl_var, start=1):
    print(f"PC{i}: {v:.3f} ({cum_expl_var[i-1]:.3f} cumulé)")

# Charges (poids de chaque composante principale sur les variables d’origine)
loadings = pd.DataFrame(
    pca.components_,
    columns=cols_regions,
    index=[f"PC{i}" for i in range(1, 5)]
)
print("\n=== Loadings (poids des PC sur les ventes régionales) ===")
print(loadings)

# =========== 2) Scree plot (graphe des éboulis) ==========

plt.figure(figsize=(5, 4))
plt.plot(range(1, 5), expl_var, "o-", label="Variance expliquée")
plt.plot(range(1, 5), cum_expl_var, "s--", label="Cumul")
plt.xticks(range(1, 5))
plt.xlabel("Composante principale")
plt.ylabel("Part de variance expliquée")
plt.title("Scree plot (ACP sur les ventes régionales)")
plt.legend()
plt.tight_layout()
plt.show()

# =========== 3) Cercle de corrélation (correlation circle) PC1–PC2 ==========

# Pour des données standardisées, la corrélation entre les variables et les PC ≈ loadings * sqrt(valeur propre)
# Mais pour tracer le cercle de corrélation, on utilise généralement directement les deux premières colonnes des loadings (proches de [-1,1])
pcs12 = pca.components_[0:2, :]          # (2, 4)
coords_var = pcs12.T                     # (4, 2)

plt.figure(figsize=(5, 5))
# Tracer le cercle unité
angle = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(angle), np.sin(angle), "k--", linewidth=0.5)

# Tracer les vecteurs des variables
for i, var in enumerate(cols_regions):
    x, y = coords_var[i, 0], coords_var[i, 1]
    plt.arrow(0, 0, x, y,
              head_width=0.03, head_length=0.03,
              length_includes_head=True)
    plt.text(x * 1.05, y * 1.05, var, ha="center", va="center")

plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cercle de corrélation (ACP ventes régionales)")
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal", "box")
plt.tight_layout()
plt.show()

# =========== 4) Plan factoriel (jeux projetés sur PC1–PC2) ==========

# Pour éviter que le graphique soit trop chargé, on peut n’afficher que les N premiers échantillons
N_PLOT = 800    # Si l’on veut tout tracer, on peut mettre df_pca.shape[0]
Z2 = Z[:N_PLOT, 0:2]
genres = df_pca["Genre"].astype(str).values[:N_PLOT]

# Attribuer des codes de couleur aux premiers Genres (s’il y en a trop, il y aura beaucoup de couleurs)
unique_genres = pd.Series(genres).unique()
# Limiter à 10 couleurs au maximum, les autres sont regroupés dans "Other"
max_colors = 10
main_genres = unique_genres[:max_colors]
color_map = {g: i for i, g in enumerate(main_genres)}
genre_idx = np.array([color_map.get(g, max_colors) for g in genres])

plt.figure(figsize=(7, 6))
scatter = plt.scatter(Z2[:, 0], Z2[:, 1],
                      c=genre_idx, cmap="tab10",
                      s=8, alpha=0.6)
plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Plan factoriel (ACP ventes régionales, coloré par Genre)")

# Construire la légende : afficher les max_colors premiers Genres
handles, _ = scatter.legend_elements(num=max_colors+1)
labels = list(main_genres) + ["Autres"]
plt.legend(handles, labels, title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========== 5) Petit aide-mémoire textuel (utile pour rédiger le rapport) ===========

print("\n=== Interprétation rapide pour le rapport (à adapter dans le texte) ===")
print("- Proportion de variance apportée par PC1 :", f"{expl_var[0]:.2%}",
      " → Si les loadings de NA/EU/JP/Other sur PC1 sont tous positifs et de valeurs proches, on peut l’interpréter comme un « axe des ventes globales (marché global) ».")
print("- Si les loadings de PC2 sont par exemple positifs pour NA/EU mais négatifs pour JP, on peut l’interpréter comme un axe « Occident vs Japon » (dynamiques régionales).")
print("- Sur le plan factoriel, les jeux proches de l’origine ont des ventes globalement faibles ; les jeux éloignés de l’origine sont des « jeux extrêmes » dont les ventes sont très élevées ou très basses dans certaines régions.")

