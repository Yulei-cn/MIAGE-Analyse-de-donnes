# td1_descriptive.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1) Lecture des données ==========
df = pd.read_csv("vgsales.csv")

# Normaliser les noms de colonnes (tolérance aux erreurs)
df = df.rename(columns={
    "Other_Sale":"Other_Sales",
    "NA_Sale":"NA_Sales",
    "EU_Sale":"EU_Sales",
    "JP_Sale":"JP_Sales",
    "Global_Sale":"Global_Sales"
})

cols_regions = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
target = "Global_Sales"

# Convertir en valeurs numériques et supprimer les valeurs manquantes
for c in cols_regions + [target]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols_regions+[target])

# ========== 2) Calcul des moyennes & écarts-types (exigence du professeur) ==========
stats = df[[target] + cols_regions].agg(["mean","std"]).T
stats.columns = ["Mean","Std"]
print("\n=== Moyennes & Écarts-types ===")
print(stats)

# ========== 3) Histogrammes de la distribution des ventes globales + régionales (exigence du professeur) ==========
for col in [target] + cols_regions:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Histogramme de {col}")
    plt.show()

# ========== 4) Diagramme en barres des ventes globales moyennes par genre (exigence du professeur) ==========
genre_mean = df.groupby("Genre")[target].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
genre_mean.plot(kind='bar')
plt.ylabel("Ventes moyennes (Global_Sales)")
plt.title("Ventes globales moyennes par Genre")
plt.tight_layout()
plt.show()
