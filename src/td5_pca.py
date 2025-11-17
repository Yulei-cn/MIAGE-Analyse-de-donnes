# td5_pca.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =========== 0) 读取数据 ===========

CSV_PATH = "vgsales.csv"   # 按需要修改路径
df = pd.read_csv(CSV_PATH)

# 统一列名（与 TD1/TD2/TD3 保持一致，容错用）
df = df.rename(columns={
    "Other_Sale": "Other_Sales",
    "Global_Sale": "Global_Sales",
    "NA_Sale": "NA_Sales",
    "EU_Sale": "EU_Sales",
    "JP_Sale": "JP_Sales",
})

cols_regions = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]

# 只保留地区销量 + Genre（用于上色），去掉缺失
for c in cols_regions + ["Genre"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c != "Genre" else df[c]
df_pca = df[cols_regions + ["Genre"]].dropna().copy()

# =========== 1) 标准化 & PCA ===========

X = df_pca[cols_regions].to_numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 保留 4 个主成分（最多 = 变量个数）
pca = PCA(n_components=4)
Z = pca.fit_transform(X_std)      # Z: (n, 4) 投影坐标

expl_var = pca.explained_variance_ratio_
cum_expl_var = np.cumsum(expl_var)

print("=== Variance expliquée par chaque composante ===")
for i, v in enumerate(expl_var, start=1):
    print(f"PC{i}: {v:.3f} ({cum_expl_var[i-1]:.3f} cumulé)")

# 载荷（每个主成分对原始变量的权重）
loadings = pd.DataFrame(
    pca.components_,
    columns=cols_regions,
    index=[f"PC{i}" for i in range(1, 5)]
)
print("\n=== Loadings (poids des PC sur les ventes régionales) ===")
print(loadings)

# =========== 2) Scree plot（碎石图）==========

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

# =========== 3) 圆圈图（correlation circle）PC1–PC2 ==========

# 对标准化数据，变量与 PC 的相关 ≈ loadings * sqrt(eigenvalue)
# 但做圆圈图通常直接用 loadings 的前两列即可（在[-1,1]附近）
pcs12 = pca.components_[0:2, :]          # (2, 4)
coords_var = pcs12.T                     # (4, 2)

plt.figure(figsize=(5, 5))
# 画单位圆
angle = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(angle), np.sin(angle), "k--", linewidth=0.5)

# 画变量向量
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

# 为了图不太密集，可以只取前 N 个样本显示
N_PLOT = 800    # 如果想全画，可以设为 df_pca.shape[0]
Z2 = Z[:N_PLOT, 0:2]
genres = df_pca["Genre"].astype(str).values[:N_PLOT]

# 给前若干 Genre 编码上色（多了就会有很多颜色）
unique_genres = pd.Series(genres).unique()
# 限制最多 10 种颜色，其他归为 "Other"
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

# 构造图例：显示前 max_colors 个 Genre
handles, _ = scatter.legend_elements(num=max_colors+1)
labels = list(main_genres) + ["Autres"]
plt.legend(handles, labels, title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========== 5) 小的文本总结提示（可用于写报告）===========

print("\n=== Interprétation rapide pour le rapport (à adapter dans le texte) ===")
print("- PC1 带来的方差比例：", f"{expl_var[0]:.2%}",
      " → 如果 loadings 中 NA/EU/JP/Other 在 PC1 上都是正且数值相近，可以解释为“全球销量维度 (marché global)”")
print("- PC2 的 loadings 如果例如 NA/EU 为正，而 JP 为负，则可以视为“欧美 vs 日本”维度 (dynamiques régionales).")
print("- 在 plan factoriel 中，靠原点的游戏销量整体小，远离原点的游戏是销量在某些区域极高或极低的‘极端游戏’。")
