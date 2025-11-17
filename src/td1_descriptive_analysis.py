# td1_descriptive.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1) 读取数据 ==========
df = pd.read_csv("vgsales.csv")

# 统一列名（容错）
df = df.rename(columns={
    "Other_Sale":"Other_Sales",
    "NA_Sale":"NA_Sales",
    "EU_Sale":"EU_Sales",
    "JP_Sale":"JP_Sales",
    "Global_Sale":"Global_Sales"
})

cols_regions = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
target = "Global_Sales"

# 转成数值并去缺失
for c in cols_regions + [target]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=cols_regions+[target])

# ========== 2) 计算均值 & 标准差（老师要求） ==========
stats = df[[target] + cols_regions].agg(["mean","std"]).T
stats.columns = ["Mean","Std"]
print("\n=== Moyennes & Écarts-types ===")
print(stats)

# ========== 3) 全局 + 区域销量分布直方图（老师要求） ==========
for col in [target] + cols_regions:
    plt.figure()
    plt.hist(df[col], bins=30)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Histogramme de {col}")
    plt.show()

# ========== 4) 各 Genre 的全球销量均值柱状图（老师要求） ==========
genre_mean = df.groupby("Genre")[target].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
genre_mean.plot(kind='bar')
plt.ylabel("Ventes moyennes (Global_Sales)")
plt.title("Ventes globales moyennes par Genre")
plt.tight_layout()
plt.show()
