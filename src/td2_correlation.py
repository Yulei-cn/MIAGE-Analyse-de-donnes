import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vgsales.csv")

cols_regions = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
target = "Global_Sales"

# 只保留相关列，去掉缺失
df2 = df[[target]+cols_regions].dropna().copy()

corr_pearson = df2.corr()                 # 默认 method='pearson'
print("=== Pearson correlation matrix ===")
print(corr_pearson)

print("\n=== Pearson corr with Global_Sales (sorted) ===")
print(corr_pearson.loc[cols_regions, target].sort_values(ascending=False))


corr_spearman = df2.corr(method="spearman")
print("\n=== Spearman correlation matrix ===")
print(corr_spearman)

print("\n=== Spearman corr with Global_Sales (sorted) ===")
print(corr_spearman.loc[cols_regions, target].sort_values(ascending=False))


def scatter_global_vs(region_col):
    plt.figure()
    plt.scatter(df2[region_col], df2[target], alpha=0.3, s=8)
    plt.xlabel(region_col); plt.ylabel(target)
    plt.title(f"{region_col} vs {target} (linear scale)")
    plt.show()

for c in cols_regions:
    scatter_global_vs(c)


def scatter_loglog(region_col):
    x = df2[region_col].to_numpy()
    y = df2[target].to_numpy()
    # 只保留>0的点，避免 log(0)
    mask = (x>0) & (y>0)
    x, y = x[mask], y[mask]

    plt.figure()
    plt.scatter(x, y, alpha=0.3, s=8)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(region_col); plt.ylabel(target)
    plt.title(f"{region_col} vs {target} (log-log)")
    plt.show()

for c in cols_regions:
    scatter_loglog(c)


def hexbin_plot(region_col, gridsize=40):
    plt.figure()
    plt.hexbin(df2[region_col], df2[target], gridsize=gridsize)
    plt.xlabel(region_col); plt.ylabel(target)
    plt.title(f"{region_col} vs {target} (hexbin)")
    plt.show()

# 如有需要再跑：
# for c in cols_regions:
#     hexbin_plot(c)
