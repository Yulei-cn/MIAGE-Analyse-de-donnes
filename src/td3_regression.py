# td3_regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 0) 读取数据（每个 TD 独立文件都要有这段） =========
CSV_PATH = "vgsales.csv"   # ← 按需修改路径/文件名
df = pd.read_csv(CSV_PATH)

# 统一列名（容错，可留着）
df = df.rename(columns={
    "Other_Sale": "Other_Sales",
    "Global_Sale": "Global_Sales",
    "NA_Sale": "NA_Sales",
    "EU_Sale": "EU_Sales",
    "JP_Sale": "JP_Sales",
})

cols_regions = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
target = "Global_Sales"

# 强制为数值并丢缺失
for c in [target] + cols_regions:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df3 = df[[target] + cols_regions].dropna().copy()

# ========= 1) 最小二乘：Global_Sales ~ NA+EU+JP+Other =========
y = df3[target].to_numpy()                    # (n,)
X = df3[cols_regions].to_numpy()              # (n,4)
X = np.c_[np.ones(len(X)), X]                 # 加截距 (n,5)

beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
intercept, b_NA, b_EU, b_JP, b_Other = beta
print("coefficients:", {
    "intercept": intercept,
    "NA_Sales": b_NA,
    "EU_Sales": b_EU,
    "JP_Sales": b_JP,
    "Other_Sales": b_Other
})

# 拟合优度 R^2
y_pred = X @ beta
SS_res = np.sum((y - y_pred)**2)
SS_tot = np.sum((y - y.mean())**2)
R2 = 1 - SS_res / SS_tot
print("R^2 =", R2)

# ========= 2) 可视化：真实 vs 预测 =========
plt.figure(figsize=(5,4))
plt.scatter(y, y_pred, s=10, alpha=0.5)
mn, mx = y.min(), y.max()
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True Global_Sales")
plt.ylabel("Predicted Global_Sales")
plt.title("Global_Sales: True vs Predicted")
plt.tight_layout()
plt.show()

# ========= 3) 标准化回归（比较相对影响力） =========
Z = (df3[cols_regions] - df3[cols_regions].mean()) / df3[cols_regions].std(ddof=1)
g = (df3[target] - df3[target].mean()) / df3[target].std(ddof=1)
XZ = np.c_[np.ones(len(Z)), Z.to_numpy()]
beta_z, *_ = np.linalg.lstsq(XZ, g.to_numpy(), rcond=None)
_, bz_NA, bz_EU, bz_JP, bz_Other = beta_z
print("standardized betas:", {"NA": bz_NA, "EU": bz_EU, "JP": bz_JP, "Other": bz_Other})

# ========= 4) VIF（多重共线性粗略检查） =========
def simple_vif(M, i):
    y_ = M[:, i]
    X_ = np.delete(M, i, axis=1)
    X_ = np.c_[np.ones(len(X_)), X_]
    b_, *_ = np.linalg.lstsq(X_, y_, rcond=None)
    yhat_ = X_ @ b_
    R2_ = 1 - np.sum((y_ - yhat_)**2) / np.sum((y_ - y_.mean())**2)
    return 1.0 / (1.0 - R2_ + 1e-12)

M = df3[cols_regions].to_numpy()
vifs = {c: simple_vif(M, i) for i, c in enumerate(cols_regions)}
print("VIF:", vifs)
