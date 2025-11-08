import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 如果你已经有 df，就注释掉下一行读取
df = pd.read_csv("vgsales.csv")  # Kaggle 常见文件名，按你实际路径改

# 常见区域列
cols_regions = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
target = "Global_Sales"
meta_cols = ["Name","Platform","Year","Genre","Publisher"]

# 只保留我们需要的列（存在就取）
keep_cols = [c for c in meta_cols+[target]+cols_regions if c in df.columns]
df = df[keep_cols].copy()

# 清理：去掉区域与全球销量都为缺失或全 0 的行（可选）
df = df.dropna(subset=[target] + cols_regions)

#均值 & 标准差
stats_overall = df[[target]+cols_regions].agg(['mean','std']).T
stats_overall.columns = ['Mean','Std']
stats_overall['CoeffVar'] = stats_overall['Std'] / stats_overall['Mean']  # 变异系数
print(stats_overall)

#哪些平台或题材主导？
by_genre = df.groupby('Genre')[[target]+cols_regions].agg(['mean','std', 'count'])
# 展示全球销量均值最高的前 10 个类型
by_genre[target].sort_values(('mean'), ascending=False).head(10)


'''#全局 & 区域直方图（总体分布是否长尾）
vars_to_plot = [target] + cols_regions
for col in vars_to_plot:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.xlabel(col); plt.ylabel("Count"); plt.title(f"Histogram of {col}")
    plt.show()

genre_global_mean = df.groupby('Genre')[target].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
genre_global_mean.plot(kind='bar')
plt.ylabel("Mean Global Sales (Millions)")
plt.title("Mean Global Sales by Genre")
plt.tight_layout()
plt.show()

plt.figure()
plt.boxplot([df[c].dropna() for c in cols_regions], labels=cols_regions)
plt.title("Regional Sales (Boxplot)")
plt.show()

q = df[[target]+cols_regions].quantile([0.25,0.5,0.75])
print(q)


cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
data = [df[c].dropna() for c in cols]
p95 = [np.percentile(x, 95) for x in data]

plt.figure(figsize=(8,5))
plt.boxplot(data, labels=cols, showfliers=False)
plt.ylim(0, max(p95))  # 只看 0 ~ 各列的 95% 分位
plt.title("Regional Sales (Boxplot, 0–P95)")
plt.ylabel("Sales (Millions)")
plt.show()'''


cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
qs = [0.5, 0.75, 0.9, 0.95, 0.99]
summary = {c: np.quantile(df[c].dropna(), qs) for c in cols}
for c in cols:
    print(c, dict(zip([f"q{int(q*100)}" for q in qs], summary[c])))

cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
data = [df[c].dropna() for c in cols]
upper = max(np.percentile(x, 95) for x in data)

plt.figure(figsize=(8,5))
plt.boxplot(data, labels=cols, showfliers=False)
plt.ylim(0, upper)           # 只看 0–P95 主体部分
plt.title("Regional Sales — 0–P95 zoom")
plt.ylabel("Sales (Millions)")
plt.show()

cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]

# 零占比
for c in cols:
    zratio = (df[c]==0).mean()
    print(f"{c} zero-share = {zratio:.2%}")

# 非零直方图，统一上限（如 0–1）
for c in cols:
    nz = df.loc[df[c]>0, c]
    plt.figure()
    plt.hist(nz, bins=30, range=(0,1), density=True)
    plt.title(f"{c} (non-zero, 0–1)")
    plt.xlabel(c); plt.ylabel("Density")
    plt.show()

def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y

cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
plt.figure(figsize=(8,5))
for c in cols:
    x = df[c].dropna().values
    x[x<0] = 0
    X, Y = ecdf(x)
    plt.plot(X, Y, label=c, alpha=0.8)
plt.xscale("log")           # 关键：看长尾
plt.xlabel("Sales (log scale)"); plt.ylabel("Cumulative share")
plt.title("ECDF of Regional Sales (log x)")
plt.legend()
plt.show()

cols = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales"]
med = [np.median(df[c].dropna()) for c in cols]
q25 = [np.quantile(df[c].dropna(), 0.25) for c in cols]
q75 = [np.quantile(df[c].dropna(), 0.75) for c in cols]
iqr = [q75[i]-q25[i] for i in range(len(cols))]

x = np.arange(len(cols))
plt.figure(figsize=(8,5))
plt.bar(x, med, yerr=iqr, capsize=5)
plt.xticks(x, cols)
plt.ylabel("Median (with IQR as error bar)")
plt.title("Typical Regional Sales (Median + IQR)")
plt.show()
