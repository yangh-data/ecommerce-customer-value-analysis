# -*- coding: utf-8 -*-
"""图表公共设置与各模块画图函数（输出 PNG 到 outputs/）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 自带微软雅黑；macOS/Linux 需安装 Noto Sans CJK）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def plot_rfm_tiers(rfm, path="outputs/rfm_tiers.png"):
    """各层客户数与交易额占比。"""
    order = ["高价值客户", "中价值客户", "低价值客户", "流失客户"]
    counts = rfm["Tier"].value_counts().reindex(order)
    shares = (
        rfm.groupby("Tier", observed=True)["Monetary"].sum().reindex(order)
        / rfm["Monetary"].sum()
        * 100
    )
    colors = [PALETTE[2], PALETTE[0], PALETTE[1], "#b9b7ae"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(counts.index, counts.values, color=colors, width=0.6)
    axes[0].set_title("各层客户数")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(shares.index, shares.values, color=colors, width=0.6)
    axes[1].set_title("各层交易额占比 (%)")
    for i, v in enumerate(shares.values):
        axes[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_elbow(inertias, path="outputs/elbow.png"):
    """肘部法则：SSE 随 K 的变化。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(inertias)
    ax.plot(ks, [inertias[k] for k in ks], marker="o", color=PALETTE[0], linewidth=2)
    ax.set_xlabel("K")
    ax.set_ylabel("SSE（惯性，越小越好）")
    ax.set_title("肘部法则：K 值选择")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pca(feats, path="outputs/pca.png"):
    """聚类结果的 PCA 二维散点图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in sorted(feats["Cluster"].unique()):
        sub = feats[feats["Cluster"] == c]
        ax.scatter(
            sub["PCA1"], sub["PCA2"], s=12,
            color=PALETTE[c % len(PALETTE)], alpha=0.6,
            label=f"聚类{c} (n={len(sub):,})",
        )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title("聚类结果 PCA 降维验证")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# 雷达图展示的关键特征（11 维中选 7 维，避免雷达图过密）
RADAR_COLS = [
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "AvgUnitPrice", "CategoryCount", "PeakSeasonRatio",
]


def plot_radar(profile, path="outputs/radar.png"):
    """各类别特征均值雷达图（按特征 min-max 归一）。"""
    vals = profile[RADAR_COLS].copy()
    vals = (vals - vals.min()) / (vals.max() - vals.min()).replace(0, 1)

    angles = np.linspace(0, 2 * np.pi, len(RADAR_COLS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for i, cluster in enumerate(vals.index):
        row = vals.loc[cluster].tolist()
        row += row[:1]
        ax.plot(angles, row, color=PALETTE[i % len(PALETTE)], linewidth=2, label=f"聚类{cluster}")
        ax.fill(angles, row, color=PALETTE[i % len(PALETTE)], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_COLS, fontsize=9)
    ax.set_title("各类别行为特征画像（归一化）")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_rules(rules, path="outputs/rules.png"):
    """关联规则散点气泡图（Top15，气泡大小=提升度）。"""
    top = rules.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        top["support"], top["confidence"], s=top["lift"] * 60,
        c=top["lift"], cmap="Blues", alpha=0.8,
    )
    for _, r in top.iterrows():
        lbl = f"{list(r['antecedents'])[0]}→{list(r['consequents'])[0]}"
        ax.annotate(lbl, (r["support"], r["confidence"]), fontsize=8)
    ax.set_xlabel("支持度")
    ax.set_ylabel("置信度")
    ax.set_title("品类关联规则 Top15（气泡大小=提升度）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trend(monthly, future, metrics, path="outputs/trend.png"):
    """月度销售趋势 + 线性拟合 + 未来预测。"""
    fig, ax = plt.subplots(figsize=(9, 4))
    months = monthly["Month"].astype(str).tolist()
    fut_months = future["Month"].astype(str).tolist()

    ax.plot(months, monthly["Amount"] / 1000, marker="o", color=PALETTE[0],
            linewidth=2, label="实际销售额（千英镑）")
    ax.plot(months, monthly["Predicted"] / 1000, "--", color=PALETTE[2],
            linewidth=1.5, label="线性拟合")
    ax.plot(fut_months, future["Amount"] / 1000, "o--", color=PALETTE[3],
            linewidth=1.5, label="预测")

    # 标注 12 月数据截断
    y_max = max(monthly["Amount"].max(), future["Amount"].max()) / 1000
    ax.axvline(len(months) - 1, color="#c3c2b7", linestyle=":", linewidth=1)
    ax.text(len(months) - 1, y_max * 0.85, "12月数据截断\n（截至12-09）",
            ha="center", fontsize=8, color="#898781")

    ax.set_xlabel("月份")
    ax.set_ylabel("销售额（千英镑）")
    ax.set_title(f"月度销售额趋势与预测（R²={metrics['r2']:.2f}，MAE=£{metrics['mae']:,.0f}）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
