# -*- coding: utf-8 -*-
"""K-Means 客户聚类 + PCA 降维验证。

在 RFM 三维基础上扩展至 11 维行为特征，Z-score 标准化消除量纲影响，
肘部法则观察 K 取值，PCA 压缩到二维验证分离度。
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 参与聚类的 11 维行为特征
CLUSTER_COLS = [
    "Recency",           # 最近购买间隔（天）
    "Frequency",         # 交易次数
    "Monetary",          # 总消费金额
    "AvgOrderValue",     # 平均客单价
    "AvgUnitPrice",      # 平均商品单价
    "ItemsPerOrder",     # 平均每单商品数
    "CategoryCount",     # 购买品类广度
    "WeekendRatio",      # 周末订单占比
    "PeakSeasonRatio",   # 旺季（11-12月）订单占比
    "IntervalMean",      # 平均购买间隔（天）
    "IntervalStd",       # 购买间隔波动（标准差）
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """按客户聚合 11 维行为特征。"""
    ref = df["InvoiceDate"].max()

    # 每笔订单汇总
    orders = (
        df.groupby(["CustomerID", "InvoiceNo"])
        .agg(
            OrderAmount=("Amount", "sum"),
            OrderItems=("Quantity", "sum"),
            OrderDate=("InvoiceDate", "max"),
        )
        .reset_index()
    )
    per_order = orders.groupby("CustomerID").agg(
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("OrderAmount", "sum"),
        AvgOrderValue=("OrderAmount", "mean"),
        ItemsPerOrder=("OrderItems", "mean"),
        Recency=("OrderDate", lambda x: (ref - x.max()).days),
    )

    # 相邻两单的购买间隔（只有 1 单的客户间隔记为 0）
    orders = orders.sort_values(["CustomerID", "OrderDate"])
    orders["Interval"] = orders.groupby("CustomerID")["OrderDate"].diff().dt.days
    interval_stats = (
        orders.groupby("CustomerID")["Interval"]
        .agg(IntervalMean="mean", IntervalStd="std")
        .fillna(0)
    )

    # 其余行为特征
    cust = df.copy()
    cust["Cat"] = cust["StockCode"].astype(str).str[0]  # 品类近似（见 association.py）
    cust_agg = cust.groupby("CustomerID").agg(
        AvgUnitPrice=("UnitPrice", "mean"),
        WeekendRatio=("IsWeekend", "mean"),
        PeakSeasonRatio=("IsPeakSeason", "mean"),
        CategoryCount=("Cat", "nunique"),
    )

    feats = per_order.join([cust_agg, interval_stats])
    return feats.reset_index().fillna(0)


def run_clustering(feats: pd.DataFrame):
    """标准化 → 肘部法则 → K-Means(K=4) → PCA 验证。

    返回 (带聚类标签的特征表, 各类别均值画像, 各 K 的惯性, 模型)
    """
    X = feats[CLUSTER_COLS]

    # 标准化：K-Means 用欧式距离，消费额（£10^4 量级）会碾压次数（10 量级）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 肘部法则：K=2 出现明显拐点；结合业务可解释性最终取 K=4
    inertias = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias[k] = km.inertia_

    # random_state=42 固定随机种子保证可复现；n_init=10 从 10 组起点各跑一遍取最优
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    feats = feats.copy()
    feats["Cluster"] = kmeans.fit_predict(X_scaled)

    # PCA 压到二维，用于散点图验证聚类分离度
    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(X_scaled)
    feats["PCA1"], feats["PCA2"] = pca_xy[:, 0], pca_xy[:, 1]
    print(f"PCA 前两个主成分解释方差: {pca.explained_variance_ratio_.sum():.1%}")

    profile = feats.groupby("Cluster")[CLUSTER_COLS].mean().round(2)
    return feats, profile, inertias, kmeans
