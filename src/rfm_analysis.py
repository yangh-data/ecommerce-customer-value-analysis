# -*- coding: utf-8 -*-
"""RFM 客户价值分层。

R（最近一次购买距参考日的天数）/ F（观察期内交易次数）/ M（总消费金额）
三维各自打分后汇总为综合得分，再按占比切分为四层。
"""
import numpy as np
import pandas as pd


def safe_qcut(series: pd.Series, q: int, ascending: bool = True) -> pd.Series:
    """降级打分函数：等频分箱失败时自动退回等距分箱。

    真实数据中半数以上客户只购买过一次（F=1），qcut 因大量重复边界值无法切分。
    qcut 按人数均分，cut 按数值均分，两者互为兜底，保证任何分布形态下
    所有客户都能获得合理评分。
    """
    try:
        cats = pd.qcut(series, q=q, labels=False, duplicates="drop")
    except Exception:
        cats = pd.cut(series, bins=min(q, series.nunique()), labels=False)
    n = cats.nunique()
    # R 越小越好 → 反向打分（ascending=False）；F / M 越大越好 → 正向打分
    score_map = {i: (i + 1 if ascending else n - i) for i in range(n)}
    return pd.Series([score_map.get(c, 1) for c in cats], index=series.index)


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """聚合 R/F/M 三个维度并分层。"""
    ref_date = df["InvoiceDate"].max()  # 参考日 = 数据中最晚交易日期
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Amount", "sum"),
    ).reset_index()

    rfm["R_Score"] = safe_qcut(rfm["Recency"], q=5, ascending=False)
    rfm["F_Score"] = safe_qcut(rfm["Frequency"], q=5, ascending=True)
    rfm["M_Score"] = safe_qcut(rfm["Monetary"], q=5, ascending=True)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    # 按综合得分排序后按占比切层（口径与报告一致：14.9% / 26.5% / 40.3% / 18.3%）
    pct = rfm["RFM_Score"].rank(pct=True)
    rfm["Tier"] = pd.cut(
        pct,
        bins=[-np.inf, 0.183, 0.586, 0.851, np.inf],
        labels=["流失客户", "低价值客户", "中价值客户", "高价值客户"],
    )
    return rfm


def tier_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """各层客户的关键指标汇总。"""
    return (
        rfm.groupby("Tier", observed=True)
        .agg(
            客户数=("CustomerID", "count"),
            平均购买次数=("Frequency", "mean"),
            最近购买间隔=("Recency", "mean"),
            人均消费=("Monetary", "mean"),
        )
        .reindex(["高价值客户", "中价值客户", "低价值客户", "流失客户"])
        .round(1)
    )
