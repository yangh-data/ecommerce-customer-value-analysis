# -*- coding: utf-8 -*-
"""一键运行全流程：清洗 → RFM 分层 → 聚类 → 关联规则 → 趋势预测。

用法:
    pip install -r requirements.txt
    python main.py      # 需先下载数据到 data/Online_Retail.xlsx（见 README）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import association, clustering, data_loader, forecast, rfm_analysis
from src.plots import plot_elbow, plot_pca, plot_radar, plot_rfm_tiers, plot_rules, plot_trend


def main():
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("步骤 1/5  数据清洗")
    df = data_loader.load_uci_online_retail()
    print(f"  有效记录 {len(df):,} 条 | 客户 {df['CustomerID'].nunique():,} 名 | "
          f"交易 {df['InvoiceNo'].nunique():,} 笔")

    print("步骤 2/5  RFM 客户价值分层")
    rfm = rfm_analysis.build_rfm(df)
    print(rfm_analysis.tier_summary(rfm).to_string())
    revenue_share = (
        rfm.groupby("Tier", observed=True)["Monetary"].sum() / rfm["Monetary"].sum() * 100
    )
    print("  各层交易额占比:\n" + revenue_share.round(1).to_string())
    plot_rfm_tiers(rfm)

    print("步骤 3/5  K-Means 聚类")
    feats = clustering.build_features(df)
    feats, profile, inertias, _ = clustering.run_clustering(feats)
    print(f"  聚类规模: {feats['Cluster'].value_counts().sort_index().to_dict()}")
    print(profile.to_string())
    plot_elbow(inertias)
    plot_pca(feats)
    plot_radar(profile)

    print("步骤 4/5  Apriori 关联规则")
    rules = association.mine_rules(df)
    print(f"  规则数 {len(rules)}，提升度 Top3:")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
          .head(3).to_string(index=False))
    plot_rules(rules)

    print("步骤 5/5  销售趋势与预测")
    monthly, future, metrics = forecast.forecast_monthly(df)
    print(f"  斜率 +£{metrics['slope']:,.0f}/月 | R²={metrics['r2']:.2f} | "
          f"MAE=£{metrics['mae']:,.0f}")
    plot_trend(monthly, future, metrics)

    print("=" * 60)
    print("全部完成，图表输出在 outputs/ 目录。")


if __name__ == "__main__":
    main()
