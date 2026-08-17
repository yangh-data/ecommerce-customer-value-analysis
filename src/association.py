# -*- coding: utf-8 -*-
"""Apriori 品类关联规则。

商品级挖掘会组合爆炸（3,600+ 商品两两组合约 670 万种，购物篮矩阵内存超限），
因此采用品类近似映射（StockCode 首位数字 → 9 个品类），在品类级挖掘关联规则。
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# StockCode 首位数字 → 品类（数据集无商品分类字段，此为近似映射）
CATEGORY_MAP = {
    "1": "装饰品", "2": "家居用品", "3": "小件礼品",
    "4": "布艺纺织", "5": "杂项", "6": "文具办公",
    "7": "玩具模型", "8": "电子配件", "9": "其他",
}


def mine_rules(df: pd.DataFrame) -> pd.DataFrame:
    """挖掘品类级关联规则，返回按提升度降序的结果。"""
    df = df.copy()
    # '85123A' → '8' → '电子配件'
    df["Category"] = df["StockCode"].astype(str).str[0].map(CATEGORY_MAP)
    df = df.dropna(subset=["Category"])

    # 购物篮矩阵：行=交易ID，列=品类，值=是否购买（0/1，不关心买几个）
    basket = (
        df.groupby(["InvoiceNo", "Category"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    basket = basket.map(lambda x: 1 if x > 0 else 0)

    # 支持度 2%：组合至少出现在约 368 笔交易中（共 18,402 笔）
    freq = apriori(basket, min_support=0.02, use_colnames=True)
    # 置信度 20%：买了 A 的订单中至少 20% 也买了 B
    rules = association_rules(freq, metric="confidence", min_threshold=0.2)
    # 提升度排序：最能体现"意外性"，>1 才说明是真实关联而非热门品类碰巧
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)
