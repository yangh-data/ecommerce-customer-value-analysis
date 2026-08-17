# -*- coding: utf-8 -*-
"""数据加载与清洗。

输入: UCI Online Retail 数据集 (data/Online_Retail.xlsx)
输出: 清洗后的 DataFrame

清洗规则（每一条都基于业务判断，而非无差别删除）:
  1. 剔除客户编号缺失的匿名交易（约 135,080 条）——匿名访客无法用于客户层面分析
  2. 剔除退货记录（约 8,905 条）——数量为负，属于退货而非购买
  3. 剔除无效价格（约 40 条）——单价非正值，无业务含义
  4. 剔除非商品条目（约 1,547 条）——如邮费、手续费等，商品代码不以数字开头
"""
import pandas as pd


def load_uci_online_retail(filepath="data/Online_Retail.xlsx") -> pd.DataFrame:
    # ID 列指定为字符串：避免 CustomerID 17850 被读成 17850.0，后续聚合才能正确分组
    df = pd.read_excel(filepath, dtype={"CustomerID": str, "InvoiceNo": str})

    # ---- 清洗（口径与报告一致）----
    df = df.dropna(subset=["CustomerID"])                  # 1. 匿名交易
    df = df[df["Quantity"] > 0]                            # 2. 退货记录
    df = df[df["UnitPrice"] > 0]                           # 3. 无效价格
    df = df[df["StockCode"].astype(str).str.match(r"^\d")]  # 4. 非商品条目

    # ---- 特征工程 ----
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Amount"] = df["Quantity"] * df["UnitPrice"]        # 单笔交易金额
    df["Month"] = df["InvoiceDate"].dt.to_period("M")      # 所属月份（趋势分析用）
    df["IsWeekend"] = (df["InvoiceDate"].dt.dayofweek >= 5).astype(int)
    df["IsPeakSeason"] = df["InvoiceDate"].dt.month.isin([11, 12]).astype(int)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_uci_online_retail()
    print(f"清洗后记录数: {len(df):,}")
    print(f"客户数: {df['CustomerID'].nunique():,}")
    print(f"交易数: {df['InvoiceNo'].nunique():,}")
