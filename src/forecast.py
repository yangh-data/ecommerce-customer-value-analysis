# -*- coding: utf-8 -*-
"""月度销售趋势与线性回归预测。

注意：数据截止 2011-12-09，12 月为不完整月份——趋势图中 12 月骤降
是数据截断所致，而非业务下滑。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def forecast_monthly(df: pd.DataFrame, periods: int = 3):
    """按月份聚合销售额，线性回归拟合趋势并外推预测。

    返回 (月度实际值, 未来预测值, 指标字典)
    """
    monthly = df.groupby("Month")["Amount"].sum().reset_index()
    monthly["t"] = np.arange(len(monthly))  # 月份序号 0..12

    model = LinearRegression()
    model.fit(monthly[["t"]], monthly["Amount"])
    monthly["Predicted"] = model.predict(monthly[["t"]])

    metrics = {
        "slope": float(model.coef_[0]),  # 平均每月增长额
        "r2": r2_score(monthly["Amount"], monthly["Predicted"]),
        "mae": mean_absolute_error(monthly["Amount"], monthly["Predicted"]),
    }

    t_future = np.arange(len(monthly), len(monthly) + periods)
    future = pd.DataFrame({
        "Month": pd.period_range(monthly["Month"].max() + 1, periods=periods, freq="M"),
        "Amount": model.predict(t_future.reshape(-1, 1)),
    })
    return monthly, future, metrics
