# 电商客户价值挖掘与运营策略分析

> 个人项目 · AI 辅助完成 · 公开数据集（已匿名化脱敏）· 可复现

📊 **交互式分析报告（在线版）**：https://<你的用户名>.github.io/ecommerce-customer-value-analysis/

基于英国某在线零售商 2010.12–2011.12 的真实交易数据（UCI Online Retail），完整走通「数据清洗 → 客户价值分层 → 聚类画像 → 关联规则 → 趋势预测」的分析全流程，并以交互式可视化报告呈现结果。

## 项目要回答的四个业务问题

1. 客户价值分布如何？是否存在头部集中？
2. 不同客户群体在频次、客单价、活跃度上有何差异？
3. 哪些品类之间存在稳定的交叉购买关系？
4. 销售如何随时间波动？如何指导备货与促销节奏？

## 核心结论

| 发现 | 结果 |
|---|---|
| 头部集中 | **14.9% 的高价值客户贡献约 57% 的交易额**，比经典二八法则更极端 |
| 聚类画像 | K-Means（K=4）识别出 4 类客群，含约 15 名超高频批发型客户（人均 78.9 次购买） |
| 品类关联 | 装饰品 / 布艺纺织 / 电子配件交叉购买信号明确，最高提升度 2.21 |
| 销售趋势 | 9–11 月为旺季；线性回归斜率 +£33,948/月，R²=0.37（季节性强，不宜精确预测） |

## 技术栈

Python · Pandas · Scikit-learn（KMeans / StandardScaler / PCA / LinearRegression）· MLxtend（Apriori）· Matplotlib · ECharts（交互报告）

## 目录结构

```
├── README.md
├── requirements.txt
├── LICENSE
├── index.html          # 交互式分析报告（ECharts）
├── main.py             # 一键运行全流程
└── src/
    ├── data_loader.py  # 数据加载与清洗（4 条业务化规则 + 特征工程）
    ├── rfm_analysis.py # RFM 分层（safe_qcut 降级打分）
    ├── clustering.py   # K-Means 聚类 + PCA 验证（11 维行为特征）
    ├── association.py  # Apriori 品类关联规则（支持度 2% / 置信度 20%）
    ├── forecast.py     # 月度销售趋势与线性回归预测
    └── plots.py        # 图表绘制（输出 outputs/）
```

## 快速开始

```bash
# 1. 安装依赖（Python 3.9+）
pip install -r requirements.txt

# 2. 下载数据集（约 23MB），解压后放到 data/ 目录并重命名
#    UCI Machine Learning Repository:
#    https://archive.ics.uci.edu/dataset/352/online+retail
#    最终路径: data/Online_Retail.xlsx

# 3. 运行
python main.py
```

图表输出至 `outputs/`，控制台打印各步骤关键数字。

> 图表中文显示依赖系统中文字体：Windows 自带微软雅黑即可；macOS / Linux 请安装 Noto Sans CJK。

## 实现要点

- **safe_qcut 降级打分**：真实数据中半数客户仅购买一次，等频分箱（qcut）因重复边界值失效；实现 try/except 降级——qcut 失败自动退回等距分箱（cut）并动态映射评分
- **标准化**：聚类前 Z-score 标准化消除量纲差异（消费额 £10⁴ 量级 vs 购买次数 10 量级），避免大数值特征主导距离计算
- **K 值选择**：肘部法则（K=2 出现拐点）+ 业务可解释性权衡，最终取 K=4；random_state=42 保证结果可复现
- **品类近似映射**：数据集无商品分类字段，以 StockCode 首位数字近似映射为 9 个品类；商品级组合爆炸（3,600+ 商品两两组合约 670 万种）在单机不可行，品类级聚合后购物篮矩阵大幅缩小
- **12 月截断识别**：数据截止 2011-12-09，趋势图中 12 月骤降为截断所致而非业务下滑——靠可视化诊断出来的口径陷阱

## 数据说明与脱敏声明

UCI Online Retail 为公开发布的学术数据集，客户编号已由发布方匿名化处理，不含任何个人身份信息。本项目仅使用聚合统计结果，不展示个体级交易明细。

## AI 使用声明

本项目使用 AI 工具（豆包 / ChatGPT 等）辅助完成：需求拆解、代码初稿生成、图表配色与布局优化。所有统计口径、阈值选择与结论均由本人核验确认，代码可在本地完整复现。

## 局限与改进方向

- 聚类对离群点敏感（存在仅 2 人的极端簇），可尝试 DBSCAN / 层次聚类对比
- 品类映射粗糙，若能获取真实商品分类树，规则质量与可解释性会显著提升
- 销售预测仅用线性趋势，可引入 SARIMA / Prophet 处理季节性

## 参考资料

- [1] Chen D, et al. Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal of Database Marketing & Customer Strategy Management, 2012. [UCI 数据集来源]
- [2] Hughes A M. Strategic Database Marketing (1st Edition). Chicago: Probus Publishing, 1994. [RFM 模型原始出处]
- [3] Agrawal R, Srikant R. Fast algorithms for mining association rules. VLDB, 1994.
