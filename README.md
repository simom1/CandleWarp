# A-Shape Similarity Demo

这个 demo 只回答目标 A：

> 历史上出现过类似形态时，后面 N 根 K 线的走势分布长什么样？

它不是预测模型，也不输出买卖信号。它做三件事：

1. 把 OHLC 窗口编码到去价格量纲、去波动率量纲的相对空间。
2. 先按“大级别趋势方向 x 波动率分位”分层，再在同状态历史片段里找相似窗口。
3. 画出 Top-K 相似片段之后 N 根 K 线的分位数带和所有样本路径。

## 数据格式

我们统一使用 **1h K 线**。CSV 至少需要这些列，大小写不敏感：

```csv
timestamp,open,high,low,close
2025-01-01 09:00:00,2632.1,2635.0,2630.5,2634.2
```

`timestamp` 可以叫 `time`、`date` 或 `datetime`。如果没有时间列，也可以用纯序号数据。

## 快速运行

先生成一份可跑通流程的样例数据：

```bash
python3 -m a_shape_tool.cli --make-sample data/sample_ohlc.csv --rows 900
```

然后运行相似形态分布分析：

```bash
python3 -m a_shape_tool.cli \
  --csv data/sample_ohlc.csv \
  --timeframe 1h \
  --window 100 \
  --horizon 50 \
  --top-k 50 \
  --output-dir output
```

输出：

- `output/distribution.png`：Top-K 相似窗口后续走势分布图
- `output/matches.csv`：匹配片段明细
- `output/report.html`：简单 HTML 报告

## Walk-forward 验证

要判断它是否真的有用，不看单次图，而是做无未来函数的滚动验证：

```bash
python3 -m a_shape_tool.cli \
  --csv data/sample_ohlc.csv \
  --timeframe 1h \
  --window 100 \
  --horizon 50 \
  --top-k 50 \
  --backtest \
  --min-history 1000 \
  --stride 50 \
  --cost-bps 2 \
  --output-dir output
```

每个评估点只使用当时已经发生的历史数据。验证输出：

- `output/backtest_trials.csv`：每个 walk-forward 样本的分布预测、真实后续收益、误差和交易规则结果
- `output/backtest_buckets.csv`：按相似分布中位数分桶后的真实收益，用来看是否单调
- `output/backtest_summary.json`：总体指标

核心判断不是单纯胜率，而是：

- 相似分布的中位数误差是否低于“同市场状态、不看形态”的基准。
- 相似分布中位数和真实未来收益是否有正的 Spearman IC。
- 真实结果落在 25%-75%、10%-90% 分位带里的比例是否合理。
- 如果用保守规则 `q25 > threshold` 做多、`q75 < -threshold` 做空，扣成本后是否仍有正收益。

## 关键参数

- `--timeframe`：报告里的周期标记，默认 `1h`。
- `--window`：用于匹配的形态窗口长度。1h 数据下建议重点看 `50-200`，默认 `100`。
- `--horizon`：相似片段后面要观察多少根 K 线。1h 数据下默认 `50`。
- `--top-k`：取多少个最相似历史片段。大窗口下默认 `50`，分位数会比 `20` 稳一点。
- `--min-match-gap`：Top-K 之间至少间隔多少根 K 线，默认 `max(window, horizon)`，避免同一段行情被重复切片。
- `--trend-lookback`：大级别趋势分层的回看长度。
- `--atr-period`：ATR 归一化周期。
- `--query-end`：查询窗口结束位置，默认 `last`，也可以传行号或时间。

## 方法说明

每段窗口会被编码成每根 K 线 5 个特征：

- `rel_open = (open - first_close) / atr`
- `rel_high = (high - first_close) / atr`
- `rel_low = (low - first_close) / atr`
- `rel_close = (close - first_close) / atr`
- `body_ratio = (close - open) / (high - low)`

距离使用欧氏距离。相似检索默认只使用查询窗口之前的历史片段，避免把查询之后的数据混进去。

如果 `--query-end` 指向历史上的某个时间点，候选片段的后续 N 根 K 线也必须在该时间点之前已经发生，避免回测时借用未来数据。
