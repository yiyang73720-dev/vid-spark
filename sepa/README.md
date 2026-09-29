# SEPA 选股引擎（Minervini）

把 Mark Minervini 的 SEPA 四要素做成可复现的检查：

| 支柱 | 含义 | 由谁判断 |
|---|---|---|
| **T** Trend 趋势 | Weinstein Stage 2 上升段 + 8 条 Trend Template | `python3 -m sepa scan`（纯机械） |
| **E** Earnings 基本面 | 季度 EPS 同比 ≥ 20–25% 且加速、营收同步加速、利润率扩张、超预期 / 上修指引 | 引擎先算数字，`/sepa-screen` 工作流里的 agent 用财报核实补全 |
| **A** Announcement 催化剂 | 机构为什么**现在**必须买：新品、新市场、上修指引、监管批准、行业拐点 | 只能研究判断，由工作流 agent 负责 |
| **S** Specific entry 入场点 | VCP 最后一次收缩的 pivot 放量突破，不追超过 pivot 5% 的票 | 引擎识别 VCP / pivot / 止损，agent 复核 |

Stage 1 不买，Stage 3/4 不做多：`trend` 支柱要求 Stage 2 **且** 8 条全部通过。

## 8 条 Trend Template（`sepa/trend.py`）

1. 股价在 150 日和 200 日均线之上
2. 150 日均线在 200 日均线之上
3. 200 日均线至少连续上升 1 个月（输出里会给出连续上升的月数，Minervini 更喜欢 4–5 个月以上）
4. 50 日均线在 150 日和 200 日均线之上
5. 股价在 50 日均线之上
6. 股价比 52 周低点至少高 30%
7. 股价距 52 周高点不超过 25%
8. RS 相对强度评级 ≥ 70（越接近 80–90 越好）

每条结果为 `true / false / null`；`null` 表示数据不足以判断（例如历史不够 200 天、没有 RS 评级），**不会**被当成通过。

**RS 评级**：按 IBD 的方式计算加权涨幅（近 3 个月 40%，前三个季度各 20%），再在数据目录里的**全部**股票中排百分位（1–99）。少于 20 只股票时百分位没有意义，第 8 条记为未知。解决办法：放入更大的股票池（比如 S&P 500 + Nasdaq 100），或用 `--rs NVDA=95` 手动填入 IBD / MarketSmith 的评级。

## VCP 与入场（`sepa/vcp.py`）

- 基底从回看窗口（最长 65 周）内的最高点开始，第一次收缩 = 最高点 → 之后的最低点。
- 之后的摆动用百分比 zigzag 识别，阈值随个股波动率自适应（1.5 × 中位真实波幅，限制在 2.5%–6%）。
- 从更高高点开始、却比前一段更深的回调，视为底部反弹中的小回调，不计为收缩。
- 有效 VCP：2–6 次收缩、逐次变小、第一次 ≤ 50%、最后一次 ≤ 10%、pivot 距基底高点 ≤ 15%、最后一段成交量低于第一段、基底前有 ≥ 25% 的上涨。
- pivot = 最后一次收缩的高点；止损 = 最后一次收缩的低点（风险 > 10% 会标记 `risk_ok: false`）。

入场状态：

| status | 含义 |
|---|---|
| `forming` | VCP 有效，价格距 pivot 超过 3% |
| `near_pivot` | 距 pivot 3% 以内：在 pivot 设买入止损单 |
| `breakout` | 收盘突破 pivot，成交量 ≥ 50 日均量的 140%，且仍在 pivot +5% 以内：可买 |
| `breakout_unconfirmed` | 突破但缩量 |
| `extended` | 已高于 pivot 5% 以上：不追，等新的基底 |
| `failed_breakout` / `stopped_out` | 突破后跌回 pivot 下方 / 跌破止损 |
| `no_base` | 没有有效 VCP（`reasons` 里写明原因） |

## 用法

```bash
# 1) 准备数据：需要 yfinance 和能访问 Yahoo Finance 的网络
pip install yfinance
python3 -m sepa fetch --tickers NVDA ANET CRWD --tickers-file my_universe.txt --benchmark SPY

# 或者自己放 CSV：sepa-data/prices/<TICKER>.csv（Date,Open,High,Low,Close,Volume）
# 基本面可选：sepa-data/fundamentals/<TICKER>.json（格式见 sepa/fundamentals.py 顶部）

# 2) 机械筛选（不需要任何第三方库）
python3 -m sepa scan --tickers NVDA ANET CRWD            # 表格
python3 -m sepa scan --json                              # 完整 JSON
python3 -m sepa scan --out-dir sepa-data/reports         # 每只股票一个 JSON + summary.json
python3 -m sepa scan --asof 2025-06-30                   # 回看历史某一天（基本面按 reported 日期过滤）
python3 -m sepa scan --config my_thresholds.json         # 覆盖 sepa/config.py 里的任何阈值
```

`my_thresholds.json` 示例（比如 52 周低点用 25% 的旧版标准）：

```json
{"trend": {"min_above_low": 0.25}, "vcp": {"breakout_volume_ratio": 1.5}}
```

## 完整 SEPA 工作流：`/sepa-screen`

`.claude/workflows/sepa-screen.js` 是一个 Claude Code 工作流，在 Claude Code 里输入例如：

```
/sepa-screen tickers NVDA ANET CRWD AXON, universeFile sp500.txt
```

流程：

1. **Mechanical**：一个 agent 跑 `fetch` + `scan`，只转述数字不做判断；不满足 Stage 2 + Trend Template 的直接淘汰（日志列出每只被淘汰的票和失败的条目）。
2. 对每只候选股并行：**Earnings**（用财报 / 10-Q / 新闻核实 EPS 与营收加速、利润率、超预期、指引）、**Catalyst**（机构为什么现在必须买，以及近期的二元事件风险）、**Entry**（对照周线复核引擎的 VCP 判断，给出买点区间和止损）。
3. **Verify**：两个唱反调的 agent（基本面/催化剂视角、技术/时机视角）尝试推翻；E 或 A 已失败的直接淘汰，不再验证。
4. 分级：`BUY`（四项全过、正在买点区间、无人推翻）> `READY`（四项全过、贴近 pivot）> `WATCH`（有一项待确认，或已拉远，或被一方质疑）> `REJECT`。RS 未验证时最高只到 `WATCH`。
5. **Report**：生成中文报告 `sepa-data/reports/sepa-report.md`。

参数：`tickers`、`universeFile`（用于 RS 排名的股票池）、`dataDir`（默认 `sepa-data`）、`benchmark`（默认 `SPY`）、`fetch`（默认 `true`）、`asof`、`rs`（如 `{"NVDA": 95}`）、`maxCandidates`（默认 12，超出部分会在日志里列出）。

## 测试

```bash
python3 -m unittest discover tests
```

测试用合成价格路径覆盖：8 条 Trend Template 的通过/失败/未知、Stage 1–4、RS 公式与百分位、EPS/营收加速与扭亏、VCP 识别（标准形态、放量/缩量突破、拉远、失败突破、收缩扩大、底部反弹小回调）以及 CLI 端到端。

> 这是机械筛选加研究笔记，不是投资建议。下单前请自行核对价格和数据。
