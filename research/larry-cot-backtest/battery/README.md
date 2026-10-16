# Larry Williams 基本面工具组合测试（第二阶段，spec v2）

第一阶段（上一级目录，[../README.md](../README.md)）只测了 COT 商业持仓指数。
这一阶段把 Larry Williams 的其他基本面工具**逐个单独回测**：升贴水（premium）、季节性（seasonality）、
估值指数（valuation）、持仓量（open interest）、跨市场（intermarket），再加上第一阶段的 COT，
六个工具用**同一套不偷看未来的评估方法**（`harness/fund_harness.py`）比较、排名，最后再测"多个工具同时看多"（共振，confluence）。
每个工具在跑数据之前都先写好 `PREREG.md`（预先登记：唯一的主规则 + 变体网格），结果以主规则为准，不挑最好的网格。

## 结论

**六个工具里，没有一个在样本外、扣除交易成本后有稳健的正期望值。**

排名：**升贴水 > 季节性 > 持仓量 > 跨市场 > 估值 > COT**（`rank/PREREG.md` 第 7 节的预定评分规则）。

| 排名 | 工具 | 主规则（全部持有 8 周） | 超额收益 | 置换检验 p | 判定 |
|---|---|---|---|---|---|
| 1 | 升贴水 | 原油近月 > 次月（现货升水）→ 做多，展期调整后的原油指数 | **+1.60%** | 0.065 | 有正期望值，但很脆弱 |
| 2 | 季节性 | 7 个市场合并，按历年同期走势做多/做空 | +1.92% | **0.004** | 统计上显著，但不能交易 |
| 3 | 持仓量 | 原油横盘时持仓量大降 → 做多 | +0.88% | 0.36 | 没有优势 |
| 4 | 跨市场 | 美元 8 周下跌 → 做多黄金 | −0.07% | 0.53 | 没有优势 |
| 5 | 估值 | 原油相对黄金被低估 → 做多 | −1.59% | 0.71 | 没有优势 |
| 6 | COT | 原油商业持仓指数（26 周）≥ 80 → 做多 | −3.21% | 0.96 | 没有优势 |

- **升贴水（第 1 名）**：超额 +1.60% / 8 周，p = 0.065，不显著。赚的是**展期收益（carry）**，不是价格上涨：
  同一信号放在现货上是 −1.38%。前半段 +1.58%，后半段只剩 +0.08%；**2010 年以后为负**（−1.54%）。
- **季节性（第 2 名）**：超额 +1.92%，p = 0.004，前后两半都为正；但一半以上（56%）来自无法交易的纽约港汽油现货，
  在可交易的期货展期指数上是 **−0.26%**（p = 0.68）。
- **多工具共振**：原油上至少 2 个工具同时看多（且多于看空）就做多，超额 **+0.13%**，p = 0.45，没有胜过任何单个工具。
- **多重检验**：五个工具的全部检验加上 COT 的 360 个组合，共 **12,919 个带 p 值的检验，Benjamini–Hochberg（q = 0.10）校正后 0 个显著**。

数字全部来自 `ROOT/work2/rank/results.json` 和 `ROOT/work2/rank/primary_table.csv`（下面的 `run_battery.sh` 会重新生成）。
"超额" = 信号周的平均 8 周收益 − 同一市场所有周的平均收益。**本项目仅供研究，不构成任何投资建议。**

## 如何运行

需要：bash、较新的 git（需支持 `--filter=blob:none`，已用 2.43 测试）、Python ≥ 3.11（依赖同 [../requirements.txt](../requirements.txt)），
能访问 github.com 和 raw.githubusercontent.com。

```bash
pip install -r requirements.txt          # 在包目录（battery/ 的上一级）运行
battery/run_battery.sh                   # 下载数据 → 自检 → 数据整理 → 五个工具 → 排名 → 共振 → 终审
battery/run_battery.sh --skip-fetch      # 数据已下载过，跳过下载
LARRY_COT_ROOT=/某个目录 battery/run_battery.sh   # 数据和输出放到别的目录
```

根目录 `ROOT` 和第一阶段相同：环境变量 `LARRY_COT_ROOT`，不设就用包目录（`battery/` 的上一级）。

```
ROOT/data/、ROOT/ext/   第一阶段的输入（tools/fetch_data.py）      ROOT/work/    第一阶段的输出
ROOT/ext2/              本阶段新增的两个第三方仓库                  ROOT/work2/   本阶段全部输出（每步一个子目录）和 logs/
```

运行顺序（`run_battery.sh`，与原始运行顺序一致）：

1. 缺数据时先下载：`tools/fetch_data.py`（第一阶段输入）和 `tools/fetch_data_battery.py`（`ROOT/ext2`）；
   缺第一阶段的输出（`work/engine/*.csv`、`work/lens_strategy/metrics.csv`）时，先跑第一阶段的引擎和策略模拟。
2. `harness/selftest_harness.py`：共用评估框架自检（逐位复现第一阶段的 COT 结果）。
3. `scout/scout_check.py`：生成 `curve_daily.csv`（近月/远月合约价格）和 `cot_smalltrader_weekly.csv`（小散户持仓）。
4. 五个工具，各自先跑预先登记的主程序，再跑独立复核：`tool_premium/` → `tool_seasonality/` → `tool_valuation/` →
   `tool_open_interest/` → `tool_intermarket/`。
5. `rank/verify_rebuild.py`（逐一重建每个工具的主结果）→ `rank/run_confluence.py`（共振）→ `rank/run_rank.py`（排名、多重检验）。
6. `audit_final/` 四个终审脚本（读取排名表），然后再跑一次 `rank/run_rank.py`，把终审结果写进 `results.json`。

全程约 15 分钟（不含下载；季节性一步约 6 分钟）。各步日志在 `ROOT/work2/logs/`。

### 新增数据（`tools/fetch_data_battery.py`：浅克隆 + 稀疏检出，只取用到的文件，并校验 SHA-256）

| 来源 | 版本 | 用到的文件 |
|---|---|---|
| github.com/robcarver17/pysystemtrade（GPL-3） | commit `bbe29e19dab73271570c94b4ee7e2f182d4b9ef1` | `data/futures/multiple_prices_csv/` 下 11 个品种（CRUDE_W、GOLD、SILVER、SOYBEAN、SUGAR11、CORN、WHEAT、COPPER、GAS_US、HEATOIL、GASOILINE）；另取 `csvconfig/rollconfig.csv` 和 `LICENSE` |
| github.com/bgithub1/cme_open_interest | commit `a5772c97fcb58f06ed8c945fcfec1b7e6ddc5263` | `cot_history.csv`（CFTC 传统报告，含非报告头寸）、`cot_new_history.csv` |

pysystemtrade 仓库很大，所以只做 `--depth 1 --filter=blob:none` 的浅获取，再用稀疏检出只取上表中的文件。
另外还读取第一阶段已下载的 `ext/accomplish999_crude-oil` 里的 EIA / FRED 序列（原油 1–4 号合约、取暖油、汽油、美元指数、10 年期国债收益率等）。
下载的内容一律当作不可信数据：只解析 CSV/JSON，从不执行；Python 一律用 `python3 -I` 运行。数据和结果都不提交（`ext2/`、`work2/` 在 `.gitignore` 里）。

### 代码结构

| 目录 | 内容 |
|---|---|
| `SPEC2.md` | 本阶段的共用规格（评估规则 1–8） |
| `harness/` | `fund_harness.py` 共用评估框架（数据序列、入场时间、远期收益、置换检验、样本外、S1/S2 策略），`selftest_harness.py` 自检 |
| `scout/` | `scout_check.py` 数据检查与整理；`INVENTORY.md` 数据清单（找到了什么、没找到什么） |
| `tool_*/` | 每个工具：`PREREG.md` + 哈希、`*_lib.py` 信号、`run_*.py` 主程序、`verify*`/`adv_verify*`/`audit*` 独立复核 |
| `rank/` | `PREREG.md`（排名与共振的预定规则）、`rank_lib.py`、`verify_rebuild.py`、`run_confluence.py`、`run_rank.py` |
| `audit_final/` | 终审：升贴水独立重建、升贴水多空交易拆解、全族多重检验复核、共振的未来数据检查 |

### 局限

- pysystemtrade 和 EIA 合约数据都只到 **2024-03/04**，所以升贴水、共振等无法检验 2024 年以后。
- pysystemtrade 的部分品种持有的不是近月合约（如原油 11 月/12 月、大豆 11 月），从 2008–2013 年起（因品种而异）部分价格是小时数据当天的最后一笔，不一定是结算价。
- 小散户持仓（CFTC 传统报告）只有 2000–2020 年，且缺 2007 年。
- 排名规则（`rank/PREREG.md` 第 7 节）是在各工具结果出来以后才写的，没有为结果调参，但不是盲测。
