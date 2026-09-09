"""全局配置:手续费、滑点、数据路径、样本切分等。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "btc_usdt_1h.csv"
RESULTS_DIR = ROOT / "results"

# 样本切分:该时间之前为训练期(样本内),之后为测试期(样本外)。
TRAIN_END = "2024-12-31"

# 1 小时 K 线的一年根数,用于年化计算
PERIODS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0  # 初始资金(USDT)
    fee_rate: float = 0.001            # 单边手续费 0.1%(现货常见费率)
    slippage: float = 0.0005           # 单边滑点 0.05%(市价单成交偏离)


DEFAULT_CONFIG = BacktestConfig()