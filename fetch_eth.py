# -*- coding: utf-8 -*-
"""下载 ETH/USDT 1 小时 K 线历史数据,复用 fetch_data.py 的多源下载逻辑。

仅使用 Binance 数据源(官方存档 + 公开接口);禁用 OKX/Gate.io 备选源,
因为那两个函数里写死了 BTC 交易对,拿到的数据会是错的。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fetch_data as fd

fd.SYMBOL = "ETHUSDT"
fd.DATA_FILE = ROOT / "data" / "eth_usdt_1h.csv"


def _disabled(*args, **kwargs):
    raise RuntimeError("OKX/Gate.io 备选源未适配 ETH,已禁用")


fd.fetch_okx = _disabled
fd.fetch_gateio = _disabled

if __name__ == "__main__":
    fd.main()