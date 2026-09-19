# -*- coding: utf-8 -*-
"""修正模拟盘账本：把被错误记账的 cash 还原, 并重算每日权益。

依据: PaperTrader.buy() 里 entry_equity = 买入前的现金, 买入后 cash = entry_equity - notional,
      其中 notional = units * entry_price / (1 - fee_rate)。
所以正确的 cash = entry_equity - units*entry_price/(1-fee)。
每日权益 equity = cash + units*btc_price (与 web_paper._record_daily 同口径)。

用法: python work/fix_paper_account.py            # 只看不写(dry-run)
      python work/fix_paper_account.py --apply    # 实际写入
"""
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "paper_state_okx.json"
DAILY = ROOT / "daily_report.csv"
FEE = 0.001
INITIAL = 100.0


def main():
    apply = "--apply" in sys.argv
    st = json.loads(STATE.read_text(encoding="utf-8"))
    units = st["units"]
    entry = st["entry_price"]
    entry_equity = st["entry_equity"]
    cash_wrong = st["cash"]
    notional = units * entry / (1.0 - FEE) if units > 0 else 0.0
    cash_ok = entry_equity - notional

    print("=" * 70)
    print("【状态修正】paper_state_okx.json")
    print(f"  持仓 units={units:.10f} | 入场={entry:.2f} | entry_equity={entry_equity:.4f}")
    print(f"  推算 notional = {notional:.4f}")
    print(f"  cash 现状 = {cash_wrong:.6f}  (异常, 比正确值多 {cash_wrong - cash_ok:+.6f})")
    print(f"  cash 正确 = {cash_ok:.6f}")
    if units > 0:
        print(f"  每日权益应为 cash + units*price; 用现价 78683 估: {cash_ok + units * 78683:.4f}")

    # 修正每日报表
    rows = list(csv.DictReader(DAILY.open(encoding="utf-8")))
    rate = 6.728  # 用原报表隐含汇率
    fixed = []
    prev_eq = INITIAL
    print("\n【每日报表重算】(equity = 正确cash + units*btc_price)")
    for r in rows:
        eq = cash_ok + float(r["units"]) * float(r["btc_price"])
        pnl = eq - prev_eq
        r2 = {
            "date": r["date"], "equity_usdt": f"{eq:.2f}", "pnl_usdt": f"{pnl:.2f}",
            "equity_cny": f"{eq * rate:.2f}", "pnl_cny": f"{pnl * rate:.2f}",
            "btc_price": r["btc_price"], "units": r["units"],
        }
        print(f"  {r['date']}: equity {r['equity_usdt']} -> {r2['equity_usdt']} | "
              f"pnl {r['pnl_usdt']} -> {r2['pnl_usdt']}")
        fixed.append(r2)
        prev_eq = eq

    if not apply:
        print("\n(dry-run, 未写入。加 --apply 实际写入)")
        return

    # 备份
    STATE.with_suffix(".json.bak-fix").write_text(STATE.read_text(encoding="utf-8"), encoding="utf-8")
    DAILY.with_suffix(".csv.bak-fix").write_text(DAILY.read_text(encoding="utf-8"), encoding="utf-8")

    st["cash"] = cash_ok
    st.setdefault("high_watermark", entry)  # 新代码需要
    STATE.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["date", "equity_usdt", "pnl_usdt",
                                        "equity_cny", "pnl_cny", "btc_price", "units"])
    w.writeheader()
    for r in fixed:
        w.writerow(r)
    DAILY.write_text(buf.getvalue(), encoding="utf-8", newline="")
    print("\n已写入(原文件备份为 .bak-fix)")


if __name__ == "__main__":
    main()
