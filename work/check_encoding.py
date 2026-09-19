# -*- coding: utf-8 -*-
"""推送前检查: 找出「不是 UTF-8」的文本文件。

GitHub 按 UTF-8 渲染, 中文文件如果是 GBK 编码, 推上去就是乱码。
只扫将要提交的范围(排除 lib/data/缓存目录)。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"lib", "data", "__pycache__", ".git", "pylibs", "foreign", "multi_data"}
EXTS = {".py", ".bat", ".md", ".txt", ".json", ".csv", ".html", ".js", ".css", ".cfg", ".toml"}


def walk():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def classify(p):
    raw = p.read_bytes()
    try:
        raw.decode("utf-8")
        return "utf8", None
    except UnicodeDecodeError:
        pass
    for enc in ("gb18030", "gbk", "big5", "latin-1"):
        try:
            raw.decode(enc)
            return enc, None
        except UnicodeDecodeError:
            continue
    return "?", None


bad = []
total = 0
for p in walk():
    total += 1
    enc, _ = classify(p)
    if enc != "utf8":
        bad.append((p.relative_to(ROOT), enc, len(p.read_bytes())))

print(f"扫描 {total} 个文本文件")
if not bad:
    print("全部是 UTF-8, 可以推。")
else:
    print(f"\n不是 UTF-8 的 {len(bad)} 个:")
    for rel, enc, n in sorted(bad, key=lambda x: -x[2]):
        print(f"  {str(rel):<50} {enc:<8} {n:>8} 字节")
    print("\n用 --fix 参数可以把它们就地转成 UTF-8(会先备份成 .enc-bak)")

if "--fix" in sys.argv:
    n = 0
    for rel, enc, _ in bad:
        p = ROOT / rel
        try:
            text = p.read_bytes().decode(enc)
        except Exception as e:
            print(f"  跳过 {rel}: {e}")
            continue
        p.with_suffix(p.suffix + ".enc-bak").write_bytes(p.read_bytes())
        p.write_text(text, encoding="utf-8", newline="")
        n += 1
    print(f"\n已转换 {n} 个文件为 UTF-8(原文件备份为 *.enc-bak)")
