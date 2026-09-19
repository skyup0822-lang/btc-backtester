#!/usr/bin/env python3
"""
chrome_render.py -- render a URL with the user's real Google Chrome via Scrapling.

Use when plain HTTP is not enough (JS-heavy pages) or a site needs a real browser.
Note: this renders through whatever network path Chrome uses, so it CANNOT bypass a
network-level block (e.g. the HK government domains blocked in this sandbox all fail
with ERR_CONNECTION_CLOSED even in Chrome).

Usage:
    python chrome_render.py <url>              # print text (first 3000 chars)
    python chrome_render.py <url> out.md       # save as Markdown
    python chrome_render.py <url> out.html     # save raw HTML
"""
import sys

from scrapling.fetchers import DynamicFetcher

CHROME = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"


def render(url, timeout=60000, wait=3000):
    return DynamicFetcher.fetch(url, headless=True, network_idle=True,
                                executable_path=CHROME, timeout=timeout, wait=wait)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    url, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    page = render(url)
    if out and out.endswith(".html"):
        body = page.body if isinstance(page.body, str) else page.body.decode("utf-8", "replace")
        open(out, "w", encoding="utf-8").write(body)
        print("status=%s wrote %s (%d chars)" % (page.status, out, len(body)))
    elif out:
        md = page.markdown()
        open(out, "w", encoding="utf-8").write(md)
        print("status=%s wrote %s (%d chars)" % (page.status, out, len(md)))
    else:
        txt = " ".join(page.css("::text").getall())
        print("status=%s\n%s" % (page.status, txt[:3000]))
