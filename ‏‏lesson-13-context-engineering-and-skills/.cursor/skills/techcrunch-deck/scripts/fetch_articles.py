#!/usr/bin/env python3
"""Fetch candidate TechCrunch article URLs from the public RSS feed.

Prints the N most recent items as JSON: [{title, url, published}, ...].
This only gathers candidate links + titles. Summaries and key takeaways
should be produced by reading each article (e.g. via WebFetch) and
summarizing in your own words — do NOT copy article text verbatim.

Usage:
    python fetch_articles.py [count]   # default count = 5

Stdlib only (urllib + xml). No third-party dependencies.
"""
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET

FEED_URL = "https://techcrunch.com/feed/"
USER_AGENT = "Mozilla/5.0 (compatible; techcrunch-deck-skill/1.0)"


def fetch_feed(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def parse_items(xml_bytes: bytes, count: int):
    root = ET.fromstring(xml_bytes)
    items = []
    # RSS 2.0: channel/item with title, link, pubDate
    for item in root.iterfind(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title and link:
            items.append({"title": title, "url": link, "published": pub})
        if len(items) >= count:
            break
    return items


def main() -> int:
    count = 5
    if len(sys.argv) > 1:
        try:
            count = max(1, int(sys.argv[1]))
        except ValueError:
            print(f"Invalid count: {sys.argv[1]!r}", file=sys.stderr)
            return 2
    try:
        xml_bytes = fetch_feed(FEED_URL)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to fetch {FEED_URL}: {exc}", file=sys.stderr)
        print("Fetching unavailable — fall back to WebFetch/WebSearch and do "
              "not fabricate articles.", file=sys.stderr)
        return 1
    items = parse_items(xml_bytes, count)
    if not items:
        print("No items parsed from feed.", file=sys.stderr)
        return 1
    print(json.dumps(items, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
