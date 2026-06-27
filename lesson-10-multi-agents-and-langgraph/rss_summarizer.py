"""
LangGraph RSS summarizer: fetch feed → download up to 3 articles → summarize each in parallel (map-reduce via Send) → executive summary → HTML report.

Usage:
  Set ANTHROPIC_API_KEY, then:
    python rss_summarizer.py
  You will be prompted to enter an RSS feed URL.
  Example: https://techcrunch.com/feed/
  An HTML report is saved under reports/ when the run completes.
"""

import html
import operator
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()

MAX_ARTICLES = 3
model = None


# --- State ---
class Article(TypedDict):
    title: str
    link: str
    content: str


class ArticleSummary(TypedDict):
    title: str
    link: str
    summary: str


class RSSSummarizerState(TypedDict):
    rss_url: str
    entries: list[Article]
    summaries: Annotated[list[ArticleSummary], operator.add]
    final_summary: str


# --- Helpers: fetch and extract article content ---
def _get_text_from_url(url: str, timeout: int = 10) -> str:
    print(
        f"_get_text_from_url: called: url={url}, timeout={timeout}\n"
        f"_get_text_from_url: [DEBUG] Downloading: {url}"
    )
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "RSS-Summarizer/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in ("script", "style", "nav", "header", "footer"):
            for e in soup.find_all(tag):
                e.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = text[:15000] if len(text) > 15000 else text  # cap length
        print(f"_get_text_from_url: [DEBUG] Downloaded OK: {url} ({len(text)} chars)")
        return text
    except Exception as e:
        print(f"_get_text_from_url: [DEBUG] Error downloading {url}: {e}")
        return f"[Could not fetch URL: {e}]"


# Feed text that is too short or placeholder-like → we fetch the article URL instead
MIN_FEED_CONTENT_LENGTH = 100
TRIVIAL_FEED_CONTENT = frozenset({"comments", "comment", "read more", "read more »", "»"})


def _content_for_entry(entry) -> str:
    print(f"_content_for_entry: called: title={getattr(entry, 'title', None)}, link={getattr(entry, 'link', None)}")
    # Prefer RSS description/summary, then try content, then fetch link.
    # If feed content is trivial (e.g. HN "Comments"), fetch the article URL.
    content = getattr(entry, "summary", None) or getattr(entry, "description", None)
    if content:
        try:
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text(separator="\n", strip=True)
        except Exception:
            pass
        content_stripped = (content or "").strip().lower()
        is_trivial = (
            len(content or "") < MIN_FEED_CONTENT_LENGTH
            or content_stripped in TRIVIAL_FEED_CONTENT
        )
        if content and not is_trivial:
            return content[:15000] if len(content) > 15000 else content
    if getattr(entry, "content", None):
        try:
            raw = entry.content[0].get("value", "")
            if len(raw) >= MIN_FEED_CONTENT_LENGTH:
                return raw[:15000]
        except Exception:
            pass
    link = getattr(entry, "link", None)
    if link:
        return _get_text_from_url(link)
    return "[No content]"


def _build_html_report(
    rss_url: str,
    executive_summary: str,
    article_summaries: list[ArticleSummary],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    article_cards = []
    for i, item in enumerate(article_summaries, 1):
        title = html.escape(item["title"])
        link = html.escape(item["link"], quote=True)
        summary = html.escape(item["summary"])
        title_html = (
            f'<a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a>'
            if item["link"]
            else title
        )
        article_cards.append(
            f"""        <article class="article-card">
          <span class="article-number">Article {i}</span>
          <h2 class="article-title">{title_html}</h2>
          <p class="article-summary">{summary}</p>
        </article>"""
        )

    articles_html = "\n".join(article_cards) if article_cards else (
        '        <p class="empty-state">No article summaries available.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RSS Summary Report</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --surface: #ffffff;
      --text: #1a1f36;
      --muted: #5c6370;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --border: #e5e7eb;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(180deg, #eef2ff 0%, var(--bg) 220px);
      color: var(--text);
      line-height: 1.6;
    }}
    .page {{
      max-width: 820px;
      margin: 0 auto;
      padding: 48px 24px 64px;
    }}
    header {{
      margin-bottom: 32px;
    }}
    .eyebrow {{
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 6px 10px;
      border-radius: 999px;
    }}
    h1 {{
      margin: 16px 0 8px;
      font-size: 2rem;
      line-height: 1.2;
      letter-spacing: -0.02em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .meta a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .meta a:hover {{ text-decoration: underline; }}
    .executive {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      border-radius: 16px;
      padding: 28px 32px;
      box-shadow: var(--shadow);
      margin-bottom: 40px;
    }}
    .executive h2 {{
      margin: 0 0 12px;
      font-size: 1.1rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .executive p {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .articles {{
      display: grid;
      gap: 20px;
    }}
    .article-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px 28px;
      box-shadow: var(--shadow);
    }}
    .article-number {{
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .article-title {{
      margin: 0 0 12px;
      font-size: 1.25rem;
      line-height: 1.35;
    }}
    .article-title a {{
      color: var(--text);
      text-decoration: none;
    }}
    .article-title a:hover {{ color: var(--accent); }}
    .article-summary {{
      margin: 0;
      color: #334155;
    }}
    .empty-state {{
      color: var(--muted);
      font-style: italic;
    }}
    footer {{
      margin-top: 40px;
      text-align: center;
      color: var(--muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <span class="eyebrow">RSS Summary Report</span>
      <h1>News Digest</h1>
      <p class="meta">
        Source: <a href="{html.escape(rss_url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(rss_url)}</a><br>
        Generated: {generated_at}
      </p>
    </header>

    <section class="executive">
      <h2>Executive Summary</h2>
      <p>{html.escape(executive_summary)}</p>
    </section>

    <section class="articles">
{articles_html}
    </section>

    <footer>
      Generated by RSS Summarizer
    </footer>
  </div>
</body>
</html>
"""


def _save_html_report(rss_url: str, executive_summary: str, article_summaries: list[ArticleSummary]) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"rss_report_{timestamp}.html"
    report_path.write_text(
        _build_html_report(rss_url, executive_summary, article_summaries),
        encoding="utf-8",
    )
    return str(report_path.resolve())


# --- Graph nodes ---
def fetch_rss(state: RSSSummarizerState) -> dict:
    print(
        "--------------------------------\n"
        f"FETCH_RSS: called: rss_url={state.get('rss_url')}\n"
        "--------------------------------"
    )
    url = state["rss_url"]
    print(f"fetch_rss: [DEBUG] Fetching RSS: {url}")
    feed = feedparser.parse(url)
    if feed.bozo and not getattr(feed, "entries", None):
        print(f"fetch_rss: [DEBUG] Error parsing feed: {feed.bozo_exception}")
        return {"entries": [], "final_summary": f"Error parsing feed: {feed.bozo_exception}"}
    entries_raw = feed.entries[:MAX_ARTICLES]
    entries: list[Article] = []
    for e in entries_raw:
        title = getattr(e, "title", None) or "Untitled"
        link = getattr(e, "link", None) or ""
        content = _content_for_entry(e)
        article = {"title": title, "link": link, "content": content}
        entries.append(article)
        # Debug: print downloaded article
        snippet = content[:400] + "..." if len(content) > 400 else content
        print(
            f"fetch_rss: [DEBUG] Article: {title}\n"
            f"fetch_rss: [DEBUG]   Link: {link}\n"
            f"fetch_rss: [DEBUG]   Content ({len(content)} chars): {snippet}"
        )
    return {"entries": entries}


def _summarize_one_article(article: Article) -> str:
    content_snippet = article["content"][:12000]
    prompt = f"""Summarize the following article in 2–4 concise sentences. Answer only with the summary, no prefix or suffix. Focus on the main point and key takeaways. The text below is the full article content (already fetched); do not ask for links or external access.

Title: {article['title']}

Article content:
{content_snippet}
"""
    response = model.invoke(
        [
            SystemMessage(content="You are a concise summarizer. Summarize only the given text. Output only the summary, no preamble or refusals."),
            HumanMessage(content=prompt),
        ]
    )
    summary = response.content if hasattr(response, "content") else str(response)
    # print(f"_summarize_one_article: [DEBUG] Summary for '{article['title']}': {summary}")
    return summary


def summarize_one(state: dict) -> dict:
    """Node run in parallel via Send(); state holds a single 'article'."""
    print(
        "--------------------------------\n"
        f"SUMMARIZE_ONE: called: article={state.get('article', {}).get('title')}, model={type(model).__name__}\n"
        "--------------------------------"
    )
    article = state["article"]
    summary = _summarize_one_article(article)
    return {
        "summaries": [
            {
                "title": article["title"],
                "link": article["link"],
                "summary": summary,
            }
        ]
    }


def route_after_fetch(state: RSSSummarizerState):
    """Map-reduce: send one task per article to summarize_one, or go to final_summary if none."""
    print(
        "--------------------------------\n"
        f"ROUTE_AFTER_FETCH: called: entries_count={len(state.get('entries', []))}\n"
        "--------------------------------"
    )
    if not state["entries"]:
        return [Send("final_summary", state)]
    return [Send("summarize_one", {"article": e}) for e in state["entries"]]


def final_summary_node(state: RSSSummarizerState) -> dict:
    print(
        "--------------------------------\n"
        f"FINAL_SUMMARY_NODE: called: summaries_count={len(state.get('summaries', []))}, model={type(model).__name__}\n"
        "--------------------------------"
    )
    # State may be partial when coming from summarize_one path (Send only passed {"article": e}),
    # so we only rely on "summaries" and use .get for anything else.
    summaries = state.get("summaries", [])
    if not summaries:
        return {"final_summary": "No articles to summarize."}

    combined = "\n\n---\n\n".join(
        f"Article {i + 1} ({s['title']}):\n{s['summary']}" for i, s in enumerate(summaries)
    )
    prompt = f"""Based on the following per-article summaries, write one short executive summary (one short paragraph) that captures the main themes and highlights across all articles.

{combined}
"""
    response = model.invoke(
        [
            SystemMessage(content="You are a concise editor. Output only the executive summary paragraph."),
            HumanMessage(content=prompt),
        ]
    )
    executive_summary = response.content if hasattr(response, "content") else str(response)
    return {"final_summary": executive_summary}


# --- Model ---
def get_model():
    print(
        "--------------------------------\n"
        "GET_MODEL: called:\n"
        "--------------------------------"
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("get_model: Error: Set ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)
    return init_chat_model(
        "anthropic:claude-haiku-4-5-20251001",
        temperature=0,
        api_key=api_key,
    )


def main():
    global model
    print("main: called:\nmain: Example: https://techcrunch.com/feed/")
    try:
        rss_url = input("RSS feed URL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("main:")
        sys.exit(0)
    if not rss_url:
        print("main: Error: RSS feed URL is required.")
        sys.exit(1)
    model = get_model()

    # build graph
    builder = StateGraph(RSSSummarizerState)
    # add nodes
    builder.add_node("fetch_rss", fetch_rss)
    builder.add_node("summarize_one", summarize_one)
    builder.add_node("final_summary", final_summary_node)
    # add edges
    builder.add_edge(START, "fetch_rss")
    builder.add_conditional_edges("fetch_rss", route_after_fetch)
    builder.add_edge("summarize_one", "final_summary")
    builder.add_edge("final_summary", END)

    # compile graph
    graph = builder.compile()
    # create initial state
    initial: RSSSummarizerState = {
        "rss_url": rss_url,
        "entries": [],
        "summaries": [],
        "final_summary": "",
    }
    # invoke graph
    result = graph.invoke(initial)

    report_path = _save_html_report(
        rss_url,
        result["final_summary"],
        result.get("summaries", []),
    )
    print(f"\nReport saved to: {report_path}")
    print(f"\nExecutive summary:\n{result['final_summary']}")


if __name__ == "__main__":
    main()
