---
name: techcrunch-deck
description: "Use this skill to build a PowerPoint (.pptx) presentation from the latest trending or most popular TechCrunch articles. Trigger whenever the user asks for a slide deck, presentation, or summary deck based on TechCrunch news, tech headlines, or 'latest tech articles'. Fetches the top 5 trending/popular TechCrunch stories, summarizes each, extracts key takeaways, and produces a designed 7-slide deck by delegating the actual .pptx generation to the pptx skill."
license: Proprietary. See the pptx skill LICENSE.txt for deck-generation terms.
---

# TechCrunch Deck Skill

Build a polished 7-slide presentation from the 5 latest trending/most popular
TechCrunch articles. This skill orchestrates the work; it **delegates the actual
.pptx creation to the [pptx](../pptx/SKILL.md) skill** — always read and follow
that skill for the generation, QA, and rendering steps.

## Workflow

### 1. Fetch the 5 articles

Pull the 5 latest **trending / most popular** TechCrunch stories. Preferred sources
(try in order until you have 5 distinct, recent articles):

- RSS feed: `https://techcrunch.com/feed/` (latest posts)
- Homepage / popular: `https://techcrunch.com/` and `https://techcrunch.com/latest/`
- If "most popular" ranking is unavailable, use the most recent stories and note
  that ranking is by recency.

Use WebFetch (or WebSearch as a fallback) to retrieve each article page, then for
**each** article capture:

- `title`
- `url`
- `author` and `published` date (if available)
- `summary` — 2-3 sentence neutral summary in your own words
- `takeaways` — 2-4 concise, concrete key takeaways (bullet points)

Optionally run `scripts/fetch_articles.py` to grab candidate URLs from the RSS feed,
then fetch each with WebFetch. Do **not** copy article text verbatim — summarize.

De-duplicate (same story from different URLs) and keep exactly 5.

### 2. Define the deck structure (7 slides)

| # | Slide | Content |
|---|-------|---------|
| 1 | **Overview / Summary** | Title + a one-line summary of each of the 5 articles (numbered list or 5 cards) |
| 2-6 | **One per article** | Article title, source/date, 2-3 sentence summary, and a "Key Takeaways" list |
| 7 | **Key Takeaways (all)** | The single most important cross-cutting takeaway per article + one overall conclusion |

Slide 1 is the at-a-glance index. Slides 2-6 go one article per page. Slide 7
synthesizes across all five (themes, what it means, what to watch).

### 3. Generate the .pptx

Follow the **pptx** skill's "Creating from Scratch" path (read
[../pptx/pptxgenjs.md](../pptx/pptxgenjs.md)). Apply its design guidance:

- Pick a bold, tech-news-appropriate palette (e.g. TechCrunch green `00D103` /
  `108A00` as accent on a dark or charcoal base) — do not default to plain blue.
- Every slide needs a visual element (icon, shape, card, or accent block).
- Dark title + conclusion slides, lighter content slides ("sandwich").
- Vary layouts across the 5 article slides (don't repeat identical bullet lists).
- Include each article's `url` as a small muted caption/source line.

Suggested filename: `techcrunch-trending-<YYYY-MM-DD>.pptx`.

### 4. QA (required)

Do **not** skip this — it is part of the pptx skill:

1. Content QA: `python -m markitdown output.pptx` — verify all 5 titles, summaries,
   takeaways, and correct order are present; no placeholder text.
2. Visual QA: convert to images and inspect with a **subagent** (see the pptx skill's
   QA section) for overflow, overlap, contrast, and alignment issues.
3. Fix and re-verify at least once before declaring done.

### 5. Report

Tell the user the output path and list the 5 article titles + source URLs used, so
the selection is transparent and verifiable.

## Notes

- If live fetching is unavailable (offline/sandbox), tell the user rather than
  inventing articles. Never fabricate headlines, quotes, or stats.
- Keep summaries in your own words; cite the source URL for attribution.
