"""
Shared news classification logic.

Classifies crypto news articles as bull/bear/not_correlated with strength
using Azure OpenAI (gpt-4o-mini) structured output.

Used by:
- news_collector.py — classify articles at collection time
- agent_for_news_analysis.py — fallback for unclassified articles
"""

import json
from typing import Literal, cast

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# -- Pydantic models ----------------------------------------------------------

class NewsItem(BaseModel):
    article_id: str = Field(description="Stable article identifier from input payload")
    title: str
    category: Literal["not_correlated", "bear", "bull"]
    strength: Literal["low", "medium", "high"] = Field(
        description="How strongly this news impacts BTC price. "
        "Only meaningful when category is bear or bull."
    )


class NewsClassificationResponse(BaseModel):
    classifications: list[NewsItem]


# -- Constants ----------------------------------------------------------------

CLASSIFIER_PROMPT = """\
You are a crypto-news classifier. For each news article, determine its likely \
impact on the BTCUSDT price.

Categories:
- "not_correlated": news about altcoins, NFTs, specific DeFi protocols, or events \
with no meaningful impact on BTC price.
- "bull": news that is likely POSITIVE for BTC price — institutional adoption, \
ETF approvals, favorable regulation, dovish Fed signals, major partnerships, \
positive on-chain metrics, supply shocks, etc.
- "bear": news that is likely NEGATIVE for BTC price — exchange hacks, \
regulatory crackdowns, hawkish Fed signals, large sell-offs, negative macro data, \
security breaches, bans, bankruptcies, etc.

For each article also estimate the STRENGTH of its impact on BTC price:
- "low": minor or indirect influence, unlikely to move price significantly.
- "medium": notable event that could contribute to a price move.
- "high": major catalyst that can directly drive significant price movement \
(e.g. ETF decision, Fed rate change, large-scale hack, institutional buy).

Return a classification for EVERY article provided.
You MUST copy both fields exactly:
- "article_id" from the input payload
- "title" from the input payload
"""

STRENGTH_WEIGHTS = {"low": 1, "medium": 2, "high": 3}

LOG_TAG = "[news_classifier]"


# -- Helpers ------------------------------------------------------------------

def _choose_batch_size(total_articles: int) -> int:
    """Smaller batches improve LLM classification accuracy;
    single call avoids overhead for small sets."""
    if total_articles < 15:
        return total_articles
    if total_articles <= 30:
        return 20
    return 30


def _prepare_for_classification(articles: list[dict]) -> list[dict]:
    """Format archive articles into LLM input format.

    Each article gets an article_id based on its index and
    content truncated to 500 chars.
    """
    from agents.news_analyser.helpers import parse_release_time, strip_html

    prepared = []
    for idx, item in enumerate(articles):
        dt = parse_release_time(item.get("article_release_time"))
        date_str = dt.strftime("%Y-%m-%d %H:%M") if dt else item.get("date", "—")
        raw_content = item.get("article_content", "")
        # Content may already be stripped (from archive) or raw HTML (from API)
        content = strip_html(raw_content) if "<" in raw_content else raw_content
        prepared.append({
            "article_id": f"news_{idx}",
            "date": date_str,
            "title": item.get("article_title", "—"),
            "source": item.get("source_name", "—"),
            "content": content[:500],
        })
    return prepared


def classify_articles(articles: list[dict]) -> None:
    """Classify articles in-place, adding 'category' and 'strength' fields.

    Args:
        articles: list of archive-format article dicts.
                  Modified in-place: each dict gets 'category' and 'strength' keys.

    On per-batch failure: marks articles as category='unclassified', strength=None.
    """
    if not articles:
        return

    total = len(articles)
    batch_size = _choose_batch_size(total)
    effective_batch_size = max(batch_size, 1)

    # Prepare LLM input format
    prepared = _prepare_for_classification(articles)

    # Build index mapping: article_id -> original article dict
    id_to_article = {prepared[i]["article_id"]: articles[i] for i in range(total)}
    title_to_article = {articles[i].get("article_title", ""): articles[i] for i in range(total)}

    classifier_llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0.0)

    for batch_start in range(0, total, effective_batch_size):
        batch_prepared = prepared[batch_start:batch_start + effective_batch_size]
        batch_articles = articles[batch_start:batch_start + effective_batch_size]
        batch_number = batch_start // effective_batch_size + 1
        print(f"{LOG_TAG} Batch {batch_number}: {len(batch_prepared)} articles")

        try:
            news_json = json.dumps(batch_prepared, ensure_ascii=False)
            classification_result = cast(
                NewsClassificationResponse,
                classifier_llm.with_structured_output(NewsClassificationResponse).invoke([
                    SystemMessage(content=CLASSIFIER_PROMPT),
                    HumanMessage(content=(
                        f"Classify these {len(batch_prepared)} articles.\n"
                        f"Return fields article_id, title, category, strength for each item:\n{news_json}"
                    )),
                ])
            )

            # Match classifications back to original articles
            classifications_by_id = {item.article_id: item for item in classification_result.classifications}
            classifications_by_title = {item.title: item for item in classification_result.classifications}

            for prep_item, orig_article in zip(batch_prepared, batch_articles):
                matched = classifications_by_id.get(prep_item["article_id"])
                if matched is None:
                    matched = classifications_by_title.get(prep_item["title"])
                if matched is not None:
                    orig_article["category"] = matched.category
                    orig_article["strength"] = matched.strength
                else:
                    orig_article["category"] = "not_correlated"
                    orig_article["strength"] = "low"

        except Exception as e:
            print(f"{LOG_TAG} ERROR in batch {batch_number}: {e}")
            # Mark entire batch as unclassified on failure
            for article in batch_articles:
                if "category" not in article:
                    article["category"] = "unclassified"
                    article["strength"] = None
