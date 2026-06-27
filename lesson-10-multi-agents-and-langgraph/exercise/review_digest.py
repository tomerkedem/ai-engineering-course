import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict
from urllib import response

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


load_dotenv()

REVIEWS_FILE = Path("reviews.json")
REPORTS_DIR = Path("reports")


class Review(TypedDict):
    review_id: str
    user_name: str
    rating: int
    review_text: str
    timestamp: str


class ReviewAnalysis(TypedDict):
    review_id: str
    sentiment: str
    pros: list[str]
    cons: list[str]


class ReviewDigestState(TypedDict):
    reviews: list[Review]
    review_analyses: Annotated[list[ReviewAnalysis], operator.add]
    final_report: str
    output_path: str


def get_model():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY in your environment.")
        sys.exit(1)

    return init_chat_model(
        "anthropic:claude-haiku-4-5-20251001",
        temperature=0,
        api_key=api_key,
    )


model = get_model()


def load_reviews_from_file(path: Path = REVIEWS_FILE) -> list[Review]:
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("reviews.json must contain a list of reviews.")

    return data


def fetch_reviews(state: ReviewDigestState) -> dict:
    reviews = load_reviews_from_file()
    print(f"Loaded {len(reviews)} reviews from {REVIEWS_FILE}")
    return {"reviews": reviews}


def route_reviews(state: ReviewDigestState):
    reviews = state.get("reviews", [])

    return [
        Send("analyze_single_review", {"review": review})
        for review in reviews
    ]


def analyze_single_review(state: dict) -> dict:
    review: Review = state["review"]

    prompt = f"""Analyze the following product review.

Return valid JSON only, with this exact structure:
{{
  "sentiment": "Positive | Negative | Neutral | Mixed",
  "pros": ["short advantage 1", "short advantage 2"],
  "cons": ["short disadvantage 1", "short disadvantage 2"]
}}

Review ID: {review["review_id"]}
Rating: {review["rating"]}
Review text:
{review["review_text"]}
"""

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a careful product review analyst. "
                    "Extract only information that is clearly supported by the review. "
                    "Return valid JSON only."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    print("RAW MODEL RESPONSE:")
    print(repr(response.content))

    content = response.content.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```").strip()
    elif content.startswith("```"):
        content = content.removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(content)

    analysis: ReviewAnalysis = {
        "review_id": review["review_id"],
        "sentiment": parsed.get("sentiment", "Neutral"),
        "pros": parsed.get("pros", []),
        "cons": parsed.get("cons", []),
    }

    return {"review_analyses": [analysis]}


def build_final_report(state: ReviewDigestState) -> dict:
    analyses = state.get("review_analyses", [])

    if not analyses:
        return {
            "final_report": (
                "# Product Buyer Advisory Report\n\n"
                "No reviews were available for analysis."
            )
        }

    analyses_json = json.dumps(analyses, ensure_ascii=False, indent=2)

    prompt = f"""You are given structured analyses of customer reviews for one product.

Each analysis includes:
- sentiment
- pros
- cons

Create a Product Buyer Advisory Report in Markdown.

The report must include:
1. Overall recommendation
2. Best fit: who should buy this product
3. Main strengths
4. Main concerns
5. Skip or be careful if
6. Short final verdict

Use only the information in the analyses.
Do not invent product details that are not supported by the reviews.

Review analyses:
{analyses_json}
"""

    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a careful product advisor. "
                    "Base your report only on the provided review analyses."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    return {"final_report": response.content}


def save_output(state: ReviewDigestState) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = REPORTS_DIR / f"product_buyer_report_{timestamp}.md"

    output_path.write_text(state["final_report"], encoding="utf-8")

    print(f"Report saved to: {output_path}")

    return {"output_path": str(output_path)}


def build_graph():
    builder = StateGraph(ReviewDigestState)

    builder.add_node("fetch_reviews", fetch_reviews)
    builder.add_node("analyze_single_review", analyze_single_review)
    builder.add_node("build_final_report", build_final_report)
    builder.add_node("save_output", save_output)

    builder.add_edge(START, "fetch_reviews")
    builder.add_conditional_edges("fetch_reviews", route_reviews)
    builder.add_edge("analyze_single_review", "build_final_report")
    builder.add_edge("build_final_report", "save_output")
    builder.add_edge("save_output", END)

    return builder.compile()


def main():
    graph = build_graph()

    initial_state: ReviewDigestState = {
        "reviews": [],
        "review_analyses": [],
        "final_report": "",
        "output_path": "",
    }

    result = graph.invoke(initial_state)

    print("\nFinal report path:")
    print(result["output_path"])


if __name__ == "__main__":
    main()
