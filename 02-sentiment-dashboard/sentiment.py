"""
Real-Time Sentiment Dashboard — classify a live stream of customer feedback
and render an updating dashboard in the terminal.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
Feedback items (reviews, tweets, support messages) arrive one at a time, as if
streaming in live. For each one we ask Claude to classify it into structured
data — sentiment, a 1-5 score, the topic, and whether it's urgent — and we
redraw a dashboard showing running totals after every item.

THE KEY SKILL THIS TEACHES: STRUCTURED OUTPUTS
----------------------------------------------
In Project #1 we read Claude's answer as free-form text. That's fine for a chat
answer, but useless if you need to COMPUTE on the result (count sentiments, sort
by urgency, feed a chart). Here we force Claude to return a strict JSON object
matching a schema we define. The API guarantees the shape, so we can do
`result["sentiment"]` with confidence instead of parsing prose.

This is the #2 most-requested AI-engineering skill after RAG. Almost every real
"AI feature" inside a product is: take messy input -> get back clean structured
data -> do something deterministic with it.

This builds directly on the Kindle customer-feedback sentiment tool you've
already worked on — same problem, now done the LLM-structured-output way.
"""

import os
import json
import time
import anthropic


# ----------------------------------------------------------------------------
# THE SCHEMA — the exact shape we force Claude to return
# ----------------------------------------------------------------------------
# This is a JSON Schema: a standard way to describe the structure of a JSON
# object. We hand it to the API and Claude is CONSTRAINED to produce JSON that
# validates against it. No more "please return JSON" and hoping — it's enforced.
#
# Read it top-down:
#   - type "object"           -> the result is a JSON object {...}
#   - properties              -> the fields it must contain, and each one's type
#   - enum                    -> the value must be ONE of this fixed list
#   - required                -> every one of these fields must be present
#   - additionalProperties:false -> NO extra fields allowed (keeps output clean)
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            # Restricting to a fixed set means we can safely switch/count on it.
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the feedback.",
        },
        "score": {
            "type": "integer",
            # The model still must pick from these; we validate range via enum
            # because JSON-schema numeric min/max aren't supported by the API's
            # structured-output validator (noted in the SDK docs).
            "enum": [1, 2, 3, 4, 5],
            "description": "Satisfaction score from 1 (worst) to 5 (best).",
        },
        "topic": {
            "type": "string",
            "enum": ["delivery", "quality", "price", "support", "usability", "other"],
            "description": "The main subject the feedback is about.",
        },
        "urgent": {
            "type": "boolean",
            "description": "True if this needs a human to follow up quickly "
                           "(e.g. an angry customer, a safety issue, a churn risk).",
        },
        "summary": {
            "type": "string",
            "description": "A 1-sentence summary of the feedback.",
        },
    },
    "required": ["sentiment", "score", "topic", "urgent", "summary"],
    # The model may not invent fields outside the five above.
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------
# CLASSIFY ONE FEEDBACK ITEM
# ----------------------------------------------------------------------------
def classify(feedback: str, client: anthropic.Anthropic) -> dict:
    """
    Send one feedback string to Claude and get back a validated dict matching
    SENTIMENT_SCHEMA.

    Returns a Python dict like:
        {"sentiment": "negative", "score": 2, "topic": "delivery",
         "urgent": True, "summary": "Package arrived broken and late."}
    """
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=512,
        # The system prompt sets the role. Kept stable (no per-request data) so
        # it could be prompt-cached later for cost savings on high volume.
        system=(
            "You are a customer-feedback analyst. Classify each piece of "
            "feedback accurately and concisely. Mark 'urgent' as true only when "
            "a human should follow up quickly."
        ),
        messages=[{"role": "user", "content": f"Feedback: {feedback}"}],
        # THIS is the structured-output part. output_config.format with a
        # json_schema tells the API: the response MUST be valid JSON matching
        # this schema. The first text block of the response will be that JSON.
        output_config={
            "format": {
                "type": "json_schema",
                "schema": SENTIMENT_SCHEMA,
            }
        },
    )

    # Because we constrained the format, the first text block IS a JSON string.
    # Find it, then json.loads() turns the JSON string into a Python dict.
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


# ----------------------------------------------------------------------------
# THE DASHBOARD — running aggregates over the stream
# ----------------------------------------------------------------------------
class Dashboard:
    """
    Holds the running totals as feedback streams in, and knows how to draw
    itself. Separating 'state' (the counts) from 'render' (drawing) is good
    structure — it's how real dashboards/UIs are organized.
    """

    def __init__(self):
        self.total = 0
        # Counters for each category. We seed every key at 0 so the dashboard
        # always shows all categories, even before any item lands in them.
        self.sentiments = {"positive": 0, "negative": 0, "neutral": 0}
        self.topics = {
            "delivery": 0, "quality": 0, "price": 0,
            "support": 0, "usability": 0, "other": 0,
        }
        self.score_sum = 0          # sum of all scores, to compute the average
        self.urgent_items = []      # list of (summary) for anything flagged urgent

    def update(self, result: dict):
        """Fold one classification result into the running totals."""
        self.total += 1
        self.sentiments[result["sentiment"]] += 1
        self.topics[result["topic"]] += 1
        self.score_sum += result["score"]
        if result["urgent"]:
            self.urgent_items.append(result["summary"])

    def _bar(self, count: int, max_count: int, width: int = 20) -> str:
        """Make a simple text bar like '########----' for a count."""
        if max_count == 0:
            return "-" * width
        filled = round(width * count / max_count)
        return "#" * filled + "-" * (width - filled)

    def render(self) -> str:
        """Build the dashboard as a string we can print."""
        avg = self.score_sum / self.total if self.total else 0
        lines = []
        lines.append("=" * 50)
        lines.append(f"  LIVE FEEDBACK DASHBOARD   ({self.total} items)")
        lines.append("=" * 50)
        lines.append(f"  Average satisfaction: {avg:.2f} / 5")
        lines.append("")
        lines.append("  Sentiment:")
        for name, count in self.sentiments.items():
            bar = self._bar(count, self.total)
            lines.append(f"    {name:<9} {bar} {count}")
        lines.append("")
        lines.append("  Topics:")
        # Only show topics that have at least one item, sorted most-common first.
        active = [(t, c) for t, c in self.topics.items() if c > 0]
        active.sort(key=lambda x: x[1], reverse=True)
        for topic, count in active:
            bar = self._bar(count, self.total)
            lines.append(f"    {topic:<9} {bar} {count}")
        lines.append("")
        # Highlight urgent items — the thing a human actually needs to act on.
        lines.append(f"  URGENT follow-ups ({len(self.urgent_items)}):")
        if self.urgent_items:
            for s in self.urgent_items:
                lines.append(f"    ! {s}")
        else:
            lines.append("    (none)")
        lines.append("=" * 50)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# A FAKE LIVE FEED — stands in for a real stream (Twitter API, review webhook…)
# ----------------------------------------------------------------------------
# In production, items would arrive from a message queue, a webhook, or a
# polling loop against some API. We hardcode a list and feed it one at a time
# with a small delay to SIMULATE a live stream. Swapping this for a real source
# is the only change needed to make it genuinely real-time.
SAMPLE_FEED = [
    "Absolutely love this product, arrived a day early and works perfectly!",
    "The app keeps crashing every time I open the settings page. Frustrating.",
    "Way too expensive for what you get. Found the same thing cheaper elsewhere.",
    "Support agent was super helpful and resolved my issue in minutes.",
    "Package showed up smashed and two days late. I want a refund NOW.",
    "It's fine. Does the job. Nothing special either way.",
    "URGENT: the device got extremely hot and smells like burning plastic!",
    "Checkout button doesn't work on mobile, had to use my laptop instead.",
]


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    client = anthropic.Anthropic()
    dashboard = Dashboard()

    for feedback in SAMPLE_FEED:
        # Classify this item (one API call).
        result = classify(feedback, client)
        # Fold it into the running totals.
        dashboard.update(result)

        # Redraw. "\033[2J\033[H" is an ANSI escape code that clears the
        # terminal screen and moves the cursor to the top-left, so each redraw
        # replaces the last one instead of scrolling — the "live" effect.
        print("\033[2J\033[H", end="")
        print(dashboard.render())
        print(f"\n  Just processed: \"{feedback[:60]}...\"")

        # Pause briefly so you can watch it update like a real feed.
        time.sleep(0.8)

    print("\nStream finished.")


if __name__ == "__main__":
    main()
