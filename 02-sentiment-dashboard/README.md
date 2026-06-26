# Project 2 — Real-Time Sentiment Dashboard

Classify a live stream of customer feedback with Claude and watch a dashboard
update in real time. Builds directly on the Kindle customer-feedback work — same
problem, now solved the modern LLM way.

## Why this project matters for an FDE portfolio

This teaches **structured outputs** — forcing the LLM to return strict JSON that
matches a schema you define. It's the #2 most-requested AI engineering skill
after RAG, because almost every real AI *feature* inside a product is:

> messy input → clean structured data → do something deterministic with it
> (count it, sort it, chart it, route it, store it).

If you can show a recruiter "I take unstructured text and reliably turn it into
typed data a system can act on," you've shown the core of production AI work.

## What it does

```
feedback stream ──▶ classify() with Claude ──▶ {sentiment, score, topic,
   (one item             (JSON schema-              urgent, summary}
    at a time)            constrained)                    │
                                                          ▼
                                              Dashboard.update() ──▶ live redraw
```

For each feedback item Claude returns a validated object:
```json
{"sentiment": "negative", "score": 2, "topic": "delivery",
 "urgent": true, "summary": "Package arrived broken and late."}
```
The dashboard folds each result into running totals (sentiment mix, average
score, topic breakdown, urgent follow-ups) and redraws after every item.

## The key idea: structured outputs

In Project #1 we read Claude's answer as free-form text. Here we pass
`output_config.format` with a **JSON Schema**, and the API *guarantees* the
response is valid JSON in that exact shape:

```python
output_config={"format": {"type": "json_schema", "schema": SENTIMENT_SCHEMA}}
```

- `enum` locks a field to a fixed set of values (so you can safely count/switch).
- `required` forces every field to be present.
- `additionalProperties: false` blocks junk fields.

That means `result["sentiment"]` is *always* one of `positive/negative/neutral`
— no fragile prose parsing, no "please return JSON" and hoping.

## Files

| File           | What it is                                              |
|----------------|---------------------------------------------------------|
| `sentiment.py` | The classifier + dashboard, heavily commented.          |
| `README.md`    | This file.                                              |

## Run it

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"
python sentiment.py
```

You'll watch 8 sample feedback items stream in, each classified, with the
dashboard redrawing live and urgent items (like an overheating device) surfaced
for follow-up.

## Make it genuinely real-time

The only fake part is `SAMPLE_FEED`. Replace it with a real source — poll a
reviews API, read from a webhook/queue, or tail a file — and feed items into
`classify()` exactly the same way. The rest is unchanged.

## What to learn before moving on

- **JSON Schema** — `type`, `properties`, `enum`, `required`,
  `additionalProperties`.
- **`output_config.format`** — how to constrain Claude's output shape.
- **State vs. render** — `Dashboard` separates the running totals from drawing,
  the way real UIs are built.
- **Streaming mindset** — process items one at a time and aggregate, instead of
  batching everything.

## Upgrade ideas (LinkedIn / interview talking points)

- Swap the terminal dashboard for a live web chart (we'll do this in the web-UI
  phase).
- Use the Batch API to classify thousands of items at 50% cost.
- Route urgent items to a Slack channel automatically.
- Add prompt caching on the stable system prompt for high-volume cost savings.
