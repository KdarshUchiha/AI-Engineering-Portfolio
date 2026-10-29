# Project 6 — Multimodal Product Inspector

Hand Claude a **photo** of a product and get back structured catalog data: what
it is, its condition, any visible defects, and a ready-to-use listing title and
description.

## Why this project matters for an FDE portfolio

Every project before this sent **text**. This one sends an **image** — the
gateway to a huge class of real features: visual QA, receipt/document parsing,
inventory tagging, accessibility descriptions. "Upload a product photo, get a
catalog listing and a quality check" is something every marketplace, retailer,
and warehouse wants.

## How it works

```
image file ─▶ base64 + media_type ─▶ [image block] ┐
                                                    ├─▶ Claude ─▶ { product, condition,
"inspect this product"           ─▶ [text block]  ┘              defects[], listing… }
                                                                        │
                                                                   render report
```

Three ideas carry the project:

1. **Vision / multimodal input** — the image "content block" format: raw bytes,
   base64-encoded, plus a `media_type` (`image/png`, `image/jpeg`, …). Image and
   instructions travel together in one message.
2. **Vision + structured output together** — we don't want a paragraph, we want
   *fields* (color, material, condition, defects) we can store and search. This
   fuses project 2/4's `json_schema` trick with an image input.
3. **Honest uncertainty** — the `condition` enum includes `unknown`, there's a
   `confidence` score, and the prompt forbids inventing anything not visible.
   Teaching the model to say "I can't tell" is what makes it catalog-worthy.

## Files

| File                   | What it is                                                       |
|------------------------|------------------------------------------------------------------|
| `product_inspector.py` | Image loading + vision inspection + structured output + report, commented. |
| `sample_product.png`   | Auto-generated on first run (tiny stdlib-built PNG) — zero setup. |
| `README.md`            | This file.                                                       |

## Run it

```bash
pip install anthropic         # or: python -m pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"

# Zero-setup: generates and inspects a built-in sample image
python product_inspector.py

# Real use: point it at your own product photos
python product_inspector.py mug.jpg chair.png headphones.webp
```

It prints an inspection report per image and the raw JSON — the machine-readable
payoff a catalog system would ingest. Supported formats: PNG, JPEG, GIF, WebP.

> The built-in `sample_product.png` is a simple shape drawn with the standard
> library so the pipeline runs with no external files or image dependencies. For
> a real demo, use actual product photos — that's where the vision model shines.

## What to learn before moving on

- **Image content blocks** — base64 + `media_type`; mixing image and text blocks
  in one `content` list.
- **Multimodal + structured output** — "look at this and fill in these fields."
- **Uncertainty by design** — an `unknown` enum value and a `confidence` score
  beat a confident wrong guess.
- **Separation of concerns** — loading, inspecting, and rendering are three
  independent functions (easy to test the first and last with no API key).

## Upgrade ideas (LinkedIn / interview talking points)

- **Batch a folder** of photos into a CSV/JSON catalog.
- **Defect-only QA mode:** flag items whose `defects` list is non-empty for human
  review — a real warehouse workflow.
- **Multiple angles:** send several photos of the *same* item in one message for
  a more reliable assessment.
- Compare two photos ("before/after") to detect new damage.
- Swap the sample for a URL-image block, or resize large photos before encoding
  to cut token cost.
