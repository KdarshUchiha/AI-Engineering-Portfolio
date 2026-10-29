"""
Multimodal Product Inspector — hand Claude a PHOTO of a product and get back
structured catalog data: what it is, its condition, any visible defects, and a
ready-to-use listing title and description.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
You point the program at an image file (a product photo). It:
  1. loads the image and base64-encodes it the way the API expects,
  2. sends the image to Claude with a structured-output schema,
  3. gets back structured fields — product name, category, color, material,
     condition, a list of visible defects, and a generated listing title +
     description,
  4. prints a tidy inspection report.

THE KEY SKILLS THIS TEACHES
---------------------------
1. VISION / MULTIMODAL INPUT. Every project so far sent TEXT. Here we send an
   IMAGE. You learn the image "content block" format: base64 data + a media type
   (image/png, image/jpeg, ...). Sending images is the gateway to a huge class
   of features — document OCR, receipt parsing, visual QA, accessibility.

2. VISION + STRUCTURED OUTPUT TOGETHER. We don't just want a paragraph about the
   photo; we want DATA (fields we can store, search, and put in a catalog). This
   combines project 2/4's json_schema trick with an image input — "look at this
   and fill in these fields" — which is how most real vision features ship.

3. HONEST UNCERTAINTY. A photo can be ambiguous. The schema has an explicit
   `condition: unknown` option and a `confidence` score, and the prompt tells
   Claude to describe only what's visibly present. Teaching the model to say "I
   can't tell" is what separates a demo from something you'd trust in a catalog.

"Upload a product photo, get a catalog listing and a QA check" is something
every marketplace, retailer, and warehouse wants. Strong portfolio piece.
"""

import os
import sys
import json
import zlib
import base64
import struct
import anthropic


# ----------------------------------------------------------------------------
# LOADING AN IMAGE — bytes on disk -> the content block the API expects
# ----------------------------------------------------------------------------
# The API doesn't take a file path. It takes the raw image bytes, base64-encoded,
# plus a "media type" string telling it the format. This helper turns a path into
# exactly the dict the Messages API wants in its `content` list.
_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def media_type_for(path: str) -> str:
    """Map a file extension to the API's media-type string."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MEDIA_TYPES:
        raise ValueError(
            f"Unsupported image type '{ext}'. Use one of: "
            f"{', '.join(sorted(_MEDIA_TYPES))}"
        )
    return _MEDIA_TYPES[ext]


def load_image_block(path: str) -> dict:
    """
    Read an image file and return an API image content block:

        {"type": "image",
         "source": {"type": "base64", "media_type": "...", "data": "<b64>"}}

    Base64 is just a way to represent binary bytes as ASCII text, so the image
    can travel inside a JSON request body.
    """
    with open(path, "rb") as f:
        raw = f.read()
    encoded = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type_for(path),
            "data": encoded,
        },
    }


# ----------------------------------------------------------------------------
# THE SCHEMA — the structured fields we extract from the photo
# ----------------------------------------------------------------------------
# Note the `condition` enum includes "unknown", and there's a `confidence`
# score. Those exist so the model has a principled way to express doubt instead
# of guessing — crucial when a human or a downstream system acts on the result.
INSPECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "description": "Best-guess name of the product in the image.",
        },
        "category": {
            "type": "string",
            "description": "Product category, e.g. 'Electronics', 'Furniture'.",
        },
        "primary_color": {"type": "string"},
        "material": {
            "type": "string",
            "description": "Best-guess material, or 'unknown' if unclear.",
        },
        "condition": {
            "type": "string",
            "enum": ["new", "used", "damaged", "unknown"],
            "description": "Visible condition. Use 'unknown' if you can't tell.",
        },
        "defects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Visible defects (scratches, dents, stains). Empty "
                           "list if none are visible.",
        },
        "listing_title": {
            "type": "string",
            "description": "A short, catalog-ready marketing title.",
        },
        "listing_description": {
            "type": "string",
            "description": "1-2 sentence marketing description based ONLY on "
                           "what is visible.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 overall confidence in this assessment.",
        },
    },
    "required": [
        "product_name", "category", "primary_color", "material",
        "condition", "defects", "listing_title", "listing_description",
        "confidence",
    ],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------
# THE INSPECTION — image in, structured catalog data out
# ----------------------------------------------------------------------------
def inspect_product(path: str, client: anthropic.Anthropic) -> dict:
    """Send the product photo to Claude and return a dict matching the schema."""
    system_prompt = (
        "You are a product-catalog vision inspector. Look at the product photo "
        "and fill in the structured fields. Describe ONLY what is visibly "
        "present — do not invent brands, materials, or features you cannot see. "
        "If something is unclear, use 'unknown' and lower your confidence. List "
        "every visible defect; if the product looks flawless, return an empty "
        "defects list."
    )

    # The content list mixes an IMAGE block and a TEXT block. This is the core
    # multimodal move: image and instructions travel together in one message.
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                load_image_block(path),
                {"type": "text", "text": "Inspect this product and fill in the fields."},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": INSPECTION_SCHEMA}},
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


# ----------------------------------------------------------------------------
# RENDERING — turn the structured data into a readable report
# ----------------------------------------------------------------------------
def render_report(data: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  PRODUCT INSPECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"\nPRODUCT     {data['product_name']}")
    lines.append(f"CATEGORY    {data['category']}")
    lines.append(f"COLOR       {data['primary_color']}")
    lines.append(f"MATERIAL    {data['material']}")
    lines.append(f"CONDITION   {data['condition']}")
    lines.append(f"CONFIDENCE  {data['confidence']:.0%}")

    lines.append("\nDEFECTS")
    if data["defects"]:
        for d in data["defects"]:
            lines.append(f"  ! {d}")
    else:
        lines.append("  (none visible)")

    lines.append("\nSUGGESTED LISTING")
    lines.append(f"  Title: {data['listing_title']}")
    lines.append(f"  Desc:  {data['listing_description']}")
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# A SAMPLE IMAGE — so the project runs with zero external files
# ----------------------------------------------------------------------------
# We write a tiny PNG by hand using only the standard library (zlib + struct),
# so there are no image dependencies to install. It draws a simple "box" product
# on a light background — enough to exercise the full vision pipeline end to end.
# In real use you'd point the program at actual product photos (see the README).
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def generate_sample_image(path: str, w: int = 240, h: int = 160) -> None:
    """Create a minimal valid RGB PNG at `path` (no third-party libraries)."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)  # PNG per-scanline filter type: 0 = none
        for x in range(w):
            r, g, b = 245, 245, 245               # light background
            if 60 <= x < 180 and 40 <= y < 130:   # the product box
                r, g, b = 70, 130, 180            # steel-blue body
                if 40 <= y < 60:                  # a darker "lid" band
                    r, g, b = 40, 90, 140
            rows.extend((r, g, b))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def main():
    # Accept one or more image paths on the command line; otherwise fall back to
    # a generated sample so the project runs out of the box.
    paths = sys.argv[1:]
    if not paths:
        sample = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sample_product.png")
        if not os.path.exists(sample):
            generate_sample_image(sample)
        print(f"No image given — using generated sample: {sample}")
        print("(Tip: pass your own photos, e.g. `python product_inspector.py a.jpg b.png`)")
        paths = [sample]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    client = anthropic.Anthropic()
    for path in paths:
        print(f"\nInspecting: {path}")
        data = inspect_product(path, client)
        print(render_report(data))
        # The raw JSON is the real payoff — what a catalog system would ingest.
        print("\nRaw structured output (what a system would consume):")
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
