"""
Meeting Notes & Action-Item Bot — turn a raw meeting transcript into a clean,
structured summary with decisions and owner-assigned action items.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
You paste in a messy meeting transcript (the kind an auto-transcription tool
spits out). The bot asks Claude to extract:
  - a short summary of the meeting,
  - the key decisions that were made,
  - action items, each with an OWNER and a DUE DATE,
  - open questions left unresolved.
It returns that as structured data and prints a tidy report.

THE KEY SKILLS THIS TEACHES
---------------------------
1. LONG INPUT -> RICH STRUCTURED OUTPUT. Project 2 turned one short message into
   one flat object. Here we take a LONG document and extract a NESTED structure:
   a list of action items, each an object {owner, task, due}. This "read a
   document, return structured fields" shape is the backbone of most real
   document-processing AI features (contracts, tickets, emails, resumes...).

2. TOKEN COUNTING. Before sending a big input, we check how many tokens it is,
   so we know the cost and whether it fits. Knowing the size of your input
   before you pay for it is basic production hygiene.

Every company on earth wants "summarize my meetings and tell me who owes what."
That's why this is a strong portfolio piece.
"""

import os
import json
import anthropic


# ----------------------------------------------------------------------------
# THE SCHEMA — the structured shape we extract from the transcript
# ----------------------------------------------------------------------------
# Note the NESTED part: `action_items` is an ARRAY of OBJECTS. Each object has
# its own required fields. This is more advanced than project 2's flat object,
# and it's where structured outputs really earn their keep — you get back data
# you can loop over, sort by owner, drop into a task tracker, etc.
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A 2-3 sentence overview of what the meeting covered.",
        },
        "decisions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete decisions that were made during the meeting.",
        },
        "action_items": {
            "type": "array",
            # Each element of the array is itself a structured object.
            "items": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Person responsible. Use 'Unassigned' if "
                                       "no owner was named.",
                    },
                    "task": {
                        "type": "string",
                        "description": "What needs to be done.",
                    },
                    "due": {
                        "type": "string",
                        "description": "Deadline if mentioned, else 'No date'.",
                    },
                },
                "required": ["owner", "task", "due"],
                "additionalProperties": False,
            },
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unresolved questions raised but not answered.",
        },
    },
    "required": ["summary", "decisions", "action_items", "open_questions"],
    "additionalProperties": False,
}


# ----------------------------------------------------------------------------
# TOKEN COUNTING — know your input size before you pay for it
# ----------------------------------------------------------------------------
def count_input_tokens(transcript: str, client: anthropic.Anthropic) -> int:
    """
    Ask the API how many tokens the transcript is, WITHOUT running a full
    (billed) generation. This is a cheap, separate endpoint.

    Why bother? Two reasons:
      - Cost awareness: input tokens cost money; big transcripts cost more.
      - Fit: every model has a context-window limit. If a transcript were
        enormous, you'd need to chunk it (like project 1) rather than send it
        whole. Counting first tells you whether you need to.
    """
    result = client.messages.count_tokens(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": transcript}],
    )
    return result.input_tokens


# ----------------------------------------------------------------------------
# THE EXTRACTION — transcript in, structured notes out
# ----------------------------------------------------------------------------
def extract_notes(transcript: str, client: anthropic.Anthropic) -> dict:
    """Send the transcript to Claude and get back a dict matching NOTES_SCHEMA."""
    system_prompt = (
        "You extract structured meeting notes from raw transcripts. Base every "
        "field ONLY on what the transcript actually says — do not invent "
        "decisions, owners, or deadlines. If an action item has no clear owner, "
        "set owner to 'Unassigned'; if no deadline is stated, set due to "
        "'No date'."
    )

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2048,   # notes can be longer than a one-line classification
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Extract the meeting notes from this transcript:\n\n{transcript}",
        }],
        # Same structured-output mechanism as project 2, but with a nested schema.
        output_config={
            "format": {"type": "json_schema", "schema": NOTES_SCHEMA}
        },
    )

    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


# ----------------------------------------------------------------------------
# RENDERING — turn the structured data into a readable report
# ----------------------------------------------------------------------------
# Keeping rendering separate from extraction is the same "state vs. render"
# lesson from project 2: the structured dict is the source of truth; how we
# display it (terminal now, HTML/email later) is a separate concern.
def render_notes(notes: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  MEETING NOTES")
    lines.append("=" * 60)
    lines.append("\nSUMMARY")
    lines.append(f"  {notes['summary']}")

    lines.append("\nDECISIONS")
    if notes["decisions"]:
        for d in notes["decisions"]:
            lines.append(f"  - {d}")
    else:
        lines.append("  (none)")

    lines.append("\nACTION ITEMS")
    if notes["action_items"]:
        # Loop over the nested objects. Because the schema guaranteed each has
        # owner/task/due, we can index them directly with no defensive checks.
        for item in notes["action_items"]:
            lines.append(f"  [ ] {item['task']}")
            lines.append(f"        owner: {item['owner']}   due: {item['due']}")
    else:
        lines.append("  (none)")

    lines.append("\nOPEN QUESTIONS")
    if notes["open_questions"]:
        for q in notes["open_questions"]:
            lines.append(f"  ? {q}")
    else:
        lines.append("  (none)")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# A SAMPLE TRANSCRIPT — stands in for real auto-transcribed audio
# ----------------------------------------------------------------------------
# In production this would come from a speech-to-text service (Whisper, AWS
# Transcribe, etc.) fed from a recording. The bot doesn't care about the source
# — text is text. We hardcode a realistic, messy transcript here.
SAMPLE_TRANSCRIPT = """
Alright everyone, thanks for joining the Q3 planning sync. So, um, first thing —
we looked at the numbers and we've decided we're going to push the mobile app
launch from August to September. The extra month gives QA enough runway.

Priya: I can own the updated launch timeline doc. I'll have it ready by Friday.

Sure, Priya owns the timeline doc, Friday. Next, the pricing question. We went
back and forth but we agreed to keep the current pricing tiers for now and
revisit after launch. Marcus, can you pull the competitor pricing analysis
together so we have data for that revisit?

Marcus: Yeah I'll take that. Not sure on timing though, maybe end of next week?

Okay end of next week-ish for Marcus. One thing we didn't resolve — do we need a
dedicated support hire before launch or can the existing team absorb it? Let's
flag that. Also someone needs to update the investor deck with the new dates but
we didn't say who. Let's leave that unassigned for now and sort it offline.

Last thing, we confirmed the marketing budget increase of 15% got approved.
Great work everyone, that's a wrap.
"""


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    client = anthropic.Anthropic()

    # 1. Count tokens first (cheap) so we know the input size and cost.
    tokens = count_input_tokens(SAMPLE_TRANSCRIPT, client)
    print(f"Transcript is {tokens} input tokens.\n")

    # 2. Extract structured notes.
    print("Extracting notes...\n")
    notes = extract_notes(SAMPLE_TRANSCRIPT, client)

    # 3. Render the human-readable report.
    print(render_notes(notes))

    # 4. Also show the raw structured data — this is what you'd feed into a task
    #    tracker, a database, or a Slack message. The structure is the payoff.
    print("\nRaw structured output (what a system would consume):")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
