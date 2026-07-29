# Project 4 — Meeting Notes & Action-Item Bot

Turn a raw meeting transcript into a clean structured report: a summary, the key
decisions, and **action items with owners and deadlines** — plus the open
questions nobody resolved.

## Why this project matters for an FDE portfolio

"Summarize my meetings and tell me who owes what" is something *every* company
wants. This project shows you can take a long, messy, unstructured document and
reliably pull rich structured data out of it — the backbone of most real
document-processing AI features (contracts, tickets, emails, resumes, support
threads).

## What's new vs. project 2

Project 2 turned **one short message** into **one flat object**. This turns
**one long document** into a **nested structure**:

```
transcript ──▶ {
  summary,
  decisions:      [ ... ],
  action_items:   [ {owner, task, due}, {owner, task, due}, ... ],  ← nested!
  open_questions: [ ... ],
}
```

`action_items` is an *array of objects*, each with its own required fields. That
nesting is where structured outputs really pay off — you get data you can loop
over, sort by owner, and drop straight into a task tracker.

It also introduces **token counting**: before sending a big input we check its
size with the (cheap, separate) `count_tokens` endpoint — so we know the cost
and whether it fits the context window before paying for a full run.

## Files

| File               | What it is                                          |
|--------------------|-----------------------------------------------------|
| `meeting_notes.py` | Extraction + rendering + token counting, commented. |
| `README.md`        | This file.                                          |

## Run it

```bash
pip install anthropic         # or: python -m pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"
python meeting_notes.py
```

It counts the transcript's tokens, extracts structured notes, prints a tidy
report, and then prints the raw JSON — the machine-readable payoff you'd feed
into another system. The sample transcript deliberately includes an action item
with **no named owner**, so you can see the `Unassigned` / `No date` fallbacks
work.

## What to learn before moving on

- **Nested schemas** — arrays of objects, each with `required` fields.
- **Grounding instructions** — the system prompt forbids inventing owners or
  deadlines, and defines the fallbacks (`Unassigned`, `No date`).
- **`count_tokens`** — measure input size/cost *before* a billed generation.
- **Structure vs. render** — the JSON is the source of truth; the printed report
  is just one view of it (same lesson as project 2's dashboard).

## Upgrade ideas (LinkedIn / interview talking points)

- Feed real audio: run a recording through a speech-to-text service, then pipe
  the transcript in — the bot doesn't care about the source.
- Post action items to Slack/Jira, one per owner.
- Chunk very long transcripts (like project 1) if they exceed the context window.
- Email each attendee just their own action items.
