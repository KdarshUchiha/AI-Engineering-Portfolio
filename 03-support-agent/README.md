# Project 3 — AI Customer-Support Agent

An agent that answers customer questions by **calling tools to look up real
data** (order status, inventory) instead of guessing — and escalates to a human
when it can't help.

## Why this project matters for an FDE portfolio

This is the jump from **"language model"** to **"agent."** Projects 1 and 2 made
Claude *produce* text/JSON. Here Claude *takes actions*: it decides which
function to call, with what arguments, reads the result, and loops until it can
answer. That loop is the core of every "AI agent" product — and "can you build
an agent that talks to our internal systems?" is one of the most common
forward-deployed asks.

## The agent loop (the whole idea in one diagram)

```
   ┌──────────────────────────────────────────────┐
   │ 1. Send conversation + tool definitions        │
   │ 2. Claude replies:                              │
   │      • a final answer   → DONE                  │
   │      • "call tool X(args)" (stop_reason=tool_use)│
   │ 3. WE run the real function, append its result  │
   │ 4. Loop back to step 1                          │
   └──────────────────────────────────────────────┘
```

Example: *"Is the USB-C Cable in stock? My order is A1002."* →
Claude calls `lookup_order("A1002")` **and** `check_inventory("USB-C Cable")` →
reads both results → answers with the real status and (out-of-stock) availability.

## How the pieces fit

| Piece | Role |
|-------|------|
| `lookup_order`, `check_inventory` | The **real functions** (fake in-memory data here; a DB/API in production). |
| `TOOLS` | JSON descriptions Claude reads to decide **when** and **how** to call each function. |
| `TOOL_FUNCTIONS` | A name → function registry, so when Claude says `"lookup_order"` we run the right code. |
| `run_agent` | The **manual loop** that ties it together. Written by hand (not the SDK auto-runner) so you can see every step. |

## Files

| File               | What it is                                     |
|--------------------|------------------------------------------------|
| `support_agent.py` | The agent + tools + loop, heavily commented.   |
| `README.md`        | This file.                                     |

## Run it

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"
python support_agent.py
```

> If you get `ModuleNotFoundError: No module named 'anthropic'` even after
> installing, your shell may be using a different Python. Install into the
> exact interpreter you run with:
> ```bash
> python -m pip install anthropic
> ```

You'll see 4 customer questions handled, with the tool calls printed inline
(`[agent] calling ...` / `[tool] -> ...`) so you can watch the agent's reasoning
turn into real lookups — including an unknown order that gets escalated.

## What to learn before moving on

- **`stop_reason == "tool_use"`** — how you detect Claude wants to act vs. answer.
- **The append order** — you must append the assistant turn (with its
  `tool_use` blocks) *before* sending back `tool_result`s, matched by
  `tool_use_id`.
- **All results in one user message** — batch every turn's `tool_result`s
  together.
- **Bounding the loop** — `max_turns` stops a runaway agent (and runaway cost).
- **Tool descriptions are a skill** — vague descriptions make the agent pick the
  wrong tool.

## Upgrade ideas (LinkedIn / interview talking points)

- Add a real tool: process a refund, create a ticket, send an email.
- Add human-in-the-loop approval before any state-changing tool runs.
- Swap the hand-written loop for the SDK's `tool_runner` and compare.
- Turn it into a chat UI (web-UI phase) so it's a live support widget.
