# Project 5 — SQL Copilot

Ask a question in plain English, get a real answer from a real database. You
type *"Which 3 customers spent the most?"*; Claude writes the SQL, SQLite runs
it, and you see the rows.

## Why this project matters for an FDE portfolio

Every company has a database and non-technical people who need answers from it
without learning SQL. "Talk to my data" is one of the most-requested AI features
in the real world. This project shows you can build it — and, just as important,
build it *safely*.

## How it works

```
question ─▶ [ + live schema ] ─▶ Claude ─▶ { sql, explanation }
                                              │
                                   is_safe_select()  ◀── guardrail
                                              │  (only a lone SELECT passes)
                                              ▼
                                        SQLite runs it ─▶ rows ─▶ table
```

Three ideas carry the whole project:

1. **Text-to-SQL** — the core "talk to my data" pattern. Translate natural
   language into a query.
2. **Grounding the model in real structure** — the model can't guess your column
   names, so we introspect the live database (`sqlite_master` + `PRAGMA
   table_info`) and hand Claude the exact schema. Same "ground the model in real
   context" lesson as RAG in project 1 — here the context is a *schema*.
3. **Executing model output safely** — the SQL is about to *run*, so we treat it
   as untrusted input. Defense in depth: reject anything that isn't a single
   read-only `SELECT`, and (in production) also open the connection read-only so
   writes fail at the driver level too.

## What's new vs. earlier projects

- Project 2 & 4 extracted structured data for *you* to read. Here the structured
  output (`{sql, explanation}`) gets **executed** — raising the stakes and
  introducing the safety guardrail.
- The model's output feeds a real system (a database), so this is your first
  taste of **sandboxing / validating LLM output before acting on it** — the
  central concern of every agent that touches the real world.

## Files

| File             | What it is                                                     |
|------------------|----------------------------------------------------------------|
| `sql_copilot.py` | Sample DB + schema introspection + NL→SQL + guardrail + runner, commented. |
| `README.md`      | This file.                                                     |

## Run it

```bash
pip install anthropic         # or: python -m pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"
python sql_copilot.py
```

It prints the schema, answers four demo questions (joins, aggregation,
grouping, sorting), then drops into an interactive prompt so you can ask your
own. The database is an in-memory SQLite store (customers / products / orders) —
zero setup, nothing to clean up.

Try asking:
- "What's the average order quantity by city?"
- "Which products have never been ordered?"
- "Show total revenue per month."

## The safety guardrail, in detail

`is_safe_select()` is the gate every generated query must pass before it runs:

- must start with `SELECT`,
- must contain no write/schema keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`,
  `ALTER`, `PRAGMA`, …),
- must be a **single** statement — so `SELECT 1; DROP TABLE orders` is rejected
  (statement stacking).

In production you'd add a second layer: open the database read-only (SQLite
supports `sqlite3.connect("file:store.db?mode=ro", uri=True)`) so even a query
that slips past the text check cannot write. Belt **and** suspenders — because
the downside of a dropped table or leaked row is severe.

## What to learn before moving on

- **Schema grounding** — introspect the live DB; never hardcode the schema.
- **Structured output that gets executed** — and why that raises the stakes.
- **Treating LLM output as untrusted** — validate before you act; least
  privilege (read-only) at the layer below.
- **Error feedback** — the model can still write invalid SQL; the natural next
  step is feeding the DB's error back to the model for a self-correcting retry.

## Upgrade ideas (LinkedIn / interview talking points)

- **Self-correcting retry:** on a `SQL ERROR`, send the error back to Claude and
  ask it to fix the query (an agent loop, like project 3).
- Point it at a real database (Postgres/MySQL) — the logic is unchanged; only
  the connection and schema introspection differ.
- Render results as a chart, or export to CSV.
- Add row-level security / allowlisted tables so the copilot can only read what
  a given user is permitted to see.
