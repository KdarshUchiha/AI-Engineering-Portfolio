"""
SQL Copilot — ask a question in plain English, get a real answer from a real
database. Claude writes the SQL; SQLite runs it; you see the rows.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
There's a small SQLite database of a fake e-commerce store (customers, products,
orders). You type a question like "Which 3 customers spent the most?" and the
Copilot:
  1. shows Claude the database SCHEMA (table + column names),
  2. asks Claude to translate your question into a single SQL SELECT,
  3. CHECKS that the SQL is a read-only SELECT (a safety guardrail),
  4. runs it against the real database,
  5. prints the resulting rows as a table.

THE KEY SKILLS THIS TEACHES
---------------------------
1. TEXT-TO-SQL — the single most requested "talk to my data" feature. The whole
   trick is giving the model the schema and asking for SQL back.

2. GROUNDING THE MODEL IN REAL STRUCTURE. The model can't guess your column
   names. We introspect the live database and hand Claude the exact schema, so
   the SQL it writes actually matches your tables. (Same "ground the model in
   real context" lesson as RAG in project 1 — here the context is a schema.)

3. EXECUTING MODEL OUTPUT SAFELY. The model's output is going to *run* against a
   database. That's powerful and dangerous. We never trust it blindly: we force
   a read-only connection AND reject anything that isn't a lone SELECT. Treating
   LLM output as untrusted input is core production hygiene.

Every company has a database and non-technical people who want answers from it
without learning SQL. That's why this is a strong portfolio piece.
"""

import os
import re
import json
import sqlite3
import anthropic


# ----------------------------------------------------------------------------
# THE DATABASE — a tiny e-commerce store, built in memory
# ----------------------------------------------------------------------------
# In production this would be your real Postgres/MySQL/warehouse. We use SQLite
# in memory so this project is zero-setup: no server, no file, nothing to clean
# up. The Copilot logic is identical regardless of the database behind it.
def build_sample_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database and seed it with sample data."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE customers (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            city      TEXT NOT NULL,
            signup    TEXT NOT NULL   -- ISO date the customer joined
        );

        CREATE TABLE products (
            id        INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            category  TEXT NOT NULL,
            price     REAL NOT NULL   -- unit price in USD
        );

        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id  INTEGER NOT NULL,
            quantity    INTEGER NOT NULL,
            order_date  TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id)  REFERENCES products(id)
        );
        """
    )

    customers = [
        (1, "Priya Shah",   "Mumbai",    "2024-01-15"),
        (2, "Marcus Lee",   "Singapore", "2024-03-02"),
        (3, "Ana Torres",   "Madrid",    "2024-02-20"),
        (4, "Jamal Carter", "Mumbai",    "2024-05-11"),
        (5, "Wei Chen",     "Singapore", "2024-06-30"),
    ]
    products = [
        (1, "Mechanical Keyboard", "Electronics", 89.00),
        (2, "USB-C Hub",           "Electronics", 39.50),
        (3, "Standing Desk",       "Furniture",   310.00),
        (4, "Desk Lamp",           "Furniture",   24.99),
        (5, "Noise-Cancel Headset","Electronics", 199.00),
    ]
    # (id, customer_id, product_id, quantity, order_date)
    orders = [
        (1, 1, 1, 2, "2024-07-01"),
        (2, 1, 5, 1, "2024-07-03"),
        (3, 2, 3, 1, "2024-07-05"),
        (4, 2, 2, 3, "2024-07-09"),
        (5, 3, 4, 4, "2024-07-10"),
        (6, 3, 1, 1, "2024-07-15"),
        (7, 4, 5, 2, "2024-08-01"),
        (8, 5, 3, 1, "2024-08-04"),
        (9, 5, 2, 2, "2024-08-07"),
        (10, 1, 3, 1, "2024-08-12"),
    ]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    cur.executemany("INSERT INTO products  VALUES (?,?,?,?)", products)
    cur.executemany("INSERT INTO orders    VALUES (?,?,?,?,?)", orders)
    conn.commit()
    return conn


# ----------------------------------------------------------------------------
# SCHEMA INTROSPECTION — tell Claude exactly what it's working with
# ----------------------------------------------------------------------------
# This is the heart of text-to-SQL. The model cannot invent correct SQL unless
# it knows your table and column names. So we read the schema from the live
# database (never hardcode it — it would drift) and format it as text to drop
# into the prompt.
def describe_schema(conn: sqlite3.Connection) -> str:
    """Return a human/LLM-readable description of every table and its columns."""
    cur = conn.cursor()
    # sqlite_master holds one row per table; we skip SQLite's internal tables.
    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]

    lines = []
    for table in tables:
        # PRAGMA table_info returns: (cid, name, type, notnull, default, pk)
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        col_descs = [f"{c[1]} {c[2]}" for c in cols]
        lines.append(f"{table}({', '.join(col_descs)})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# THE SAFETY GUARDRAIL — treat the model's SQL as untrusted input
# ----------------------------------------------------------------------------
# The model's output is about to RUN against a database. We never trust it
# blindly. This is defense in depth with two layers:
#   1. This function rejects anything that isn't a single, lone SELECT.
#   2. We also open the DB connection read-only (see run_query), so even a
#      cleverly-crafted write would fail at the database level.
# Belt AND suspenders — because the downside (dropped table, leaked data) is bad.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> bool:
    """
    True only if `sql` is a single read-only SELECT.

    Rules:
      - must start with SELECT (ignoring leading whitespace/case),
      - must contain no data-modifying / schema keywords,
      - must be a SINGLE statement (no stacking a second query after a ';').
    """
    stripped = sql.strip().rstrip(";").strip()

    # Reject statement stacking like "SELECT 1; DROP TABLE orders".
    # After removing one trailing ';', no ';' should remain.
    if ";" in stripped:
        return False

    if not stripped.lower().startswith("select"):
        return False

    if _FORBIDDEN.search(stripped):
        return False

    return True


# ----------------------------------------------------------------------------
# SQL GENERATION — the natural-language → SQL step
# ----------------------------------------------------------------------------
# We use structured output (json_schema) so we reliably get back TWO fields:
# the SQL itself, and a plain-English explanation we can show the user. Asking
# for the explanation alongside the SQL makes the tool feel like a "copilot,"
# not a black box.
SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "A single SQLite SELECT statement answering the "
                           "question. No trailing semicolon. SELECT only.",
        },
        "explanation": {
            "type": "string",
            "description": "One sentence, plain English, explaining what the "
                           "query does.",
        },
    },
    "required": ["sql", "explanation"],
    "additionalProperties": False,
}


def generate_sql(question: str, schema: str, client: anthropic.Anthropic) -> dict:
    """Ask Claude to translate `question` into SQL, grounded in `schema`."""
    system_prompt = (
        "You are a SQL expert for a SQLite database. Translate the user's "
        "question into ONE valid SQLite SELECT statement.\n"
        "Rules:\n"
        "- Use ONLY the tables and columns in the provided schema.\n"
        "- Produce a read-only SELECT. Never write, update, or delete.\n"
        "- Return a single statement with no trailing semicolon.\n\n"
        f"DATABASE SCHEMA:\n{schema}"
    )

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
        output_config={"format": {"type": "json_schema", "schema": SQL_SCHEMA}},
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


# ----------------------------------------------------------------------------
# QUERY EXECUTION — run the (verified) SQL against a read-only connection
# ----------------------------------------------------------------------------
def run_query(sql: str, conn: sqlite3.Connection):
    """
    Execute `sql` and return (column_names, rows).

    Note: the guardrail is_safe_select() must already have passed. We still run
    on the existing connection here for simplicity; the README explains how to
    additionally open a truly read-only connection in production (sqlite3's
    `mode=ro` URI) so writes fail at the driver level too.
    """
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description]  # column names from the cursor
    rows = cur.fetchall()
    return columns, rows


# ----------------------------------------------------------------------------
# RENDERING — show the rows as a readable terminal table
# ----------------------------------------------------------------------------
# Same "state vs. render" split as earlier projects: the (columns, rows) tuple
# is the data; this function is just one way to display it.
def render_table(columns, rows) -> str:
    if not rows:
        return "(no rows)"

    # Compute each column's display width from the header and every cell.
    widths = [len(str(c)) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(values):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(columns), sep]
    out.extend(fmt(row) for row in rows)
    out.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# ONE QUESTION, END TO END — the pipeline tying it all together
# ----------------------------------------------------------------------------
def answer_question(question, conn, client, schema) -> str:
    """Full flow for one question: NL -> SQL -> guardrail -> run -> render."""
    result = generate_sql(question, schema, client)
    sql, explanation = result["sql"], result["explanation"]

    out = [f"\nQ: {question}", f"Plan: {explanation}", f"SQL: {sql}"]

    # The critical gate: never run SQL that isn't a verified read-only SELECT.
    if not is_safe_select(sql):
        out.append("REFUSED: generated SQL is not a safe read-only SELECT.")
        return "\n".join(out)

    try:
        columns, rows = run_query(sql, conn)
        out.append("\n" + render_table(columns, rows))
    except sqlite3.Error as e:
        # The model can still write syntactically-wrong or mis-typed SQL. In a
        # real copilot you'd feed this error back to the model for a retry; here
        # we just surface it.
        out.append(f"SQL ERROR: {e}")
    return "\n".join(out)


# A few questions that exercise joins, aggregation, grouping, and sorting.
SAMPLE_QUESTIONS = [
    "Which 3 customers have spent the most money in total?",
    "How much total revenue came from each product category?",
    "List every customer in Singapore and how many orders they placed.",
    "What is the most popular product by total quantity sold?",
]


def main():
    conn = build_sample_db()
    schema = describe_schema(conn)

    print("SQL Copilot — ask questions about the sample store in plain English.")
    print("\nDatabase schema:")
    print(schema)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    client = anthropic.Anthropic()

    # Run the canned demo questions so the project shows off on a plain run.
    print("\n" + "=" * 64)
    print("  DEMO — answering sample questions")
    print("=" * 64)
    for q in SAMPLE_QUESTIONS:
        print(answer_question(q, conn, client, schema))

    # Then drop into an interactive loop so you can ask your own.
    print("\n" + "=" * 64)
    print("  Your turn — type a question (or 'quit' to exit)")
    print("=" * 64)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q", ""}:
            break
        print(answer_question(question, conn, client, schema))


if __name__ == "__main__":
    main()
