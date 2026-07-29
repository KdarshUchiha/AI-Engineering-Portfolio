"""
AI Customer-Support Agent — an agent that answers customer questions by CALLING
TOOLS to look up real data, instead of guessing from memory.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
A customer asks something like "Where is my order A1001 and is the item in
stock?" The agent:
  - realizes it needs facts it doesn't have,
  - CALLS a `lookup_order` tool and an `check_inventory` tool (functions we
    wrote that read a fake database),
  - reads the results,
  - and answers using the real data — or escalates to a human if it can't help.

THE KEY SKILL THIS TEACHES: TOOL USE / AGENTS
---------------------------------------------
This is the jump from "language model" to "agent." In projects 1 and 2, Claude
produced text/JSON. Here Claude decides to TAKE ACTIONS: it looks at the
question, decides which of our functions to call and with what arguments, we
run those functions, hand back the results, and Claude continues — looping until
it has enough to answer.

That loop is THE core agent pattern:

    ┌─────────────────────────────────────────────┐
    │ 1. Send conversation + tool definitions      │
    │ 2. Claude replies: either a final answer      │
    │    OR "call tool X with these arguments"      │
    │ 3. If tool call: WE run the function,         │
    │    append the result, go back to step 1       │
    │ 4. If final answer: done                      │
    └─────────────────────────────────────────────┘

We write the loop by hand (instead of using the SDK's auto-runner) so you can
SEE each step. Every "AI agent" product is a fancier version of this loop.
"""

import os
import json
import anthropic


# ============================================================================
# PART 1 — THE "BACKEND": fake data + the functions the agent can call
# ============================================================================
# In a real company these functions would hit a database or internal API. We
# fake them with in-memory dicts so the project runs with no infrastructure.
# The agent doesn't know or care that it's fake — it just calls the function.

ORDERS = {
    "A1001": {"status": "shipped", "item": "Wireless Headphones",
              "eta": "2026-08-02", "tracking": "1Z999AA10123456784"},
    "A1002": {"status": "processing", "item": "USB-C Cable",
              "eta": "2026-08-05", "tracking": None},
    "A1003": {"status": "delivered", "item": "Laptop Stand",
              "eta": "2026-07-25", "tracking": "1Z999AA10987654321"},
}

INVENTORY = {
    "Wireless Headphones": 42,
    "USB-C Cable": 0,          # out of stock — a good edge case to watch
    "Laptop Stand": 17,
    "Mechanical Keyboard": 8,
}


def lookup_order(order_id: str) -> dict:
    """Return the order record for `order_id`, or an error if not found."""
    # .get() returns None if the key is missing, so we can give a clean message
    # instead of crashing. Tools should return structured, predictable results.
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"error": f"No order found with ID {order_id}."}
    return order


def check_inventory(item_name: str) -> dict:
    """Return how many of `item_name` are in stock."""
    count = INVENTORY.get(item_name)
    if count is None:
        return {"error": f"'{item_name}' is not a product we carry."}
    return {"item": item_name, "in_stock": count, "available": count > 0}


# A registry mapping tool NAME -> the Python function that implements it.
# When Claude says "call lookup_order", we look the name up here and run it.
TOOL_FUNCTIONS = {
    "lookup_order": lookup_order,
    "check_inventory": check_inventory,
}


# ============================================================================
# PART 2 — TOOL DEFINITIONS: describing the tools to Claude
# ============================================================================
# Claude can't see our Python functions. We describe each one with a JSON
# schema: a name, a description (Claude reads this to decide WHEN to use it),
# and the inputs it accepts. Good descriptions = good tool selection. This is a
# real skill: vague descriptions make the agent call the wrong tool.
TOOLS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up the status, item, ETA, and tracking number for a customer "
            "order. Use this whenever the customer references an order ID "
            "(format like A1001)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. 'A1001'.",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "check_inventory",
        "description": (
            "Check how many units of a product are currently in stock. Use this "
            "when the customer asks about availability or whether they can buy "
            "or reorder something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The exact product name, e.g. 'USB-C Cable'.",
                },
            },
            "required": ["item_name"],
        },
    },
]


# ============================================================================
# PART 3 — THE AGENT LOOP: the heart of the project
# ============================================================================
def run_agent(user_question: str, client: anthropic.Anthropic,
              verbose: bool = True) -> str:
    """
    Answer `user_question` by letting Claude call tools until it can respond.

    Returns Claude's final text answer.
    """
    system_prompt = (
        "You are a helpful customer-support agent for an online electronics "
        "store. Use the provided tools to look up real order and inventory "
        "information before answering — never invent order statuses, ETAs, or "
        "stock levels. If a tool reports an error or you cannot resolve the "
        "request, apologize and tell the customer you'll escalate to a human "
        "agent. Be concise and friendly."
    )

    # `messages` is the running conversation. We START with just the user's
    # question, and the loop appends to it as Claude calls tools and we reply
    # with results. Remember the API is stateless — we resend the whole
    # conversation every turn, which is why we keep growing this list.
    messages = [{"role": "user", "content": user_question}]

    # Safety cap: never loop more than N times, so a misbehaving agent can't
    # spin forever (and rack up API cost). Real agents always bound the loop.
    max_turns = 6

    for turn in range(max_turns):
        # --- STEP 1: ask Claude, giving it the tools it's allowed to use ---
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,          # <-- this is what makes it an agent
            messages=messages,
        )

        # --- STEP 2: did Claude ask to use a tool, or is it done? ---
        # `stop_reason` tells us WHY Claude stopped. "tool_use" means it wants
        # us to run one or more tools. "end_turn" means it produced a final
        # answer and is finished.
        if response.stop_reason != "tool_use":
            # Claude is done. Extract and return the final text answer.
            final = "".join(b.text for b in response.content if b.type == "text")
            return final

        # --- STEP 3: Claude wants tools. Run them and collect the results. ---
        # IMPORTANT: we must append Claude's response (the assistant turn,
        # including its tool_use blocks) to the conversation BEFORE we send the
        # results back — the API requires the tool_result to reference the
        # matching tool_use by id.
        messages.append({"role": "assistant", "content": response.content})

        # A single response can contain MULTIPLE tool_use blocks (Claude can ask
        # to call several tools at once — "parallel tool use"). We run each one.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name        # e.g. "lookup_order"
            tool_input = block.input      # e.g. {"order_id": "A1001"}
            tool_id = block.id            # unique id linking call <-> result

            if verbose:
                print(f"  [agent] calling {tool_name}({json.dumps(tool_input)})")

            # Find the real Python function and run it with Claude's arguments.
            # `**tool_input` unpacks the dict into keyword args, so
            # {"order_id": "A1001"} becomes lookup_order(order_id="A1001").
            func = TOOL_FUNCTIONS[tool_name]
            result = func(**tool_input)

            if verbose:
                print(f"  [tool ] -> {json.dumps(result)}")

            # Package the result as a tool_result block. `tool_use_id` MUST
            # match the id of the call, so Claude knows which result is which.
            # `content` must be a string, so we JSON-encode the dict.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(result),
            })

        # --- STEP 4: send ALL tool results back in one user message ---
        # (All results for a turn go in a single user message — splitting them
        # across multiple messages confuses the model.)
        messages.append({"role": "user", "content": tool_results})
        # Loop back to STEP 1: Claude now sees the results and continues.

    # If we somehow exhausted max_turns without a final answer, fail gracefully.
    return ("I'm sorry, I wasn't able to resolve that automatically. "
            "Let me escalate you to a human agent.")


# ============================================================================
# DEMO / ENTRY POINT
# ============================================================================
def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    client = anthropic.Anthropic()

    # A range of questions that exercise different tool-use paths:
    questions = [
        "Where is my order A1001?",                          # one tool call
        "Is the USB-C Cable in stock? My order is A1002.",   # TWO tool calls
        "Can I reorder a Mechanical Keyboard?",              # inventory only
        "What's the status of order A9999?",                 # not found -> escalate
    ]

    for q in questions:
        print("=" * 70)
        print(f"Customer: {q}")
        answer = run_agent(q, client)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
