"""
Smart Document Q&A — a Retrieval-Augmented Generation (RAG) engine.

WHAT THIS PROGRAM DOES (the 30-second version)
-----------------------------------------------
You give it a text document and ask a question. Instead of stuffing the WHOLE
document into the AI (slow, expensive, and impossible for big docs), it:

  1. CHUNKS    the document into small passages.
  2. EMBEDS    each passage into a vector (a list of numbers that captures meaning).
  3. RETRIEVES the few passages most relevant to your question (vector similarity).
  4. GENERATES an answer with Claude, given ONLY those relevant passages.

That 4-step pipeline IS retrieval-augmented generation. It's the single most
common pattern companies pay engineers to build right now, so understanding
every line here is high-leverage.

We use the Anthropic SDK for step 4 (the AI answer). Steps 1-3 are written in
pure Python (no numpy, no ML library) so you can SEE the mechanics — in
production you'd swap step 2 for a real embedding model (a one-line change,
marked below).
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
# Standard library only for retrieval — nothing to install.
import math                 # for sqrt and log, used in the vector math
import re                   # regular expressions, for splitting text into words
import os                   # to read the API key from an environment variable
from collections import Counter   # a dict subclass that counts things for us

# The Anthropic SDK — this is the official library for talking to Claude.
# `pip install anthropic` puts it on your machine.
import anthropic


# ----------------------------------------------------------------------------
# STEP 1 — CHUNKING: break a big document into small, searchable passages
# ----------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` WORDS each.

    WHY CHUNK AT ALL?
      - Retrieval works better on small passages. If a chunk is a whole chapter,
        a match tells you "the answer is somewhere in this chapter" — not helpful.
        Small chunks pinpoint the exact relevant sentence or two.
      - The AI has a context-window limit. We can't paste a 500-page PDF in.

    WHY OVERLAP?
      - A sentence answering the question might straddle a chunk boundary.
        Overlapping chunks (each chunk repeats the last `overlap` words of the
        previous one) make sure no idea gets cut in half and lost.

    Args:
        text:       the full document as one string.
        chunk_size: target number of words per chunk.
        overlap:    how many words each chunk shares with the previous chunk.

    Returns:
        a list of chunk strings.
    """
    # `text.split()` breaks the string on any whitespace into a list of words.
    words = text.split()

    chunks = []          # we'll accumulate chunks here
    start = 0            # index of the first word of the current chunk

    # Keep slicing until we've consumed every word.
    while start < len(words):
        # Take `chunk_size` words starting at `start`.
        # Python slicing is safe past the end — words[start:start+chunk_size]
        # just stops at the list end if there aren't enough words left.
        end = start + chunk_size
        chunk_words = words[start:end]

        # Join the word-list back into a single string with spaces between words.
        chunks.append(" ".join(chunk_words))

        # Advance the window. We move forward by (chunk_size - overlap) so the
        # next chunk re-includes `overlap` words from the end of this one.
        # Example: chunk_size=80, overlap=20 -> we step forward 60 words, so the
        # last 20 words of chunk N are also the first 20 words of chunk N+1.
        start += chunk_size - overlap

    return chunks


# ----------------------------------------------------------------------------
# STEP 2 — EMBEDDING: turn each chunk (and the question) into a vector
# ----------------------------------------------------------------------------
# An "embedding" is just a list of numbers representing a piece of text. Texts
# with similar meaning get similar vectors. We compare vectors to find relevant
# chunks.
#
# We implement a classic, transparent embedding called TF-IDF. Real production
# systems use a neural embedding model (Voyage, OpenAI, or a local
# sentence-transformer) that captures deeper semantic meaning — but TF-IDF is
# real, it works, and crucially you can READ exactly what it's doing.
#
# TF-IDF = Term Frequency × Inverse Document Frequency:
#   - Term Frequency (TF): how often a word appears in THIS chunk. A chunk that
#     says "firewall" five times is probably about firewalls.
#   - Inverse Document Frequency (IDF): how RARE a word is across ALL chunks.
#     "the" appears everywhere, so it carries no signal — IDF down-weights it.
#     "firewall" appears in few chunks, so it's a strong topical signal —
#     IDF up-weights it.
# Multiply them and you get a number per word that says "how important is this
# word, for this chunk, relative to the whole document."


def tokenize(text: str) -> list[str]:
    r"""
    Turn a string into a clean list of lowercase words ("tokens").

    re.findall(r"\b\w+\b", ...) finds every run of "word characters"
    (letters, digits, underscore). \b is a word boundary. So "Hello, world!"
    becomes ["hello", "world"] — punctuation dropped, everything lowercased so
    "Firewall" and "firewall" count as the same word.
    """
    return re.findall(r"\b\w+\b", text.lower())


def compute_idf(chunks: list[str]) -> dict[str, float]:
    """
    Compute the IDF (rarity) score for every word across all chunks.

    Returns a dict mapping each word -> its IDF score (a float).
    Higher score = rarer word = more useful for distinguishing chunks.
    """
    num_chunks = len(chunks)

    # doc_count[word] = how many chunks contain `word` at least once.
    doc_count: Counter = Counter()
    for chunk in chunks:
        # set(...) removes duplicates, so a word appearing 5 times in one chunk
        # still only counts as "present in 1 chunk."
        unique_words = set(tokenize(chunk))
        for word in unique_words:
            doc_count[word] += 1

    # The IDF formula. The classic version is log(N / df). We use
    # log(N / (1 + df)) + 1 — the "+1"s prevent division-by-zero and keep every
    # score positive. The exact constants matter less than the shape: rarer
    # words (small df) get bigger scores.
    idf = {}
    for word, df in doc_count.items():
        idf[word] = math.log(num_chunks / (1 + df)) + 1.0
    return idf


def embed(text: str, idf: dict[str, float]) -> dict[str, float]:
    """
    Embed one piece of text into a TF-IDF vector.

    We represent the vector as a DICT {word: weight} instead of a fixed-length
    list. This is a "sparse" vector — we only store words that actually appear,
    which is memory-efficient since any single chunk uses few of the document's
    total vocabulary. Mathematically it's identical to a giant list where every
    other word is 0.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}

    # Term Frequency: count each word, then divide by total words so that long
    # and short texts are comparable (we want PROPORTIONS, not raw counts).
    counts = Counter(tokens)
    total = len(tokens)

    vector = {}
    for word, count in counts.items():
        tf = count / total
        # idf.get(word, 0.0): if the word never appeared in the indexed document
        # (e.g. a typo in the question), it has no IDF and contributes nothing.
        vector[word] = tf * idf.get(word, 0.0)
    return vector


# ----------------------------------------------------------------------------
# STEP 3 — RETRIEVAL: find the chunks whose vectors are closest to the question
# ----------------------------------------------------------------------------
def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """
    Measure how similar two vectors are, from 0.0 (nothing in common) to
    1.0 (identical direction).

    COSINE SIMILARITY measures the ANGLE between two vectors, ignoring their
    length. We care about direction (which words matter), not magnitude (how
    long the text is). Formula:

        cosine(A, B) = (A · B) / (|A| × |B|)

    where:
        A · B   is the "dot product": multiply matching components, sum them up.
        |A|     is the "magnitude": sqrt of the sum of squares of A's components.
    """
    # Dot product: for every word the two vectors share, multiply the two
    # weights and add to the running total. Words in only one vector contribute
    # 0 (you can't multiply a value by a missing one), so we only loop over the
    # smaller vector's keys for efficiency.
    #
    # Decide which dict is smaller so we iterate the shorter one.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = 0.0
    for word, weight_a in vec_a.items():
        if word in vec_b:
            dot += weight_a * vec_b[word]

    # Magnitude of each vector = sqrt(sum of each weight squared).
    mag_a = math.sqrt(sum(w * w for w in vec_a.values()))
    mag_b = math.sqrt(sum(w * w for w in vec_b.values()))

    # Guard against divide-by-zero: an empty vector has magnitude 0.
    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def retrieve(question: str, chunks: list[str], chunk_vectors: list[dict],
             idf: dict[str, float], top_k: int = 3) -> list[tuple[int, float, str]]:
    """
    Return the `top_k` chunks most relevant to `question`.

    Args:
        question:      the user's question string.
        chunks:        the list of chunk strings.
        chunk_vectors: pre-computed TF-IDF vector for each chunk (same order).
        idf:           the IDF scores, so we embed the question the same way.
        top_k:         how many chunks to return.

    Returns:
        a list of (chunk_index, similarity_score, chunk_text) tuples,
        sorted best-first.
    """
    # Embed the question into the SAME vector space as the chunks. This is the
    # key trick: question and chunks must be embedded identically so their
    # vectors are comparable.
    question_vector = embed(question, idf)

    # Score every chunk against the question.
    scored = []
    for i, chunk_vector in enumerate(chunk_vectors):
        score = cosine_similarity(question_vector, chunk_vector)
        scored.append((i, score, chunks[i]))

    # Sort by score, highest first. key=lambda x: x[1] means "sort by the second
    # item of each tuple (the score)". reverse=True makes it descending.
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return only the best `top_k`.
    return scored[:top_k]


# ----------------------------------------------------------------------------
# STEP 4 — GENERATION: ask Claude to answer using ONLY the retrieved chunks
# ----------------------------------------------------------------------------
def generate_answer(question: str, retrieved: list[tuple[int, float, str]],
                    client: anthropic.Anthropic) -> str:
    """
    Send the question + retrieved passages to Claude and get a grounded answer.

    THE CORE RAG IDEA: we don't ask Claude to answer from its own memory (it
    might hallucinate or not know your private document). Instead we paste the
    relevant passages into the prompt and say "answer using ONLY this." This
    keeps answers accurate and lets you cite sources.
    """
    # Build a "context" string: the retrieved passages, each labelled so Claude
    # (and we) can cite them by number.
    context_blocks = []
    for rank, (chunk_index, score, chunk_text) in enumerate(retrieved, start=1):
        context_blocks.append(f"[Passage {rank}] {chunk_text}")
    context = "\n\n".join(context_blocks)

    # The system prompt sets Claude's role and rules. Keeping it stable (no
    # timestamps/IDs) also means it can be prompt-cached later for cost savings.
    system_prompt = (
        "You are a precise document Q&A assistant. Answer the user's question "
        "using ONLY the provided passages. If the passages do not contain the "
        "answer, say so plainly — do not guess or use outside knowledge. "
        "When you use a passage, cite it like [Passage 1]."
    )

    # The user message contains the context and the actual question.
    user_message = (
        f"Here are the relevant passages from the document:\n\n{context}\n\n"
        f"Question: {question}"
    )

    # THE API CALL. This is the one network request to Claude.
    #   - model:      which Claude to use. claude-opus-4-8 is the current,
    #                 most capable model. (See README for cheaper options.)
    #   - max_tokens: the maximum length of Claude's reply, in tokens
    #                 (~3/4 of a word each). 1024 is plenty for a Q&A answer.
    #   - system:     the role/rules string from above.
    #   - messages:   the conversation. A list of {role, content} dicts. The API
    #                 is stateless, so for multi-turn chat you'd resend history.
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message},
        ],
    )

    # `response.content` is a LIST of content blocks (Claude can return text,
    # tool calls, thinking, etc.). For a plain answer we want the text blocks.
    # We collect every block whose .type is "text" and join them.
    answer_parts = [block.text for block in response.content if block.type == "text"]
    return "".join(answer_parts)


# ----------------------------------------------------------------------------
# PUTTING IT ALL TOGETHER — a small class that owns the indexed document
# ----------------------------------------------------------------------------
class DocumentQA:
    """
    Wraps the whole pipeline. You build it once from a document (the "indexing"
    work happens here, up front), then call .ask(question) as many times as you
    like — each call only does the cheap retrieval + one Claude request.
    """

    def __init__(self, document_text: str):
        # The Anthropic() client reads your API key from the ANTHROPIC_API_KEY
        # environment variable automatically. Never hardcode the key in code.
        self.client = anthropic.Anthropic()

        # --- INDEX THE DOCUMENT (do the expensive prep once) ---
        # 1. Chunk it.
        self.chunks = chunk_text(document_text)
        # 2. Compute IDF across all chunks (needs to see every chunk first).
        self.idf = compute_idf(self.chunks)
        # 3. Embed every chunk into a vector and cache the vectors.
        self.chunk_vectors = [embed(chunk, self.idf) for chunk in self.chunks]

    def ask(self, question: str, top_k: int = 3, show_sources: bool = True) -> str:
        """Answer one question against the indexed document."""
        # Retrieve the most relevant chunks.
        retrieved = retrieve(question, self.chunks, self.chunk_vectors,
                             self.idf, top_k=top_k)

        # Generate the grounded answer.
        answer = generate_answer(question, retrieved, self.client)

        # Optionally show which passages were used (great for debugging and for
        # demoing that the answer is actually grounded in the document).
        if show_sources:
            sources = "\n".join(
                f"  [Passage {rank}] (similarity {score:.3f}): {text[:90]}..."
                for rank, (idx, score, text) in enumerate(retrieved, start=1)
            )
            return f"{answer}\n\n--- Retrieved passages ---\n{sources}"
        return answer


# ----------------------------------------------------------------------------
# DEMO / ENTRY POINT
# ----------------------------------------------------------------------------
# This block runs only when you execute the file directly (`python rag.py`),
# not when another file imports it. `__name__` is "__main__" only for the file
# you run directly — a standard Python idiom.
def main():
    # Fail early with a friendly message if the key isn't set, instead of a
    # confusing error deep inside the SDK.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        print("See README.md for how to get a key.")
        return

    # Load the sample document that ships next to this file.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sample_document.txt"), "r") as f:
        document = f.read()

    print("Indexing document...")
    qa = DocumentQA(document)
    print(f"Done. Document split into {len(qa.chunks)} chunks.\n")

    # A few sample questions to show it working.
    questions = [
        "What is a forward deployed engineer?",
        "What skills does an FDE need?",
        "What is the capital of France?",  # NOT in the doc — tests honesty
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {qa.ask(q)}\n")
        print("=" * 70)


if __name__ == "__main__":
    main()
