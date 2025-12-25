import os
import json
import streamlit as st
from openai import OpenAI

# ---------------------------
# App Setup
# ---------------------------
st.set_page_config(page_title="Mapping Moral Logic", page_icon="🧭", layout="centered")
st.title("🧭 Mapping Moral Logic")
st.caption("Surfaces the assumptions and values that make people’s views make sense to them.")

# ---------------------------
# Safe load of library.json
# ---------------------------
LIBRARY = {}
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "library.json")

def load_library(path: str):
    if not os.path.exists(path):
        st.sidebar.error(f"`library.json` not found at: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # Peek at the first 300 chars for debugging
        st.sidebar.markdown("**Loaded `library.json` (first 300 chars):**")
        st.sidebar.code(raw[:300])
        return json.loads(raw)
    except json.JSONDecodeError as e:
        st.sidebar.error(
            "Your `library.json` is not valid JSON.\n\n"
            f"Line {e.lineno}, column {e.colno}: {e.msg}"
        )
        return {}
    except Exception as e:
        st.sidebar.error(f"Error reading library.json: {e}")
        return {}

LIBRARY = load_library(LIBRARY_PATH)
st.sidebar.markdown("**Library sections detected:**")
if isinstance(LIBRARY, dict):
    st.sidebar.write(list(LIBRARY.keys()))
    st.sidebar.metric("principles", len(LIBRARY.get("general_principles", [])))
    st.sidebar.metric("entries", len(LIBRARY.get("entries", [])))
else:
    st.sidebar.write("No library loaded")

# ---------------------------
# API Setup
# ---------------------------
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Set the OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key or "missing")

# ---------------------------
# Usage guard (per-session)
# ---------------------------
if "requests_today" not in st.session_state:
    st.session_state.requests_today = 0
DAILY_LIMIT = 50  # adjust for your budget

# ---------------------------
# Model + options
# ---------------------------
col1, col2 = st.columns([2, 1])
with col1:
    model = st.selectbox(
        "Model",
        ["gpt-4.1-mini", "gpt-3.5-turbo"],
        index=0,
        help="Use gpt-4.1-mini for best quality per $. gpt-3.5-turbo is cheaper but weaker."
    )
with col2:
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05)

# ---------------------------
# System Prompt (behavior)
# ---------------------------
SYSTEM_PROMPT = """
This GPT, called Mapping Moral Logic, helps users surface the assumptions and values that make people’s views make sense to them. Modeled on Ilana Redstone’s reflective approach, it uses a calm, plainspoken tone to reveal the reasoning paths beneath polarized or strongly held views.

How it works:
1. The user types a statement, judgment, or belief to unpack. The GPT helps surface the assumptions and values beneath it.
2. It first checks a library of general principles and topic entries (e.g., fairness, autonomy, equity, consistency, humility) for a close match to the user’s input.
3. If a close match exists, it paraphrases the relevant reflection and probing questions in Ilana Redstone’s style, prefacing them with “Library-based:”.
4. If no close match exists, it infers the underlying assumptions and values from context, prefacing the response with “Inferred (no close library match):”.

Response structure:
1. One or two sentences calmly paraphrasing the reasoning or assumption beneath the statement.
2. A blank line.
3. One or two short, open-ended questions that invite thought rather than argument.

Tone and purpose:
- Calm, lightly challenging, and plainspoken.
- Invites reflection, not agreement.
- Avoids moralizing, diagnosing, or emotionally balancing language.
- Focuses on surfacing principles, goals, and assumptions beneath moral or factual claims.
- Treats each position as a morally legitimate destination that people can reach through different reasoning paths.

Transparency:
- Clearly label whether a response is library-based or inferred.
- Do not display internal topic names or metadata; paraphrase naturally.

Goal:
Help users make their reasoning visible by uncovering the values, goals, and assumptions beneath judgments or claims—without taking sides or prescribing conclusions.
"""

# ---------------------------
# UI
# ---------------------------
prompt = st.text_area(
    "Type a statement, judgment, or belief you’d like to unpack:",
    placeholder="e.g., 'Universities should ban controversial speakers to keep students safe.'",
    height=150
)

colA, colB = st.columns([1, 1])
with colA:
    go = st.button("Unpack")
with colB:
    reset = st.button("Reset")

if reset:
    st.session_state.clear()
    st.rerun()

# ---------------------------
# Helper: match library category
# ---------------------------
import numpy as np

# ---------------------------
# Embedding utilities
# ---------------------------
EMBED_MODEL = "text-embedding-3-small"

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@st.cache_resource(show_spinner=False)
def build_library_embeddings(lib: dict):
    """Create and cache embeddings for each entry and principle."""
    if not lib or not isinstance(lib, dict):
        return []

    items = []
    for section, entries in [("entry", lib.get("entries", [])),
                             ("principle", lib.get("general_principles", []))]:
        for e in entries or []:
            text_bits = []
            if section == "entry":
                text_bits += [
                    e.get("topic", ""),
                    " ".join(e.get("tags", []) or []),
                ]
                sc = e.get("sample_claim", [])
                if isinstance(sc, str):
                    sc = [sc]
                text_bits += sc or []
            else:  # principle
                text_bits.append(e.get("name", ""))
                text_bits.append(e.get("description", ""))

            joined = " ".join(text_bits).strip()
            if not joined:
                continue

            try:
                emb = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=joined
                ).data[0].embedding
                items.append({
                    "kind": section,
                    "item": e,
                    "embedding": emb
                })
            except Exception as ex:
                st.sidebar.warning(f"Embedding failed for {joined[:40]}... ({ex})")

    return items

# Build embeddings once
LIBRARY_EMBEDS = build_library_embeddings(LIBRARY)
def match_category(text: str, threshold: float = 0.70):
    """
    Find the closest semantic match in the library using cosine similarity.
    Returns ('entry'|'principle', dict) or None if no good match.
    """
    if not text or not LIBRARY_EMBEDS:
        return None

    try:
        q_emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        ).data[0].embedding
    except Exception as e:
        st.sidebar.warning(f"Embedding query failed: {e}")
        return None

    best = None
    best_score = -1
    for rec in LIBRARY_EMBEDS:
        sim = cosine_similarity(q_emb, rec["embedding"])
        if sim > best_score:
            best_score = sim
            best = rec

    if best and best_score >= threshold:
        st.sidebar.info(f"✅ Library match ({best['kind']}) — similarity: {best_score:.2f}")
        return (best["kind"], best["item"])
    else:
        st.sidebar.info(f"❌ No close library match (best={best_score:.2f})")
        return None
# ---------------------------
# Handle submit
# ---------------------------
if go:
    if st.session_state.requests_today >= DAILY_LIMIT:
        st.error("Daily limit reached. Please try again tomorrow.")
    elif not prompt.strip():
        st.warning("Please enter a statement to unpack.")
    else:
        st.session_state.requests_today += 1

        with st.spinner("Thinking…"):
            user_input = prompt.strip()
            if len(user_input) > 4000:
                user_input = user_input[:4000] + "\n\n[Truncated for length]"

            # Build dynamic library hint
            match = match_category(user_input)
            if match:
                kind, item = match
                if kind == "entry":
                    label = f"Library-based ({item.get('topic', 'entry')})"
                    qs = item.get("questions", []) or []
                    questions_hint = " ".join([q for q in qs[:2] if isinstance(q, str)])

                    reflection_hint = item.get("reflection", "")
                    if isinstance(reflection_hint, list):
                        reflection_hint = " ".join([x for x in reflection_hint[:2] if isinstance(x, str)])

                    library_hint = (
                        f"{label}: Use the entry’s spirit. Paraphrase briefly, then ask one or two open-ended questions "
                        f"in this spirit: {questions_hint}. "
                        f"(Background for you, not to quote: {reflection_hint})"
                    )
                else:  # principle
                    label = f"Library-based (principle: {item.get('name','')})"
                    desc = item.get("description", "")
                    library_hint = (
                        f"{label}: Ground the reflection in this principle’s description. "
                        f"(Background for you, not to quote: {desc})"
                    )
            else:
                label = "Inferred (no close library match)"
                library_hint = f"{label}: No obvious library match; proceed normally."

            # Call OpenAI with both the system prompt and the dynamic library hint
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "system", "content": library_hint},
                        {"role": "user", "content": user_input}
                    ]
                )
                output = resp.choices[0].message.content
            except Exception as e:
                output = f"Error: {e}"

        # Display result
        st.subheader("Reflection")
        st.write(output)

        # Minimal usage meter
        st.caption(f"Requests this session: {st.session_state.requests_today}/{DAILY_LIMIT}")
        st.caption("Note: You are not billed; the app owner covers API costs.")
