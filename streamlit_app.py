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
def match_category(text: str):
    """
    Try to match the user's text to either a specific 'entry'
    (via tags, sample_claim, or topic) or to a general principle by name.
    Returns a tuple ('entry'|'principle', dict) or None.
    """
    if not isinstance(LIBRARY, dict):
        return None

    t = (text or "").lower()

    # 1) Entries by tags, sample_claim, or topic
    for entry in LIBRARY.get("entries", []) or []:
        # tags
        for tag in entry.get("tags", []) or []:
            if isinstance(tag, str) and tag.lower() in t:
                return ("entry", entry)
        # sample_claim can be str or list
        sc = entry.get("sample_claim", [])
        if isinstance(sc, str):
            sc = [sc]
        for s in sc or []:
            if isinstance(s, str) and s.lower() in t:
                return ("entry", entry)
        # topic substring
        topic = entry.get("topic", "")
        if isinstance(topic, str) and topic.lower() in t:
            return ("entry", entry)

    # 2) Principles by name substring
    for gp in LIBRARY.get("general_principles", []) or []:
        name = gp.get("name", "")
        if isinstance(name, str) and name.lower() in t:
            return ("principle", gp)

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
