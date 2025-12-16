import os
import time
import streamlit as st
from openai import OpenAI

# --- App Setup ---
st.set_page_config(page_title="Mapping Moral Logic", page_icon="🧭", layout="centered")
st.title("🧭 Mapping Moral Logic")
st.caption("Surfaces the assumptions and values that make people’s views make sense to them.")

# --- API Setup ---
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Set the OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key)

# --- Basic cost/abuse guard (per-session) ---
if "requests_today" not in st.session_state:
    st.session_state.requests_today = 0
DAILY_LIMIT = 50  # adjust for your budget

# --- Model + options ---
col1, col2 = st.columns([2,1])
with col1:
    model = st.selectbox(
        "Model",
        ["gpt-4.1-mini", "gpt-3.5-turbo"],
        index=0,
        help="Use gpt-4.1-mini for best quality per $; gpt-3.5-turbo is cheaper but weaker."
    )
with col2:
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05)

# --- System Prompt (your GPT’s behavior) ---
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

# --- UI ---
prompt = st.text_area(
    "Type a statement, judgment, or belief you’d like to unpack:",
    placeholder="e.g., 'Universities should ban controversial speakers to keep students safe.'",
    height=150
)

colA, colB = st.columns([1,1])
with colA:
    go = st.button("Unpack")
with colB:
    reset = st.button("Reset")

if reset:
    st.session_state.requests_today = 0
    st.experimental_rerun()

# --- Handle submit ---
if go:
    if st.session_state.requests_today >= DAILY_LIMIT:
        st.error("Daily limit reached. Please try again tomorrow.")
    elif not prompt.strip():
        st.warning("Please enter a statement to unpack.")
    else:
        st.session_state.requests_today += 1

        with st.spinner("Thinking…"):
            # Small input guard to avoid accidental long pastes
            user_input = prompt.strip()
            if len(user_input) > 4000:
                user_input = user_input[:4000] + "\n\n[Truncated for length]"

            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_input}
                    ]
                )
                output = resp.choices[0].message.content
            except Exception as e:
                output = f"Error: {e}"

        # --- Display result ---
        st.subheader("Reflection")
        st.write(output)

        # --- Minimal usage meter ---
        st.caption(f"Requests this session: {st.session_state.requests_today}/{DAILY_LIMIT}")
        st.caption("Note: You are not billed; the app owner covers API costs.")
