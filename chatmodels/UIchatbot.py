import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "labs-leanstral-1-5"
TEMPERATURE = 0.9
APP_NAME = "AI Assistant"

system_instruction = (
    "You are a highly capable, concise, and helpful AI Assistant.\n"
    "Your core objective is to provide clear, direct, and factually accurate answers.\n\n"
    "CRITICAL BEHAVIORAL RULES:\n"
    "1. Be direct: Avoid conversational filler, pleasantries, or repeating the user's question.\n"
    "2. Be concise: Keep answers brief and focused unless deeply analytical details are requested.\n"
    "3. Format cleanly: Use bold text, bullet points, and short paragraphs to make answers easily scannable.\n"
    "4. Code formatting: When writing code blocks, always specify the language syntax (e.g., ```python) and include brief comments.\n"
    "5. Humility: If you do not know the answer, say 'I don't know' instead of hallucinating or making up facts."
)

SUGGESTIONS = [
    ("Explain a concept", ":material/lightbulb:",
     "Explain how neural networks learn, in simple terms."),
    ("Help me write code", ":material/code:",
     "Help me write a Python function that removes duplicates from a list while keeping the order."),
    ("Analyze an idea", ":material/insights:",
     "Help me analyze the strengths and weaknesses of a startup idea."),
    ("Learn something new", ":material/school:",
     "Teach me something fascinating I probably don't know yet."),
]

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="auto",
)

logger = logging.getLogger("chat_app")


# ────────────────────────────────────────────────────────────────────────────
# Model initialization (same model / temperature as before)
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatMistralAI(model=MODEL_NAME, temperature=TEMPERATURE)


llm = get_llm()


# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────
def reset_chat():
    st.session_state.messages = [SystemMessage(content=system_instruction)]
    st.session_state.pending_prompt = None
    st.session_state.failed_prompt = None


def queue_prompt(text: str):
    """Used by suggestion cards and the retry button."""
    st.session_state.pending_prompt = text
    st.session_state.failed_prompt = None


if "messages" not in st.session_state:
    reset_chat()

messages = st.session_state.messages


# ────────────────────────────────────────────────────────────────────────────
# Chat logic (unchanged flow: HumanMessage -> invoke -> AIMessage)
# ────────────────────────────────────────────────────────────────────────────
def generate_reply(prompt: str):
    """Returns the reply text, or None if the model call failed."""
    messages.append(HumanMessage(content=prompt))
    try:
        response = llm.invoke(messages)
    except Exception:
        # Logged server-side only; the user never sees tracebacks or keys.
        logger.exception("Model invocation failed")
        messages.pop()  # keep history valid (no dangling HumanMessage)
        st.session_state.failed_prompt = prompt
        return None
    messages.append(AIMessage(content=response.content))
    return response.content


# ────────────────────────────────────────────────────────────────────────────
# CSS — edit the variables in :root to re-theme the whole app
# ────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:          #0e0f13;
  --bg-elev:     #13151a;
  --surface:     #181a21;
  --surface-2:   #1e2129;
  --border:      rgba(255,255,255,0.07);
  --border-hi:   rgba(255,255,255,0.14);
  --text:        #ececf1;
  --text-muted:  #9a9cab;
  --text-faint:  #6c6e7d;
  --accent:      #8b93ff;
  --accent-soft: rgba(139,147,255,0.14);
  --radius:      14px;
  --font:        'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono:        'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}

/* ── App shell ───────────────────────────────────────────── */
html, body, .stApp { font-family: var(--font); }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 46rem; padding: 1.25rem 1.25rem 9rem; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--bg-elev); border-right: 1px solid var(--border); }
[data-testid="stSidebarUserContent"] { padding-top: 1.25rem; }

.brand { display:flex; align-items:center; gap:.65rem; margin-bottom:1.25rem; }
.brand-mark {
  width:32px; height:32px; border-radius:9px; display:grid; place-items:center;
  background: var(--accent-soft); color: var(--accent);
  border:1px solid rgba(139,147,255,.3); font-size:16px;
}
.brand-name { font-weight:600; font-size:.98rem; letter-spacing:-.01em; color:var(--text); }

.side-label {
  font-size:.68rem; text-transform:uppercase; letter-spacing:.09em;
  color:var(--text-faint); margin:1.5rem 0 .5rem; font-weight:500;
}
.side-row { display:flex; justify-content:space-between; gap:.75rem; font-size:.83rem; color:var(--text-muted); padding:.3rem 0; }
.side-row b { color:var(--text); font-weight:500; font-family: var(--mono); font-size:.76rem; text-align:right; }
.side-note { font-size:.8rem; line-height:1.6; color:var(--text-faint); }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  background: var(--surface); color: var(--text);
  border: 1px solid var(--border); border-radius: 12px;
  padding: .7rem 1rem; font-size: .9rem; font-weight: 500;
  transition: border-color .15s, background .15s, transform .15s;
  width: 100%; justify-content: flex-start;
}
.stButton > button:hover { background: var(--surface-2); border-color: var(--border-hi); color: var(--text); }
.stButton > button:active { transform: scale(.985); }
.stButton > button:focus:not(:active) { border-color: var(--accent); color: var(--text); }

/* ── Top bar ─────────────────────────────────────────────── */
.topbar {
  display:flex; align-items:center; justify-content:space-between;
  padding:.35rem 0 1rem; margin-bottom:.5rem; border-bottom:1px solid var(--border);
}
.topbar-title { font-weight:600; font-size:.95rem; letter-spacing:-.01em; }
.pill {
  display:inline-flex; align-items:center; gap:.45rem; font-size:.74rem;
  color:var(--text-muted); padding:.25rem .65rem; border-radius:999px;
  background: var(--surface); border:1px solid var(--border);
}
.pill .dot { width:6px; height:6px; border-radius:50%; background:#4ade80; box-shadow:0 0 0 3px rgba(74,222,128,.15); }

/* ── Empty state ─────────────────────────────────────────── */
.hero { text-align:center; padding: 12vh 0 2rem; }
.hero h1 {
  font-size: clamp(1.9rem, 4.5vw, 2.6rem); font-weight:600;
  letter-spacing:-.03em; margin:0 0 .7rem; color:var(--text); padding:0;
}
.hero p { color: var(--text-muted); font-size:1.02rem; margin:0; line-height:1.6; }

/* ── Messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent; border: none; padding: 1rem 0; gap: .9rem; align-items: flex-start;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--surface); border:1px solid var(--border);
  border-radius: var(--radius); padding: .85rem 1.1rem; margin: .6rem 0;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  width:28px; height:28px; border-radius:8px; background: var(--surface-2);
  color: var(--text-muted); border:1px solid var(--border);
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--accent-soft); color: var(--accent); border-color: rgba(139,147,255,.3);
}
[data-testid="stChatMessageContent"] { color: var(--text); font-size:.95rem; line-height:1.72; }
[data-testid="stChatMessageContent"] p { margin: 0 0 .75rem; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol { padding-left: 1.3rem; margin: .25rem 0 .75rem; }
[data-testid="stChatMessageContent"] li { margin-bottom: .3rem; }
[data-testid="stChatMessageContent"] strong { color:#fff; font-weight:600; }
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 { letter-spacing:-.02em; font-weight:600; margin:1.1rem 0 .5rem; padding:0; }
[data-testid="stChatMessageContent"] a { color: var(--accent); }

[data-testid="stChatMessageContent"] code {
  font-family: var(--mono); font-size:.84em; background: var(--surface-2);
  border:1px solid var(--border); border-radius:6px; padding:.12em .4em; color:#d7d9ff;
}
[data-testid="stChatMessageContent"] pre {
  background:#0a0b0f !important; border:1px solid var(--border);
  border-radius:12px; padding:1rem 1.1rem; margin:.5rem 0 1rem; overflow-x:auto;
}
[data-testid="stChatMessageContent"] pre code {
  background:transparent; border:none; padding:0; font-size:.83rem; line-height:1.65; color:#dcdde6;
}

/* ── Thinking indicator ──────────────────────────────────── */
.thinking { display:inline-flex; align-items:center; gap:.5rem; color:var(--text-muted); font-size:.92rem; padding:.2rem 0; }
.thinking i {
  width:6px; height:6px; border-radius:50%; background:var(--accent);
  display:inline-block; animation: pulse 1.2s infinite ease-in-out;
}
.thinking i:nth-child(2) { animation-delay:.15s; }
.thinking i:nth-child(3) { animation-delay:.3s; }
.thinking span { margin-left:.25rem; }
@keyframes pulse { 0%,80%,100% { opacity:.25; transform:scale(.8);} 40% { opacity:1; transform:scale(1);} }

/* ── Error card ──────────────────────────────────────────── */
.err {
  border:1px solid rgba(248,113,113,.3); background: rgba(248,113,113,.07);
  border-radius: var(--radius); padding: .9rem 1.1rem; margin: .5rem 0 .75rem;
  color:#fca5a5; font-size:.9rem; line-height:1.55;
}
.err b { color:#fecaca; font-weight:600; }

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stBottom"] { background: transparent; }
[data-testid="stBottom"] > div {
  background: linear-gradient(to top, var(--bg) 62%, rgba(14,15,19,0));
  padding-top: 1.5rem;
}
[data-testid="stBottomBlockContainer"] { max-width: 46rem; padding-bottom: 1.25rem; }
[data-testid="stChatInput"] {
  background: var(--surface); border:1px solid var(--border-hi);
  border-radius: 20px; box-shadow: 0 8px 30px rgba(0,0,0,.35);
  transition: border-color .15s, box-shadow .15s;
}
[data-testid="stChatInput"]:focus-within {
  border-color: rgba(139,147,255,.55);
  box-shadow: 0 8px 30px rgba(0,0,0,.35), 0 0 0 3px var(--accent-soft);
}
[data-testid="stChatInput"] > div { background: transparent; border: none; }
[data-testid="stChatInput"] textarea {
  color: var(--text); font-family: var(--font); font-size:.95rem; padding: .85rem .5rem .85rem 1rem;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-faint); }
[data-testid="stChatInputSubmitButton"] { color: var(--accent); }
[data-testid="stChatInputSubmitButton"]:disabled { color: var(--text-faint); }

/* ── Misc ────────────────────────────────────────────────── */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius:10px; border:2px solid var(--bg); }

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 768px) {
  .block-container { padding: 1rem .9rem 8rem; }
  .hero { padding-top: 7vh; }
  [data-testid="stChatMessage"] { gap: .65rem; }
  [data-testid="stChatMessageContent"] { font-size:.92rem; }
  [data-testid="stBottomBlockContainer"] { padding-left:.9rem; padding-right:.9rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# UI components
# ────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""<div class="brand"><div class="brand-mark">✦</div>
            <div class="brand-name">{APP_NAME}</div></div>""",
            unsafe_allow_html=True,
        )
        st.button("New chat", icon=":material/add:", on_click=reset_chat,
                  use_container_width=True, key="new_chat")

        n = len(messages) - 1  # exclude the system message
        st.markdown(
            f"""
            <div class="side-label">Session</div>
            <div class="side-row"><span>Messages</span><b>{n}</b></div>
            <div class="side-row"><span>Model</span><b>{MODEL_NAME}</b></div>
            <div class="side-row"><span>Provider</span><b>Mistral AI</b></div>
            <div class="side-label">About</div>
            <div class="side-note">
              A concise, direct assistant built with LangChain and Mistral.
              Conversation history is kept for this browser session only;
              starting a new chat clears it.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_topbar():
    st.markdown(
        f"""<div class="topbar">
        <div class="topbar-title">{APP_NAME}</div>
        <div class="pill"><span class="dot"></span>Mistral</div></div>""",
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """<div class="hero">
        <h1>How can I help you today?</h1>
        <p>Ask anything, explore ideas, write code, or solve problems.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, icon, prompt_text) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            st.button(label, icon=icon, key=f"sugg_{i}",
                      on_click=queue_prompt, args=(prompt_text,),
                      use_container_width=True)


def render_history():
    for m in messages:
        if isinstance(m, HumanMessage):
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(m.content)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(m.content)


def render_error():
    failed = st.session_state.failed_prompt
    if not failed:
        return
    st.markdown(
        """<div class="err"><b>Something went wrong.</b><br>
        The assistant couldn't respond just now, and your last message wasn't sent.
        Please check your connection or API configuration and try again.</div>""",
        unsafe_allow_html=True,
    )
    st.button("Retry", icon=":material/refresh:", key="retry",
              on_click=queue_prompt, args=(failed,))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
typed = st.chat_input("Message your AI assistant...")
prompt = st.session_state.pending_prompt or typed
st.session_state.pending_prompt = None

render_sidebar()
render_topbar()

if len(messages) == 1 and not prompt and not st.session_state.failed_prompt:
    render_empty_state()
else:
    render_history()

if prompt:
    st.session_state.failed_prompt = None

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        slot = st.empty()
        slot.markdown(
            '<div class="thinking"><i></i><i></i><i></i><span>Thinking...</span></div>',
            unsafe_allow_html=True,
        )
        reply = generate_reply(prompt)
        if reply is not None:
            slot.markdown(reply)

    # Redraw so the sidebar count updates (and the error card shows on failure)
    st.rerun()
else:
    render_error()