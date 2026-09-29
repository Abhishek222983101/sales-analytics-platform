"""Ask-your-data — a friendly chat grounded in the loaded dataset."""
import streamlit as st

from core import llm
from core.branding import APP_NAME
from core.nlp import answer, quick_facts

st.set_page_config(page_title=f"Ask · {APP_NAME}", page_icon="💬", layout="wide")

try:
    llm.bridge_secrets(st.secrets)
except Exception:
    pass

st.title("💬 Ask your data anything")

data = st.session_state.get("data")
if data is None:
    st.info("Pop over to 📄 Upload first and load some data — then ask away. 💛")
    st.page_link("pages/1_Upload.py", label="Go to Upload", icon="📄")
    st.stop()

# --- always-available computed facts ---------------------------------------
st.caption("A few quick facts to start with:")
facts = quick_facts(data)
cols = st.columns(len(facts))
for col, (label, value) in zip(cols, facts.items()):
    col.metric(label, value)

st.divider()

if not llm.available():
    st.info("Add a Groq/OpenAI key in secrets and I'll answer free-form questions here too. "
            "For now, the quick facts above are always available. 💛")
    st.stop()

# --- chat ------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state["chat"] = []

for role, text in st.session_state["chat"]:
    with st.chat_message(role):
        st.write(text)

prompt = st.chat_input("e.g. Which region is doing best, and why?")
if prompt:
    st.session_state["chat"].append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Looking through your data…"):
            reply = answer(prompt, data, st.session_state.get("insight"))
        reply = reply or "Sorry — I couldn't find that in your data. Try asking about " \
                         "revenue, regions, categories, or your best month. 🙈"
        st.write(reply)
    st.session_state["chat"].append(("assistant", reply))
