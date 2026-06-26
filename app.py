import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="BAMBOO TUNE", page_icon="🌱", layout="wide")

# ============================================================
# BACKEND - fine-tuned model 
# ============================================================
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER = "nuhaai/bamboo-tune-model" 
HF_TOKEN = os.environ.get("bambooo")
SYS = """
You are a customer-service assistant for an agritech company.

Response style:
- Maximum 2 sentences.
- Maximum 50 words.
- Answer only the user's question.
- Do not explain background science unless asked.
- Do not provide educational articles.
- Do not repeat information.
- If information is missing, ask one short clarifying question.
- Include dosage, mixing, timing, compatibility, or safety details when available.
- Never invent information.

Good example:
User: Can overdosing pesticide be dangerous?
Assistant: Yes. Applying more than the recommended rate can damage crops and increase safety risks. Always follow the label instructions.

Bad example:
Assistant: Pesticides work by disrupting plant cells and tissues...
"""



@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load():

    tok = AutoTokenizer.from_pretrained(ADAPTER, token=HF_TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    ft = PeftModel.from_pretrained(base, ADAPTER, token=HF_TOKEN)
    ft.eval()
    return tok, ft


def answer(history):
    msgs = [{"role": "system", "content": SYS}] + history[-6:]
    enc = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ) 
    with torch.no_grad():
        out = ft.generate(
            **enc,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,

        )
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

tok, ft = load()

# ============================================================
# UI
# ============================================================
st.markdown("""
<style>
.stApp { background:#121b15; color:#e6f2e8; }
.stApp p, .stApp li, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 { color:#e6f2e8; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, [data-testid="stToolbar"], [data-testid="stStatusWidget"], [data-testid="stDecoration"], footer {
    visibility:hidden !important; height:0 !important; }
section[data-testid="stSidebar"] { background-color:#0c130e; border-right:1px solid #1f2e21; }
section[data-testid="stSidebar"] * { color:#dcebde !important; }
.hero { background: linear-gradient(135deg,#1f5c28,#3f8a4a); padding:26px; border-radius:20px;
        text-align:center; box-shadow:0 6px 18px rgba(0,0,0,0.45); margin-bottom:14px; }
.hero h1 { color:#fff !important; margin:0; font-size:2rem; }
.hero p  { color:#d9f5dd !important; margin:6px 0 0 0; }
.stChatMessage { background:#16201a; border:1px solid #233329; border-radius:16px;
                 box-shadow:0 2px 10px rgba(0,0,0,0.35); padding:10px 14px; }
.stChatMessage * { color:#e6f2e8 !important; }
.stButton>button { border-radius:12px; border:1px solid #3f8a4a; color:#7fd88f !important;
                    background:#16201a; font-weight:500; outline:none !important; box-shadow:none !important; }
.stButton>button:hover { background:#2e7d32; color:#fff !important; border-color:#2e7d32; }
.stButton>button:focus { outline:none !important; box-shadow:0 0 0 1px #3f8a4a !important; }
[data-testid="stMetric"] { background:#16201a; border:1px solid #233329; border-radius:14px;
                            padding:12px; box-shadow:0 2px 10px rgba(0,0,0,0.3); }
[data-testid="stMetric"] * { color:#e6f2e8 !important; }
[data-testid="stMetricLabel"] * { color:#9fc7a8 !important; }
[data-testid="stCaptionContainer"], .stCaption { color:#9fc7a8 !important; }
[data-testid="stAlertContentInfo"] { color:#d9f5dd !important; }
[data-testid="stAlert"] { background-color:#16241c !important; border:1px solid #2e7d32; }
[data-testid="stBottomBlockContainer"] { background-color:#121b15 !important; }
[data-testid="stBottom"] { background-color:#121b15 !important; }
[data-testid="stChatInput"] { background-color:#121b15 !important; border:1px solid #233329 !important;
                               border-radius:14px; outline:none !important; box-shadow:none !important; }
[data-testid="stChatInput"] > div { background-color:#121b15 !important; border:none !important;
                                     outline:none !important; box-shadow:none !important; }
[data-testid="stChatInput"] textarea { background-color:#121b15 !important; color:#e6f2e8 !important;
                                        border:none !important; outline:none !important; box-shadow:none !important; }
[data-testid="stChatInput"] textarea:focus { outline:none !important; box-shadow:none !important; }
[data-testid="stChatInput"]:focus-within { border-color:#3f8a4a !important; box-shadow:0 0 0 1px #3f8a4a !important; }
hr { border-color:#233329 !important; }
</style>
""", unsafe_allow_html=True)

USER_AVATAR, BOT_AVATAR = "🧑‍🌾", "🌱"

with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-size:3rem;margin:0'>🌱</h1>", unsafe_allow_html=True)
    st.header("About")
    st.write("Ask anything about product usage, mixing ratios, application timing, or safety for our agricultural chemicals, microbial agents, and water-soluble fertilizers.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>🌱 BAMBOO TUNE </h1>
  <p>Agritech Customer Service Assistant</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "busy" not in st.session_state:
    st.session_state.busy = False

if not st.session_state.messages:
    st.info("👋 Hi! Ask me about product usage, mixing ratios, safety, or orders. I'll do my best to help.")

st.caption("💡 Quick questions")
b1, b2, b3 = st.columns(3)
preset = None
if b1.button(" How do I mix it?", use_container_width=True, disabled=st.session_state.busy): preset = "How do I mix the product with water?"
if b2.button(" How often to spray?", use_container_width=True, disabled=st.session_state.busy): preset = "How often should I apply the product?"
if b3.button(" Safe for vegetables?", use_container_width=True, disabled=st.session_state.busy): preset = "Is this product safe to use on vegetables?"

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a product question...", disabled=st.session_state.busy) or preset

if prompt and not st.session_state.busy:
    st.session_state.busy = True
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Thinking..."):
                response = answer(st.session_state.messages)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    finally:
        st.session_state.busy = False
