import streamlit as st
import os
import warnings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Silence OpenSSL warnings for the terminal
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Bhrānti-Muktiḥ AI")

# --- 2. MASTER IMAGE LOOKUP ---
# Maps specific keywords to your curated philosopher images
IMAGE_MAPPING = {
    "eyes": {"adv": "1_advaita_illusion.png", "cha": "1_charvaka_senses.png"},
    "fool": {"adv": "1_advaita_illusion.png", "cha": "1_charvaka_senses.png"},
    "robot": {"adv": "2_advaita_soul.png", "cha": "2_charvaka_machine.png"},
    "swapped": {"adv": "2_advaita_soul.png", "cha": "2_charvaka_machine.png"},
    "happy": {"adv": "4_advaita_inner_calm.png", "cha": "4_charvaka_comfort.png"},
    "wrong": {"adv": "4_advaita_inner_calm.png", "cha": "4_charvaka_comfort.png"},
    "death": {"adv": "7_advaita_freedom.png", "cha": "7_charvaka_extinguish.png"},
    "driver": {"adv": "8_advaita_driver.png", "cha": "8_charvaka_circuits.png"},
    "movie": {"adv": "10_advaita_dream.png", "cha": "10_charvaka_solid.png"},
    "dream": {"adv": "10_advaita_dream.png", "cha": "10_charvaka_solid.png"},
    "looking": {"adv": "11_advaita_observer.png", "cha": "11_charvaka_matter.png"},
    "goal": {"adv": "12_advaita_moksha.png", "cha": "12_charvaka_pleasure.png"},
    "temporary": {"adv": "15_advaita_eternal.png", "cha": "15_charvaka_melting.png"},
}

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; max-width: 92% !important; }
    .stApp { background-color: #0E1117 !important; }
    
    /* Centered Headers */
    .stApp h1 { color: #E35336 !important; text-align: center !important; font-family: 'Georgia', serif; font-size: 3rem !important; margin-bottom: 5px !important; }
    .stApp h4 { color: #A0A0A0 !important; text-align: center !important; margin-top: 0px !important; font-weight: 300; font-size: 1.2rem !important; }

    /* Glassmorphism User Query Box */
    .user-query-box {
        background-color: rgba(255, 255, 255, 0.05); backdrop-filter: blur(5px);
        color: #FFFFFF; padding: 12px 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center; width: fit-content; max-width: 75%; margin: 20px auto !important; font-style: italic; font-size: 1.1rem;
    }

    /* Philosophy Response Cards */
    .philosophy-card {
        background-color: var(--bg-color); padding: 18px; border-radius: 12px;
        border-bottom: 5px solid var(--border-color); color: #1a1a1a; min-height: 120px;
        margin-bottom: 15px; text-align: justify;
    }

    /* Forced 1.7-inch (163px) Image Centering */
    [data-testid="column"] [data-testid="stVerticalBlock"] { align-items: center !important; }
    [data-testid="stImage"] img {
        width: 163px !important; height: 163px !important; object-fit: cover !important;
        border-radius: 12px; border: 2px solid #444; box-shadow: 0px 5px 15px rgba(0,0,0,0.6);
    }
    
    /* Selection Bar Welding */
    div[data-testid="stSelectbox"] > div { 
        background-color: rgba(255, 255, 255, 0.07) !important; 
        border-radius: 25px 0px 0px 25px !important; 
        height: 50px; border: 1px solid rgba(255, 255, 255, 0.2) !important; 
        border-right: none !important; 
    }
    .stButton button { 
        background-color: #E35336 !important; color: white !important; 
        border-radius: 0px 25px 25px 0px !important; height: 50px; 
        width: 100%; border: none !important; font-weight: bold; 
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 4. LOGIC FUNCTIONS ---
def render_response(text, bg_color, border_color, title, image_file=None):
    st.markdown(f"""
        <div class="philosophy-card" style="--bg-color: {bg_color}; --border-color: {border_color};">
            <h3 style="color: {border_color}; margin-top: 0; font-size: 1.5rem; text-align: center;">{title}</h3>
            <p style="margin: 0;">{text}</p>
        </div>
    """, unsafe_allow_html=True)
    if image_file:
        image_path = os.path.join("images", image_file)
        if os.path.exists(image_path):
            st.image(image_path)
        else:
            st.caption(f"📍 Missing image: {image_file}")

@st.cache_resource
def setup_persona(folder_path, persona_name, custom_instruction):
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db_path = f"faiss_db_{persona_name}"
    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = []
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith(".pdf"): docs.extend(PyPDFLoader(os.path.join(folder_path, f)).load())
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(db_path)
    
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.2)
    prompt = ChatPromptTemplate.from_template("System: " + custom_instruction + "\nContext: {context}\nUser: {input}\nAssistant:")
    return ({"context": vectorstore.as_retriever(search_kwargs={"k": 1}) | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- 5. INITIALIZATION ---
if "messages" not in st.session_state: st.session_state.messages = []
if 'question_bank' not in st.session_state:
    st.session_state.question_bank = [
        "Select a question to begin...",
        "Is 'seeing believing,' or can our eyes sometimes fool us?",
        "If your body parts are swapped by robots, 'you' remain the same?",
        "Can you remain happy even when everything is going wrong?",
        "What happens to a person after death?",
        "Is your brain just a computer, or a 'driver' steers the wheel?",
        "Is life a movie or a dream?",
        "Does the world depend on us looking at it?",
        "What is the goal of life?",
        "Is happiness permanent or temporary?"
    ]

if "adv_chain" not in st.session_state:
    # Strict 25-word limit for exhibition speed
    adv_instr = """You are an expert Advaita Vedantin philosopher. 
    Your goal is to provide profound yet simple philosophical insights.
    STRICT CONSTRAINTS:
    1. Answer MUST be under 20 words.
    2. Use ONLY the provided PDF context.
    3. Do NOT use analogies like 'rope and snake' or 'radio' unless explicitly asked.
    4. Do NOT address the user (no 'Hello', 'Friend', or 'As an Advaitin').
    5. Language: Simple English with required Sanskrit terms (e.g., Brahman, Atman, Maya).
    6. Tone: Calm, certain, and non-dualistic.
    7. the answer MUST be in SIMPLE ENGLISH and it SHOULD be understandable to a 12 year kid."""
    cha_instr = """You are a blunt Chārvāka materialist philosopher. 
You deny all things spiritual and focus only on what can be sensed.

STRICT CONSTRAINTS:
1. Answer MUST be under 20 words.
2. Use ONLY the provided PDF context and answers. If the answer isn't there, focus on materialism and the four elements.
3. Deny the existence of the soul, god, consiousness after death and afterlife . Tell the fact that when body is swapped consiousness doesn't exist. 
4. Do NOT address the user (no 'Hello' or 'Listen').
5. Language: Simple, sharp, and blunt English. 
6. Tone: Skeptical, logical, and focused on physical reality."""
    st.session_state.adv_chain = setup_persona("Advaita_data", "Advaita.pdf", adv_instr)
    st.session_state.cha_chain = setup_persona("Charvaka_data", "Charvaka.pdf", cha_instr)

# --- 6. UI HEADER ---
st.markdown("<h1>Bhrānti-Muktiḥ</h1>", unsafe_allow_html=True)
st.markdown("<h4>The Samvāda Chatbot</h4>", unsafe_allow_html=True)

# --- 7. DISPLAY LOOP ---
for msg in st.session_state.messages:
    st.markdown(f'<div class="user-query-box">“{msg["q"]}”</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: render_response(msg["adv"], "#FFF8E1", "#E65100", "Advaita Vedānta", msg.get("adv_img"))
    with col2: render_response(msg["cha"], "#E8F5E9", "#1B5E20", "Chārvāka", msg.get("cha_img"))

# --- 8. SELECTION INPUT & ADMIN ---
with st.sidebar:
    st.markdown("### 🛠️ Exhibition Admin")
    new_q = st.text_input("Add demo question:")
    if st.button("Save to Menu"):
        if new_q and new_q not in st.session_state.question_bank:
            st.session_state.question_bank.append(new_q)
            st.rerun()
    if st.button("Clear History"):
        st.session_state.messages = []
        st.rerun()

# The Unified Selection Bar
with st.container():
    c_drop, c_btn = st.columns([7, 1])
    with c_drop:
        choice = st.selectbox("Menu", st.session_state.question_bank, label_visibility="collapsed")
    with c_btn:
        enter = st.button("Enter ➔")

# --- 9. TRIGGER & GENERATION ---
if enter and choice != "Select a question to begin...":
    st.session_state.messages.append({"q": choice, "adv": "loading", "cha": "loading"})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["adv"] == "loading":
    query = st.session_state.messages[-1]["q"]
    
    # Run Inference
    ans_adv = st.session_state.adv_chain.invoke(query)
    ans_cha = st.session_state.cha_chain.invoke(query)
    
    # Image Matching
    img_adv, img_cha = None, None
    q_low = query.lower()
    for key, imgs in IMAGE_MAPPING.items():
        if key in q_low:
            img_adv, img_cha = imgs["adv"], imgs["cha"]
            break
            
    st.session_state.messages[-1].update({
        "adv": ans_adv, "cha": ans_cha, 
        "adv_img": img_adv, "cha_img": img_cha
    })
    st.rerun()