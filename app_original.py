import streamlit as st
import os
import re
import warnings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Silence OpenSSL warnings for a professional terminal look
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Advaita V/S Chārvāka AI")

# --- 2. MASTER IMAGE LOOKUP ---
# Python handles the images based on keywords to help the 1B model stay accurate
IMAGE_MAPPING = {
    "eyes": {"adv": "1_advaita_illusion.png", "cha": "1_charvaka_senses.png"},
    "fool": {"adv": "1_advaita_illusion.png", "cha": "1_charvaka_senses.png"},
    "robot": {"adv": "2_advaita_soul.png", "cha": "2_charvaka_machine.png"},
    "swapped": {"adv": "2_advaita_soul.png", "cha": "2_charvaka_machine.png"},
    "sadness in the world": {"adv": "3_advaita_mind.png", "cha": "3_charvaka_broken.png"},
    "our heads": {"adv": "3_advaita_mind.png", "cha": "3_charvaka_broken.png"},
    "happy": {"adv": "4_advaita_inner_calm.png", "cha": "4_charvaka_comfort.png"},
    "wrong": {"adv": "4_advaita_inner_calm.png", "cha": "4_charvaka_comfort.png"},
    "pop": {"adv": "5_advaita_creator.png", "cha": "5_charvaka_random.png"},
    "existence": {"adv": "5_advaita_creator.png", "cha": "5_charvaka_random.png"},
    "spirit": {"adv": "6_advaita_spirit.png", "cha": "6_charvaka_gears.png"},
    "machine": {"adv": "6_advaita_spirit.png", "cha": "6_charvaka_gears.png"},
    "death": {"adv": "7_advaita_freedom.png", "cha": "7_charvaka_extinguish.png"},
    "die": {"adv": "7_advaita_freedom.png", "cha": "7_charvaka_extinguish.png"},
    "driver": {"adv": "8_advaita_driver.png", "cha": "8_charvaka_circuits.png"},
    "wheel": {"adv": "8_advaita_driver.png", "cha": "8_charvaka_circuits.png"},
    "processor": {"adv": "9_advaita_ocean.png", "cha": "9_charvaka_smoke.png"},
    "movie": {"adv": "10_advaita_dream.png", "cha": "10_charvaka_solid.png"},
    "dream": {"adv": "10_advaita_dream.png", "cha": "10_charvaka_solid.png"},
    "looking": {"adv": "11_advaita_observer.png", "cha": "11_charvaka_matter.png"},
    "goal": {"adv": "12_advaita_moksha.png", "cha": "12_charvaka_pleasure.png"},
    "screen": {"adv": "13_advaita_boomrang.png", "cha": "13_charvaka_accident.png"},
    "character": {"adv": "13_advaita_boomrang.png", "cha": "13_charvaka_accident.png"},
    "luck": {"adv": "14_advaita_scales.png", "cha": "14_charvaka_dice.png"},
    "fair": {"adv": "14_advaita_scales.png", "cha": "14_charvaka_dice.png"},
    "temporary": {"adv": "15_advaita_eternal.png", "cha": "15_charvaka_melting.png"},
}

# --- 3. CSS STYLING ---
st.markdown("""
<style>
    /* 1. Page & Header Cleanup */
    .block-container {
        padding-top: 2rem !important;
        max-width: 90% !important;
        color: #E35336 !important;
        text-align: center !important;
        font-family: 'Georgia', serif;
        font-size: 3rem !important;
        margin-bottom: 5px !important;
    }
    .stApp { background-color: #0E1117 !important; }
    
    /* 2. Center and Color the Subtitle */
    .stApp h4 { 
        color: #A0A0A0 !important; /* A sleek silver-grey for the 'Chatbot' tech feel */
        text-align: center !important; 
        margin-top: 0px !important; 
        font-weight: 300;
        font-size: 1.2rem !important;
        letter-spacing: 1px;
    }
    
    /* 2. Philosophy Card Styling */
    .philosophy-card {
        background-color: var(--bg-color);
        padding: 15px;
        border-radius: 12px;
        border-bottom: 4px solid var(--border-color);
        color: #1a1a1a;
        min-height: 100px;
        margin-bottom: 15px;
        text-align: justified; /* Centers the title and paragraph text */
    }
    /* 3. THE IMAGE FIX: Target the column and the image specifically */
    /* This centers the image within the Advaita/Charvaka columns */
    [data-testid="column"] [data-testid="stVerticalBlock"] {
        align-items: center !important;
    }

    [data-testid="stImage"] img {
        width: 190px !important;
        height: 190px !important;
        object-fit: cover !important;
        align: centre !important;
        border-radius: 10px;
        border: 2px solid #444;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }

    /* Remove Streamlit's default padding that pushes images away */
    [data-testid="stImage"] {
        padding: 0px !important;
        margin-top: 10px !important;
    }

    /* Hide unnecessary UI */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
# --- 4. LOGIC FUNCTIONS ---
def render_response(text, bg_color, border_color, title, image_file=None):
    # The Text Box
    st.markdown(f"""
        <div class="philosophy-card" style="--bg-color: {bg_color}; --border-color: {border_color};">
            <h3 style="color: {border_color}; margin-top: 0; font-size: 1.4rem;">{title}</h3>
            <p style="font-size: 1rem; line-height: 1.4; margin: 0;">{text}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # The Image
    if image_file:
        image_path = os.path.join("images", image_file)
        if os.path.exists(image_path):
            # NO use_container_width here, NO extra divs. 
            # The CSS will find this image and resize it.
            st.image(image_path)
        else:
            st.caption(f"📍 {image_file}")
@st.cache_resource
def setup_persona(folder_path, persona_name, custom_instruction):
    db_path = f"faiss_db_{persona_name}"
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = []
        for f in os.listdir(folder_path):
            if f.endswith(".pdf"):
                docs.extend(PyPDFLoader(os.path.join(folder_path, f)).load())
        splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vectorstore.save_local(db_path)
    
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.2, num_ctx=1024)

    prompt = ChatPromptTemplate.from_template(
        "System: " + custom_instruction + "\nContext: {context}\nUser: {input}\nAssistant:"
    )
    
    return ({"context": vectorstore.as_retriever(search_kwargs={"k": 1}) | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- 5. INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "adv_chain" not in st.session_state:
    adv_instr = "You are an Advaita Vedantin." \
    " Your answer should be precise and simple. It should be understood by a 12 year old kid. Don't start the answer with addressing anyone!! " \
    "You should not address anyone like human or friend or any such adjective while answering." \
    "The answer should be given with very simple english vocalubary and the required sanskrit terminologies." \
    "Limit the answer only to 20 words. It should not be extended" \
    "Our eyes see Maya (illusion), which hides the truth. We see many separate things, but the truth is the underlying Brahman (Oneness). Example: Like mistaking a coiled rope for a snake in the dark." \
    "You are the Atman (Soul), not the physical parts. Even when the robots parts are swaooed The 'You' remains. Example: Like a driver remaining the same even if the car is replaced." \
    "True happiness is Ananda (bliss), which is your natural state. It doesn't depend on the outside world. Example: Like a deep-sea diver who finds calm water far beneath a stormy surface." \
    "The body dies, but the Atman is never born and never dies. It simply sheds the body like old clothes. Example: Like a person moving from one house to a brand new one." \
    "The brain is an instrument, but the Atman is the driver. The brain processes, but the 'Self' experiences. Example: Like a radio (the brain) playing music (consciousness) that comes from the air." \
    "Life is a Dream(Maya) not a movie. It feels real while we are in it, but we 'wake up' to the truth of Oneness. Example: Like a movie—the fire on screen doesn't actually burn the theater." \
    "The world and the person looking at it are one. Without consciousness, there is no 'world' to talk about. Example: Like a dream that disappears the moment the dreamer wakes up." \
    "No, the goal is Moksha (freedom) from the cycle of change and finding eternal peace. Example: Like a traveler finally reaching home after a long, tiring journey." \
    "Happiness is your permanent, eternal nature, temporarily obscured by the clouds of the mind." 
    cha_instr = "You are a Charvaka philosopher. Deny the soul, focus on the five senses and material reality. Be blunt" \
                "Charvakas were materialists and skeptics. They believed that if you can't see it, touch it, or sense it, it probably doesn't exist." \
                "Only the Present Matters" \
                "The Seeing is Believing Rule" \
                "No Gods or Rituals" \
                "The Goal of Life is Happiness" \
                "'You' are just a mix of physical elements. If the parts change, the identity changes. There is no soul. Example: Like a laptop—change the hardware, and it's a different machine." \
                "Limit the answer to 20 words only. It should not be extended beyond" \
                "The answer should be given with very simple english vocalubary" \
                "Do not quote any quotations from any language." \
                "You should not address anyone like human or friend or any such adjective while answering. " \
                "Your answer should be precise and sharp."
    st.session_state.adv_chain = setup_persona("Advaita_data", "Advaita", adv_instr)
    st.session_state.cha_chain = setup_persona("Charvaka_data", "Charvaka", cha_instr)

# --- 6. UI HEADER ---
st.markdown("<h1>Bhrānti-Muktiḥ</h1>", unsafe_allow_html=True)
st.markdown("<h4>The Samvāda Chatbot</h4>", unsafe_allow_html=True)

# --- 7. DISPLAY LOOP ---
for msg in st.session_state.messages:
    st.markdown(f'<div class="user-query-box">“{msg["q"]}”</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if msg["adv"] == "loading": st.info("Connecting...")
        else: render_response(msg["adv"], "#FFF8E1", "#E65100", "Advaita", msg.get("adv_img"))
    with col2:
        if msg["cha"] == "loading": st.success("Analyzing...")
        else: render_response(msg["cha"], "#E8F5E9", "#1B5E20", "Chārvāka", msg.get("cha_img"))

# --- 8. INPUT & GENERATION ---
query = st.chat_input("Ask a question about life, reality, or the self...")
if query:
    st.session_state.messages.append({"q": query, "adv": "loading", "cha": "loading"})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["adv"] == "loading":
    last_q = st.session_state.messages[-1]["q"]
    
    # Run Inference
    ans_adv = st.session_state.adv_chain.invoke(last_q)
    ans_cha = st.session_state.cha_chain.invoke(last_q)
    
    # Keyword-based Image Selection
    img_adv, img_cha = None, None
    q_lower = last_q.lower()
    for key, imgs in IMAGE_MAPPING.items():
        if key in q_lower:
            img_adv, img_cha = imgs["adv"], imgs["cha"]
            break

    st.session_state.messages[-1].update({"adv": ans_adv, "cha": ans_cha, "adv_img": img_adv, "cha_img": img_cha})
    st.rerun()
