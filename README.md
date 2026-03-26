# Bhrānti-Muktiḥ (The Vāsanā Project) 🕉️🤖
### An Interactive AI Samvāda between Advaita Vedānta and Chārvāka (Materialism)
## 📖 Overview
Developed for a 3-day Vēdānta Makathon organised by Vēdānta Bhārati, **Bhrānti-Muktiḥ** is a Retrieval-Augmented Generation (RAG) system that facilitates a digital dialogue between two contrasting schools of Indian Philosophy. 

The project demonstrates how local LLMs can be used to navigate complex metaphysical queries—such as the nature of the self, consciousness, and the material world—without relying on cloud-based APIs.

## 🛠️ Tech Stack
* **LLM Engine:** Ollama (Llama 3.2 1B)
* **Orchestration:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** FastEmbed (`BAAI/bge-small-en-v1.5`)
* **Frontend:** Streamlit with custom Glassmorphism CSS
* **Language:** Python 3.10+

## 🚀 Features
* **100% Local & Offline:** Optimized to run entirely without internet during live exhibitions.
* **Persona-Driven RAG:** Specialized system prompts ensure the AI maintains a "calm" Advaitin persona and a "blunt" Chārvāka persona.
* **Hybrid Sanskrit Support:** Uses Gemma 2 9B to provide accurate philosophical terminology in Devanagari script.
* **Selection-Only Input:** A custom UI designed to prevent out-of-scope user queries during high-traffic events.

## ⚡ Setup Instructions
1. **Install Ollama** and pull the models:
   ```bash
   ollama pull llama3.2:1b
   pip install streamlit langchain-ollama langchain-community faiss-cpu fastembed pypdf
   streamlit run app.py

   ## 6. Directory Structure
Show that your project is organized.
```markdown
## 📂 Folder Structure
* `Advaita_data/`: Source PDFs for Advaita Vedānta philosophy.
* `Charvaka_data/`: Source PDFs for Chārvāka Materialism.
* `images/`: 163px curated philosopher icons used in the UI.
* `app.py`: The core application logic and UI code.
