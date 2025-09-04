import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import shutil
from huggingface_hub import HfFolder

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face API Token (⚠️ replace with your actual token)
HF_TOKEN = "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"
HfFolder.save_token(HF_TOKEN)

# LLM Configuration
LLM_MODEL_PATH = "ibm-granite/granite-3.3-8b-instruct"
LLM_TOKENIZER = None
LLM_MODEL = None

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = None

# Data Storage
FAISS_INDEX = None
TEXT_CHUNKS = []
CHUNK_METADATA = []

# --- 2. Load Models ---
@st.cache_resource
def load_llm_model():
    global LLM_MODEL, LLM_TOKENIZER
    if LLM_MODEL is None or LLM_TOKENIZER is None:
        st.info(f"Loading LLM model: {LLM_MODEL_PATH} on {DEVICE}...")
        try:
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, use_auth_token=HF_TOKEN)
            LLM_MODEL = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                device_map=DEVICE,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                use_auth_token=HF_TOKEN
            )
            st.success("LLM model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load LLM model: {e}")
            LLM_TOKENIZER = None
            LLM_MODEL = None
    return LLM_TOKENIZER, LLM_MODEL

@st.cache_resource
def load_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        st.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            st.success("Embedding model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            EMBEDDING_MODEL = None
    return EMBEDDING_MODEL

# --- 3. PDF Processing ---
def extract_text_from_pdf(pdf_path):
    doc_texts = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    doc_texts.append({
                        "text": text,
                        "source_file": os.path.basename(pdf_path),
                        "page_number": page_num + 1
                    })
        return doc_texts
    except Exception as e:
        st.error(f"Error processing PDF {os.path.basename(pdf_path)}: {e}")
        return []

def chunk_text(doc_texts, chunk_size=512, overlap_size=50):
    all_chunks = []
    for doc_info in doc_texts:
        text = doc_info["text"]
        source_file = doc_info["source_file"]
        page_number = doc_info["page_number"]

        paragraphs = text.split('\n\n')
        current_chunk_content = ""
        for para in paragraphs:
            if len(current_chunk_content) + len(para) + 2 < chunk_size:
                current_chunk_content += (para + "\n\n")
            else:
                if current_chunk_content.strip():
                    all_chunks.append({
                        "content": current_chunk_content.strip(),
                        "source_file": source_file,
                        "page_number": page_number
                    })
                current_chunk_content = para + "\n\n"
        if current_chunk_content.strip():
            all_chunks.append({
                "content": current_chunk_content.strip(),
                "source_file": source_file,
                "page_number": page_number
            })

        if not all_chunks or (len(text) > chunk_size * 2 and len(all_chunks) == len(doc_texts)):
            all_chunks = [c for c in all_chunks if c["source_file"] != source_file or c["page_number"] != page_number]
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    all_chunks.append({
                        "content": chunk.strip(),
                        "source_file": source_file,
                        "page_number": page_number
                    })
    return all_chunks

# --- 4. Embedding and FAISS ---
def create_faiss_index(chunks):
    global FAISS_INDEX, TEXT_CHUNKS, CHUNK_METADATA, EMBEDDING_MODEL
    EMBEDDING_MODEL = load_embedding_model()
    if EMBEDDING_MODEL is None:
        st.error("Embedding model not loaded")
        return
    if not chunks:
        st.warning("No text chunks to process")
        return

    st.info("Generating embeddings and building FAISS index...")
    TEXT_CHUNKS = [chunk["content"] for chunk in chunks]
    CHUNK_METADATA = [{"source_file": c["source_file"], "page_number": c["page_number"]} for c in chunks]

    embeddings = EMBEDDING_MODEL.encode(
        TEXT_CHUNKS,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=DEVICE
    )
    embeddings = embeddings.cpu().numpy().astype('float32')

    dimension = embeddings.shape[1]
    FAISS_INDEX = faiss.IndexFlatL2(dimension)
    FAISS_INDEX.add(embeddings)

    st.success(f"FAISS index created with {len(TEXT_CHUNKS)} chunks.")

def search_faiss_index(query, k=5):
    global FAISS_INDEX, TEXT_CHUNKS, CHUNK_METADATA, EMBEDDING_MODEL
    if FAISS_INDEX is None:
        st.error("FAISS index not initialized")
        return []
    EMBEDDING_MODEL = load_embedding_model()
    if EMBEDDING_MODEL is None:
        return []
    query_embedding = EMBEDDING_MODEL.encode([query], convert_to_tensor=True, device=DEVICE)
    query_embedding = query_embedding.cpu().numpy().astype('float32')
    distances, indices = FAISS_INDEX.search(query_embedding, k)
    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(TEXT_CHUNKS):
            relevant_chunks.append({
                "content": TEXT_CHUNKS[idx],
                "source_file": CHUNK_METADATA[idx]["source_file"],
                "page_number": CHUNK_METADATA[idx]["page_number"],
                "distance": distances[0][i]
            })
    return relevant_chunks

# --- 5. LLM Answer Generation ---
def generate_llm_answer(question, relevant_chunks):
    global LLM_TOKENIZER, LLM_MODEL
    LLM_TOKENIZER, LLM_MODEL = load_llm_model()
    if LLM_MODEL is None or LLM_TOKENIZER is None:
        return "Sorry, the AI model is not available.", "No sources."

    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    source_info = "\n".join([f"- {chunk['source_file']} (Page {chunk['page_number']})" for chunk in relevant_chunks])

    prompt = f"""You are an academic assistant. Answer the question truthfully based ONLY on the context.
Context:
{context}

Question: {question}

Answer:"""

    conv = [{"role": "user", "content": prompt}]
    input_ids = LLM_TOKENIZER.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True).to(DEVICE)

    set_seed(42)
    output = LLM_MODEL.generate(
        **input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    prediction = LLM_TOKENIZER.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction, source_info

# --- 6. Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="StudyMate AI Assistant")
st.title("📚 StudyMate AI Academic Assistant")
st.markdown("Upload your academic PDFs and ask natural-language questions!")

uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if "processed_files_hash" not in st.session_state:
        st.session_state.processed_files_hash = None
    current_files_hash = hash(tuple(f.name for f in uploaded_files))
    if st.session_state.processed_files_hash != current_files_hash:
        st.session_state.processed_files_hash = current_files_hash
        st.session_state.all_chunks = []
        st.session_state.pdf_names = []
        temp_pdf_dir = "./temp_pdfs"
        os.makedirs(temp_pdf_dir, exist_ok=True)

        with st.spinner("Processing PDFs..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_pdf_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.pdf_names.append(uploaded_file.name)
                doc_texts = extract_text_from_pdf(file_path)
                if doc_texts:
                    st.session_state.all_chunks.extend(chunk_text(doc_texts))
                os.remove(file_path)
            if st.session_state.all_chunks:
                create_faiss_index(st.session_state.all_chunks)
                st.session_state.FAISS_INDEX = FAISS_INDEX
                st.session_state.TEXT_CHUNKS = TEXT_CHUNKS
                st.session_state.CHUNK_METADATA = CHUNK_METADATA
                
            else:
                st.warning("No text extracted from uploaded PDFs.")
                FAISS_INDEX = None
        if not os.listdir(temp_pdf_dir):
            shutil.rmtree(temp_pdf_dir, ignore_errors=True)

    if "pdf_names" in st.session_state and st.session_state.pdf_names:
        st.sidebar.success(f"Processed {len(st.session_state.pdf_names)} PDFs with {len(st.session_state.all_chunks)} chunks.")
    else:
        st.sidebar.info("No PDFs processed yet.")

st.subheader("Ask a Question:")
question = st.text_area("Type your question here:", height=100, key="user_question")

if st.button("Get Answer", type="primary") and question:
    if "FAISS_INDEX" in st.session_state and st.session_state.FAISS_INDEX is not None:

        with st.spinner("Searching and generating answer..."):
            relevant_chunks = search_faiss_index(question, k=5)
            if relevant_chunks:
                answer, source_info = generate_llm_answer(question, relevant_chunks)
                st.success("Answer:")
                st.write(answer)
                st.info("Sources:")
                st.markdown(source_info)
            else:
                st.warning("No relevant info found in documents.")
    else:
        st.warning("Please upload and process PDFs first.")

elif "processed_files_hash" in st.session_state and not uploaded_files:
    del st.session_state.processed_files_hash
    if "all_chunks" in st.session_state: del st.session_state.all_chunks
    if "pdf_names" in st.session_state: del st.session_state.pdf_names
    FAISS_INDEX, TEXT_CHUNKS, CHUNK_METADATA = None, [], []
    st.info("No PDFs currently uploaded.")

if not uploaded_files and "processed_files_hash" not in st.session_state:
    st.info("Upload PDFs in the sidebar to start!")
