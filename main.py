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

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 2. Load Models ---
@st.cache_resource
def load_llm_model():
    st.info(f"Loading LLM model: {LLM_MODEL_PATH} on {DEVICE}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            device_map=DEVICE,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            use_auth_token=HF_TOKEN
        )
        st.success("LLM model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load LLM model: {e}")
        return None, None

@st.cache_resource
def load_embedding_model():
    st.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        st.success("Embedding model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

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

        # First try paragraph-based chunking
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
        
        # Add remaining content
        if current_chunk_content.strip():
            all_chunks.append({
                "content": current_chunk_content.strip(),
                "source_file": source_file,
                "page_number": page_number
            })
        
        # If paragraph chunking didn't work well, fall back to word-based chunking
        if len(all_chunks) == 0 or (len(text) > chunk_size * 2 and len([c for c in all_chunks if c["source_file"] == source_file and c["page_number"] == page_number]) <= 1):
            # Remove existing chunks for this page
            all_chunks = [c for c in all_chunks if not (c["source_file"] == source_file and c["page_number"] == page_number)]
            
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
    embedding_model = load_embedding_model()
    if embedding_model is None:
        st.error("Embedding model not loaded")
        return None, [], []
    
    if not chunks:
        st.warning("No text chunks to process")
        return None, [], []

    st.info("Generating embeddings and building FAISS index...")
    
    text_chunks = [chunk["content"] for chunk in chunks]
    chunk_metadata = [{"source_file": c["source_file"], "page_number": c["page_number"]} for c in chunks]

    try:
        embeddings = embedding_model.encode(
            text_chunks,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=DEVICE
        )
        embeddings = embeddings.cpu().numpy().astype('float32')

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)

        st.success(f"FAISS index created with {len(text_chunks)} chunks.")
        return faiss_index, text_chunks, chunk_metadata
    
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, [], []

def search_faiss_index(query, faiss_index, text_chunks, chunk_metadata, k=5):
    if faiss_index is None:
        st.error("FAISS index not initialized")
        return []
    
    embedding_model = load_embedding_model()
    if embedding_model is None:
        return []
    
    try:
        query_embedding = embedding_model.encode([query], convert_to_tensor=True, device=DEVICE)
        query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        distances, indices = faiss_index.search(query_embedding, k)
        
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if idx < len(text_chunks):
                relevant_chunks.append({
                    "content": text_chunks[idx],
                    "source_file": chunk_metadata[idx]["source_file"],
                    "page_number": chunk_metadata[idx]["page_number"],
                    "distance": distances[0][i]
                })
        return relevant_chunks
    
    except Exception as e:
        st.error(f"Error searching FAISS index: {e}")
        return []

# --- 5. LLM Answer Generation ---
def generate_llm_answer(question, relevant_chunks):
    tokenizer, model = load_llm_model()
    if model is None or tokenizer is None:
        return "Sorry, the AI model is not available.", "No sources."

    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    source_info = "\n".join([f"- {chunk['source_file']} (Page {chunk['page_number']})" for chunk in relevant_chunks])

    prompt = f"""You are an academic assistant. Answer the question truthfully based ONLY on the context.
Context:
{context}

Question: {question}

Answer:"""

    try:
        conv = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True).to(DEVICE)

        set_seed(42)
        with torch.no_grad():
            output = model.generate(
                **input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return prediction.strip(), source_info
    
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return f"Sorry, there was an error generating the answer: {e}", source_info

# --- 6. Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="StudyMate AI Assistant")
st.title("📚 StudyMate AI Academic Assistant")
st.markdown("Upload your academic PDFs and ask natural-language questions!")

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

uploaded_files = st.sidebar.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Check if files have changed
    current_files_hash = hash(tuple(f.name for f in uploaded_files))
    
    if "processed_files_hash" not in st.session_state or st.session_state.processed_files_hash != current_files_hash:
        st.session_state.processed_files_hash = current_files_hash
        st.session_state.all_chunks = []
        st.session_state.pdf_names = []
        
        # Create temporary directory
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
                    chunks = chunk_text(doc_texts)
                    st.session_state.all_chunks.extend(chunks)
                    st.info(f"Processed {uploaded_file.name}: {len(chunks)} chunks extracted")
                else:
                    st.warning(f"No text extracted from {uploaded_file.name}")
                
                os.remove(file_path)
            
            # Create FAISS index
            if st.session_state.all_chunks:
                faiss_index, text_chunks, chunk_metadata = create_faiss_index(st.session_state.all_chunks)
                st.session_state.faiss_index = faiss_index
                st.session_state.text_chunks = text_chunks
                st.session_state.chunk_metadata = chunk_metadata
            else:
                st.warning("No text extracted from uploaded PDFs.")
                st.session_state.faiss_index = None
                st.session_state.text_chunks = []
                st.session_state.chunk_metadata = []
        
        # Clean up temp directory
        if os.path.exists(temp_pdf_dir):
            shutil.rmtree(temp_pdf_dir, ignore_errors=True)

    # Display processing status
    if st.session_state.pdf_names:
        st.sidebar.success(f"✅ Processed {len(st.session_state.pdf_names)} PDFs with {len(st.session_state.all_chunks)} chunks.")
        
        # Display file list
        st.sidebar.write("**Processed files:**")
        for pdf_name in st.session_state.pdf_names:
            st.sidebar.write(f"• {pdf_name}")
    else:
        st.sidebar.info("No PDFs processed yet.")

else:
    # Clear session state when no files are uploaded
    if "processed_files_hash" in st.session_state:
        for key in ["processed_files_hash", "all_chunks", "pdf_names", "faiss_index", "text_chunks", "chunk_metadata"]:
            if key in st.session_state:
                del st.session_state[key]
    st.info("📁 Upload PDFs in the sidebar to start!")

# Question answering interface
st.subheader("Ask a Question:")
question = st.text_area("Type your question here:", height=100, key="user_question")

if st.button("Get Answer", type="primary") and question:
    if st.session_state.faiss_index is not None and st.session_state.text_chunks:
        with st.spinner("Searching for relevant information..."):
            relevant_chunks = search_faiss_index(
                question, 
                st.session_state.faiss_index, 
                st.session_state.text_chunks, 
                st.session_state.chunk_metadata, 
                k=5
            )
            
            if relevant_chunks:
                st.success("📄 Found relevant information!")
                
                # Show relevant chunks (optional - for debugging)
                with st.expander("View relevant text chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}** (Distance: {chunk['distance']:.4f})")
                        st.write(f"Source: {chunk['source_file']} - Page {chunk['page_number']}")
                        st.write(chunk['content'])
                        st.write("---")
                
                with st.spinner("Generating answer..."):
                    answer, source_info = generate_llm_answer(question, relevant_chunks)
                    
                    st.success("🤖 Answer:")
                    st.write(answer)
                    
                    if source_info:
                        st.info("📚 Sources:")
                        st.markdown(source_info)
            else:
                st.warning("❌ No relevant information found in documents.")
                st.info("Try rephrasing your question or check if the information exists in your uploaded documents.")
    else:
        if not uploaded_files:
            st.warning("📁 Please upload PDF documents first.")
        else:
            st.error("❌ FAISS index not initialized. Please try re-uploading your documents.")

# Display current status
if uploaded_files and st.session_state.faiss_index is not None:
    st.sidebar.info(f"🔍 Ready to answer questions!\nIndex contains {len(st.session_state.text_chunks)} text chunks.")
elif uploaded_files:
    st.sidebar.warning("⚠️ Processing documents...")
else:
    st.sidebar.info("📤 Upload documents to get started.")