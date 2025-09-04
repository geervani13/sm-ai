import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import os
import shutil

# --- 1. Configuration ---
# Set the device for PyTorch operations
# Automatically uses GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LLM Configuration
LLM_MODEL_PATH = "ibm-granite/granite-3.3-8b-instruct" # Make sure this model is accessible
LLM_TOKENIZER = None # Will be loaded once
LLM_MODEL = None     # Will be loaded once

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A good general-purpose embedding model
EMBEDDING_MODEL = None # Will be loaded once

# Data Storage for processed chunks and FAISS index
FAISS_INDEX = None
TEXT_CHUNKS = [] # Stores the actual text content of the chunks
CHUNK_METADATA = [] # Stores metadata like source file, page number for each chunk

# --- 2. Load Models (Cached for Streamlit efficiency) ---
@st.cache_resource
def load_llm_model():
    """
    Loads the LLM model and tokenizer.
    Uses Streamlit's cache_resource to load these heavy objects only once.
    """
    global LLM_MODEL, LLM_TOKENIZER
    if LLM_MODEL is None or LLM_TOKENIZER is None:
        st.info(f"Loading LLM model: {LLM_MODEL_PATH} on {DEVICE}...")
        try:
            LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
            LLM_MODEL = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PATH,
                device_map=DEVICE,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            )
            st.success("LLM model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load LLM model: {e}. Please check model path and dependencies.")
            LLM_TOKENIZER = None
            LLM_MODEL = None
    return LLM_TOKENIZER, LLM_MODEL

@st.cache_resource
def load_embedding_model():
    """
    Loads the SentenceTransformer embedding model.
    Uses Streamlit's cache_resource to load this object only once.
    """
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        st.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        try:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            st.success("Embedding model loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}. Please check model name and internet connection.")
            EMBEDDING_MODEL = None
    return EMBEDDING_MODEL

# --- 3. PDF Processing ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF document, page by page.
    Returns a list of dictionaries, each containing text, source file, and page number.
    """
    doc_texts = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip(): # Only add if there's actual text
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
    """
    Splits document texts (from extract_text_from_pdf output) into smaller chunks with overlap.
    Prioritizes splitting by paragraphs, then falls back to character-level splitting for very long texts.
    Each chunk retains its source_file and an approximate page_number.
    """
    all_chunks = []
    # Process each page's text
    for doc_info in doc_texts:
        text = doc_info["text"]
        source_file = doc_info["source_file"]
        page_number = doc_info["page_number"] # Start page of this text block

        # Attempt to split by paragraphs first for better semantic coherence
        paragraphs = text.split('\n\n')
        current_chunk_content = ""
        for para in paragraphs:
            # If adding the next paragraph keeps the chunk within size, add it
            if len(current_chunk_content) + len(para) + 2 < chunk_size: # +2 for potential newlines
                current_chunk_content += (para + "\n\n")
            else:
                # If current chunk has content, save it
                if current_chunk_content.strip():
                    all_chunks.append({
                        "content": current_chunk_content.strip(),
                        "source_file": source_file,
                        "page_number": page_number
                    })
                # Start a new chunk with the current paragraph
                current_chunk_content = para + "\n\n"
        
        # Add any remaining content in current_chunk_content
        if current_chunk_content.strip():
            all_chunks.append({
                "content": current_chunk_content.strip(),
                "source_file": source_file,
                "page_number": page_number
            })

        # Fallback for cases where paragraph splitting isn't enough (e.g., very long paragraphs)
        # or if the document was just one giant block of text and didn't make enough chunks
        if not all_chunks or (len(text) > chunk_size * 2 and len(all_chunks) == len(doc_texts)): # Heuristic
            # Clear chunks from previous attempt for this doc_info if fallback is needed
            all_chunks = [c for c in all_chunks if c["source_file"] != source_file or c["page_number"] != page_number]

            words = text.split()
            for i in range(0, len(words), chunk_size - overlap_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    all_chunks.append({
                        "content": chunk.strip(),
                        "source_file": source_file,
                        "page_number": page_number # Still using page start for simplicity
                    })
    return all_chunks

# --- 4. Embedding and FAISS ---
def create_faiss_index(chunks):
    """
    Generates embeddings for a list of text chunks and creates a FAISS index for efficient similarity search.
    """
    global FAISS_INDEX, TEXT_CHUNKS, CHUNK_METADATA, EMBEDDING_MODEL
    EMBEDDING_MODEL = load_embedding_model() # Ensure embedding model is loaded

    if EMBEDDING_MODEL is None:
        st.error("Embedding model not loaded, cannot create FAISS index.")
        return

    if not chunks:
        st.warning("No text chunks to process for FAISS index.")
        FAISS_INDEX = None
        TEXT_CHUNKS = []
        CHUNK_METADATA = []
        return

    st.info("Generating embeddings and building FAISS index...")
    # Separate chunk content from metadata for embedding and storage
    TEXT_CHUNKS = [chunk["content"] for chunk in chunks]
    CHUNK_METADATA = [{"source_file": c["source_file"], "page_number": c["page_number"]} for c in chunks]

    # Encode texts to get embeddings using the SentenceTransformer model
    # Progress bar is useful for larger document sets
    embeddings = EMBEDDING_MODEL.encode(
        TEXT_CHUNKS,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=DEVICE
    )
    embeddings = embeddings.cpu().numpy() # Move embeddings to CPU for FAISS

    # Ensure embeddings are float32, as required by FAISS
    embeddings = embeddings.astype('float32')

    # Create a FAISS index (IndexFlatL2 uses Euclidean distance for similarity)
    dimension = embeddings.shape[1] # Dimension of the embedding vectors
    FAISS_INDEX = faiss.IndexFlatL2(dimension)
    FAISS_INDEX.add(embeddings) # Add all embeddings to the index

    st.success(f"FAISS index created with {len(TEXT_CHUNKS)} chunks.")

def search_faiss_index(query, k=5):
    """
    Searches the FAISS index for the 'k' most relevant chunks based on a user query.
    Returns a list of dictionaries containing chunk content and its metadata.
    """
    global FAISS_INDEX, TEXT_CHUNKS, CHUNK_METADATA, EMBEDDING_MODEL
    if FAISS_INDEX is None:
        st.error("FAISS index not initialized. Please upload PDFs first.")
        return []
    
    EMBEDDING_MODEL = load_embedding_model() # Ensure embedding model is loaded
    if EMBEDDING_MODEL is None:
        return []

    # Encode the user query into an embedding
    query_embedding = EMBEDDING_MODEL.encode([query], convert_to_tensor=True, device=DEVICE)
    query_embedding = query_embedding.cpu().numpy().astype('float32')

    # Search the FAISS index for the 'k' nearest neighbors
    distances, indices = FAISS_INDEX.search(query_embedding, k)

    relevant_chunks = []
    # Retrieve the actual chunk content and metadata for the found indices
    for i, idx in enumerate(indices[0]):
        if idx < len(TEXT_CHUNKS): # Ensure index is valid
            chunk_content = TEXT_CHUNKS[idx]
            metadata = CHUNK_METADATA[idx]
            relevant_chunks.append({
                "content": chunk_content,
                "source_file": metadata["source_file"],
                "page_number": metadata["page_number"],
                "distance": distances[0][i] # Include distance for potential debugging/sorting
            })
    return relevant_chunks

# --- 5. LLM Answer Generation (RAG) ---
def generate_llm_answer(question, relevant_chunks):
    """
    Generates an answer using the LLM, augmented with retrieved context (RAG).
    Constructs a detailed prompt including the context and explicit instructions for the LLM.
    """
    global LLM_TOKENIZER, LLM_MODEL
    LLM_TOKENIZER, LLM_MODEL = load_llm_model() # Ensure LLM and tokenizer are loaded

    if LLM_MODEL is None or LLM_TOKENIZER is None:
        st.error("LLM model not loaded, cannot generate answer.")
        return "Sorry, the AI model is not available.", "No sources."

    # Combine the content of relevant chunks into a single context string
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    # Format source information for display
    source_info = "\n".join(
        [f"- {chunk['source_file']} (Page {chunk['page_number']})" for chunk in relevant_chunks]
    )

    # Construct the RAG prompt for the LLM
    prompt = f"""You are an academic assistant. Answer the following question truthfully and concisely, based ONLY on the provided context. If the answer cannot be found in the context, state that you don't have enough information from the provided documents.

Context:
{context}

Question: {question}

Answer:"""

    # Prepare the prompt for the chat-templated LLM
    conv = [{"role": "user", "content": prompt}]

    input_ids = LLM_TOKENIZER.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True).to(DEVICE)

    set_seed(42) # For reproducibility of LLM's output
    
    # Generate the answer using the LLM
    output = LLM_MODEL.generate(
        **input_ids,
        max_new_tokens=1024, # Limit output length for answers
        temperature=0.7,    # Controls randomness: lower for more deterministic answers
        do_sample=True,     # Enable sampling for more varied responses
        top_p=0.9           # Nucleus sampling: consider only tokens with cumulative probability up to top_p
    )

    # Decode the generated output to get the text answer
    prediction = LLM_TOKENIZER.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)

    return prediction, source_info

# --- 6. Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="StudyMate AI Assistant")

st.title("📚 StudyMate AI Academic Assistant")
st.markdown("Upload your academic PDFs and ask natural-language questions!")

# Sidebar for PDF uploads
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", type="pdf", accept_multiple_files=True
)

# Process uploaded files if any, and only if they are new or have changed
if uploaded_files:
    # Use session state to track processed files and avoid reprocessing unnecessarily
    if "processed_files_hash" not in st.session_state:
        st.session_state.processed_files_hash = None

    # Create a unique hash for the current set of uploaded files
    current_files_hash = hash(tuple(f.name for f in uploaded_files))

    # If the files are new or different from previously processed ones, start processing
    if st.session_state.processed_files_hash != current_files_hash:
        st.session_state.processed_files_hash = current_files_hash
        st.session_state.all_chunks = []
        st.session_state.pdf_names = []
        
        # Create a temporary directory for uploaded PDFs
        temp_pdf_dir = "./temp_pdfs"
        os.makedirs(temp_pdf_dir, exist_ok=True)

        with st.spinner("Processing PDFs and building knowledge base..."):
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily to process
                file_path = os.path.join(temp_pdf_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.pdf_names.append(uploaded_file.name)
                
                # Extract and chunk text from the saved PDF
                doc_texts = extract_text_from_pdf(file_path)
                if doc_texts:
                    st.session_state.all_chunks.extend(chunk_text(doc_texts))
                
                # Clean up temporary file immediately after processing
                os.remove(file_path)
            
            # After processing all files, create the FAISS index
            if st.session_state.all_chunks:
                create_faiss_index(st.session_state.all_chunks)
            else:
                st.warning("No text extracted from uploaded PDFs. FAISS index will not be built.")
                FAISS_INDEX = None # Reset if no chunks
        
        # Clean up the temporary directory if it's empty
        if not os.listdir(temp_pdf_dir):
            shutil.rmtree(temp_pdf_dir, ignore_errors=True)
            

    # Display status of processed documents
    if "pdf_names" in st.session_state and st.session_state.pdf_names:
        st.sidebar.success(f"Processed {len(st.session_state.pdf_names)} PDFs with {len(st.session_state.all_chunks)} chunks.")
    else:
        st.sidebar.info("No PDFs processed yet.")

# Main area for questions
st.subheader("Ask a Question:")
question = st.text_area("Type your question here:", height=100, key="user_question")

if st.button("Get Answer", type="primary") and question:
    if FAISS_INDEX is not None:
        with st.spinner("Searching and generating answer..."):
            # Search for relevant chunks
            relevant_chunks = search_faiss_index(question, k=5)
            
            if relevant_chunks:
                # Generate answer using RAG
                answer, source_info = generate_llm_answer(question, relevant_chunks)
                st.success("Answer:")
                st.write(answer)
                st.info("Sources:")
                st.markdown(source_info)
            else:
                st.warning("Could not find relevant information in the uploaded documents for your question.")
    else:
        st.warning("Please upload and process PDFs first to enable questioning.")

# Clear session state if no files are uploaded (e.g., user removes all files)
elif "processed_files_hash" in st.session_state:
    # This block ensures state is reset when user clears uploaded files
    del st.session_state.processed_files_hash
    if "all_chunks" in st.session_state:
        del st.session_state.all_chunks
    if "pdf_names" in st.session_state:
        del st.session_state.pdf_names
    # Reset global variables as well
    FAISS_INDEX = None
    TEXT_CHUNKS = []
    CHUNK_METADATA = []
    st.info("No PDFs currently uploaded. Upload your study materials to begin!")

# Initial message when no files are uploaded at all
if not uploaded_files and "processed_files_hash" not in st.session_state:
    st.info("Upload your academic PDFs in the sidebar to start asking questions!")