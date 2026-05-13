# rag/ingest.py
# Builds a FAISS vector index from two sources:
#   1. Real NASA PDF — "Damage Propagation Modeling" (Saxena et al.)
#   2. Structured maintenance knowledge base (rag/documents.py)
# Combined index enables grounded LLM recommendations.

import os
import sys
import pickle
import numpy as np
import fitz  # pymupdf — reads PDF files

from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.documents import MAINTENANCE_DOCUMENTS

# ── Configuration ────────────────────────────────────────────────
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_DIR       = 'rag/index'
CHUNK_SIZE      = 250   # words per chunk
CHUNK_OVERLAP   = 40    # overlap to preserve context at boundaries

PDF_SOURCES = [
    {
        'path':  'data/raw/Damage Propagation Modeling.pdf',
        'id':    'nasa_pdf_001',
        'title': 'Damage Propagation Modeling for Aircraft Engine '
                 'Run-to-Failure Simulation (Saxena et al., 2008)'
    }
]


# ── 1. PDF text extraction ───────────────────────────────────────
def extract_pdf_text(pdf_path):
    """
    Extracts all text from a PDF file using PyMuPDF.
    Returns cleaned text as a single string.
    """
    doc   = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Basic cleaning
        # Remove lines that are just page numbers or whitespace
        lines = [
            line.strip()
            for line in text.split('\n')
            if len(line.strip()) > 20  # skip very short lines
        ]
        pages.append('\n'.join(lines))

    doc.close()
    full_text = '\n\n'.join(pages)
    return full_text


# ── 2. Text chunking ─────────────────────────────────────────────
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits long text into overlapping word-level chunks.

    Why overlap? If a key sentence sits at the boundary of two chunks,
    overlap ensures it appears in at least one complete chunk.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])

        # Only keep chunks with enough content to be meaningful
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── 3. Load all documents ────────────────────────────────────────
def load_all_documents():
    """
    Combines two document sources:
    1. Real NASA PDFs from data/raw/
    2. Structured maintenance knowledge from rag/documents.py

    Returns list of dicts with keys: id, title, content, source
    """
    all_docs = []

    # ── Source 1: Real NASA PDFs ──
    print("Loading NASA PDF documents...")
    for pdf_info in PDF_SOURCES:
        path = pdf_info['path']
        if not os.path.exists(path):
            print(f"  WARNING: PDF not found: {path} — skipping")
            continue

        print(f"  Extracting: {os.path.basename(path)}")
        text = extract_pdf_text(path)

        # Split PDF into sections by chunking
        chunks = chunk_text(text)
        print(f"  Extracted {len(text):,} characters → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_docs.append({
                'id':      f"{pdf_info['id']}_chunk{i:03d}",
                'title':   pdf_info['title'],
                'content': chunk,
                'source':  'NASA PDF'
            })

    # ── Source 2: Structured knowledge base ──
    print(f"\nLoading structured maintenance documents...")
    for doc in MAINTENANCE_DOCUMENTS:
        chunks = chunk_text(doc['content'])
        print(f"  {doc['id']}: '{doc['title'][:50]}...' → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_docs.append({
                'id':      f"{doc['id']}_chunk{i:03d}",
                'title':   doc['title'],
                'content': chunk,
                'source':  'Structured Knowledge Base'
            })

    print(f"\nTotal document chunks: {len(all_docs)}")
    print(f"  From NASA PDFs:      "
          f"{sum(1 for d in all_docs if d['source'] == 'NASA PDF')}")
    print(f"  From Knowledge Base: "
          f"{sum(1 for d in all_docs if d['source'] == 'Structured Knowledge Base')}")

    return all_docs


# ── 4. Build FAISS index ─────────────────────────────────────────
def build_index(save=True):
    """
    1. Loads all documents (PDF + structured)
    2. Embeds each chunk into a 384-dim vector
    3. Stores in FAISS index for fast similarity search
    4. Saves index + metadata to disk
    """
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        return None, None

    # Load documents
    all_docs = load_all_documents()

    if not all_docs:
        print("No documents loaded. Check your file paths.")
        return None, None

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(Downloads ~90MB on first run, cached after that)")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Extract just the text for embedding
    texts = [doc['content'] for doc in all_docs]

    # Embed all chunks
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    embeddings_norm = (embeddings / norms).astype(np.float32)

    # Build FAISS index (IndexFlatIP = exact cosine similarity search)
    dimension = embeddings_norm.shape[1]
    index     = faiss.IndexFlatIP(dimension)
    index.add(embeddings_norm)

    print(f"\nFAISS index: {index.ntotal} vectors, {dimension} dimensions")

    if save:
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(INDEX_DIR, 'docs.index')
        faiss.write_index(index, index_path)

        # Save metadata alongside index
        meta_path = os.path.join(INDEX_DIR, 'metadata.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(all_docs, f)

        # Save embedding model name for retriever to load same model
        config_path = os.path.join(INDEX_DIR, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump({'embedding_model': EMBEDDING_MODEL}, f)

        print(f"Saved → {index_path}")
        print(f"Saved → {meta_path}")
        print(f"Saved → {config_path}")

    return index, all_docs


# ── 5. Verify index with a test query ────────────────────────────
def test_index(query="What should I do when RUL is less than 20 cycles?"):
    """
    Quick test: search the index with a sample query
    and print the top 3 most relevant chunks.
    """
    import faiss
    import pickle

    index_path = os.path.join(INDEX_DIR, 'docs.index')
    meta_path  = os.path.join(INDEX_DIR, 'metadata.pkl')

    if not os.path.exists(index_path):
        print("Index not found. Run build_index() first.")
        return

    index    = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    embedder   = SentenceTransformer(EMBEDDING_MODEL)
    query_vec  = embedder.encode([query], convert_to_numpy=True)
    query_norm = (query_vec /
                  np.linalg.norm(query_vec, keepdims=True)).astype(np.float32)

    # Search top 3
    scores, indices = index.search(query_norm, k=3)

    print(f"\nTest query: '{query}'")
    print(f"{'='*60}")
    for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), 1):
        doc = metadata[idx]
        print(f"\nRank {rank} (score={score:.3f})")
        print(f"  Source: {doc['source']}")
        print(f"  Title:  {doc['title'][:60]}")
        print(f"  Text:   {doc['content'][:150]}...")


# ── 6. Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("Building RAG index from NASA PDF + Knowledge Base")
    print("=" * 55)

    index, all_docs = build_index(save=True)

    if index is not None:
        print(f"\n✓ Index built successfully with {index.ntotal} chunks")
        test_index()
        print("\nRAG pipeline ready.")
        print("Next: build rag/retriever.py to use this index")