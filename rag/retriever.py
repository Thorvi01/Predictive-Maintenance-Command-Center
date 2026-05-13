# rag/retriever.py
# Retrieval function — searches the FAISS index and returns
# relevant document chunks as formatted context for the LLM.

import os
import sys
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Configuration ────────────────────────────────────────────────
INDEX_DIR  = 'rag/index'
TOP_K      = 3   # number of chunks to retrieve per query


# ── Retriever class ──────────────────────────────────────────────
class MaintenanceRetriever:
    """
    Loads the FAISS index once and provides a retrieve() method.
    Designed to be instantiated once and reused across queries.
    """

    def __init__(self, index_dir=INDEX_DIR):
        import faiss

        index_path  = os.path.join(index_dir, 'docs.index')
        meta_path   = os.path.join(index_dir, 'metadata.pkl')
        config_path = os.path.join(index_dir, 'config.pkl')

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                f"Run 'python rag/ingest.py' first."
            )

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load embedding model name
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Load sentence transformer
        model_name    = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedder = SentenceTransformer(model_name)

        print(f"Retriever ready — {self.index.ntotal} chunks indexed")

    def retrieve(self, query, top_k=TOP_K):
        """
        Searches index for chunks most semantically similar to query.

        Args:
            query:  natural language question or statement
            top_k:  number of chunks to return

        Returns:
            List of dicts with keys: title, source, text, score
        """
        # Embed the query
        query_vec  = self.embedder.encode(
            [query], convert_to_numpy=True
        )
        # Normalize for cosine similarity
        query_norm = (query_vec /
                      np.linalg.norm(query_vec, keepdims=True)
                      ).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_norm, k=top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 if not enough results
                continue
            doc = self.metadata[idx]
            results.append({
                'title':  doc['title'],
                'source': doc['source'],
                'text':   doc['content'],
                'score':  float(score)
            })

        return results

    def format_context(self, query, top_k=TOP_K):
        """
        Retrieves relevant chunks and formats them as a context
        string ready to be injected into an LLM prompt.

        Returns:
            context_str: formatted string with source citations
            results:     raw list of retrieved chunks
        """
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return "No relevant documentation found.", []

        lines = ["RELEVANT MAINTENANCE DOCUMENTATION:", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"[Source {i}] {r['title']}")
            lines.append(f"Origin: {r['source']}")
            lines.append(f"{r['text']}")
            lines.append("")

        context_str = '\n'.join(lines)
        return context_str, results


# ── Quick test ───────────────────────────────────────────────────
if __name__ == '__main__':
    retriever = MaintenanceRetriever()

    test_queries = [
        "Engine RUL is 15 cycles, what action should I take?",
        "What does T30 sensor indicate about HPC degradation?",
        "How do I detect model drift in production?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 55)
        context, results = retriever.format_context(query)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['source']} — "
                  f"{r['title'][:50]}")
            print(f"           {r['text'][:100]}...")