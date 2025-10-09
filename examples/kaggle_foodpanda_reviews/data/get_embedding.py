#!/usr/bin/env python3
"""
Simple helper script to get embeddings for query strings.
Called by the Go demo to ensure embedding compatibility.
"""
import sys
from sentence_transformers import SentenceTransformer

# Load model once (it will be cached after first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_embedding.py 'your query text'", file=sys.stderr)
        sys.exit(1)
    
    # Get query from command line
    query = " ".join(sys.argv[1:])
    
    # Generate embedding (normalized to match training data)
    embedding = model.encode([query], normalize_embeddings=True)[0]
    
    # Output as comma-separated values (easy to parse in Go)
    print(",".join(map(str, embedding)))