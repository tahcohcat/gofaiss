# 🍔🐼 FoodPanda Reviews Semantic Search Demo

An interactive demo showing how GoFAISS can power semantic search over real restaurant reviews from Bangladesh.

## What Does This Demo Do?

This demo lets you search through 1,000 real restaurant reviews using **natural language**. Instead of matching exact keywords, it understands the *meaning* of your query.

**Try it yourself:**
- Search for "delicious biryani" → finds reviews mentioning tasty biryani, even if they say "amazing rice dish"
- Search for "slow delivery" → finds complaints about wait times, poor service, late orders
- Search for "best chinese food" → surfaces positive reviews about Chinese restaurants

The demo uses **vector similarity search** powered by GoFAISS's HNSW (Hierarchical Navigable Small World) index for fast, accurate results.

---

## Quick Start

### Prerequisites

- **Go 1.21+** - [Install Go](https://go.dev/doc/install)
- **Python 3.8+** with `sentence-transformers` - For generating embeddings
  ```bash
  pip install sentence-transformers
  ```

### Setup (5 minutes)

**1. Get the dataset**

Download from Kaggle: [BD Food Review Dataset](https://www.kaggle.com/datasets/sanjidh090/bd-food-review-dataset)

Or use the Kaggle CLI:
```bash
kaggle datasets download -d sanjidh090/bd-food-review-dataset
```

**2. Generate embeddings**

Save this as `generate_data.py`:
```python
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Update this path to where you downloaded the dataset
path = "path/to/BDFoodSent-334k.csv"
df = pd.read_csv(path, dtype={17: str})

# Create searchable text combining review + metadata
df["text_full"] = (
    df["text"].fillna("") + " | Restaurant: " + df["name"].fillna("") +
    " | Cuisine: " + df["primary_cuisine"].fillna("") +
    " | City: " + df["city"].fillna("")
)

# Use first 1000 reviews for demo
texts = df["text_full"].tolist()[:1000]

# Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# Save files
os.makedirs("examples/kaggle_foodpanda_reviews/data", exist_ok=True)
np.savetxt("examples/kaggle_foodpanda_reviews/data/embeddings_1k.csv", embs, delimiter=",")
with open("examples/kaggle_foodpanda_reviews/data/texts_1k.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

print(f"✓ Generated {len(texts)} embeddings (dim={embs.shape[1]})")
```

Run it:
```bash
python generate_data.py
```

**3. Create the embedding helper**

Save this as `examples/kaggle_foodpanda_reviews/data/get_embedding.py`:
```python
#!/usr/bin/env python3
import sys
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_embedding.py 'your query'", file=sys.stderr)
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    embedding = model.encode([query], normalize_embeddings=True)[0]
    print(",".join(map(str, embedding)))
```

Test it:
```bash
cd examples/kaggle_foodpanda_reviews/data
python get_embedding.py "delicious food"
# Should print 384 comma-separated numbers
```

**4. Run the demo**

```bash
cd ../../..  # back to project root
go run examples/kaggle_foodpanda_reviews/main.go
```

---

##  How to Use

Once running, you'll see:
```
🍔🐼 FoodPanda Reviews Semantic Search Demo
==========================================

Loading embeddings... ✓ Loaded 1000 embeddings (dim=384)
Loading reviews... ✓ Loaded 1000 reviews
Building HNSW index... ✓ Index built

Ready to search! Type your query (or 'quit' to exit)
Example queries:
  - delicious biryani
  - bad service slow delivery
  - best chinese food

> 
```

Type your query and press Enter. The demo will:
1. Convert your query to an embedding (using the same model as the dataset)
2. Search the HNSW index for the 5 most similar reviews
3. Display results with similarity scores

**Example:**
```
> amazing pizza

Getting embedding... ✓

 Top 5 matches:
--------------------------------------------------------------------------------

1. [Similarity: 0.892]
   Review: Best pizza in town! Crispy crust and fresh toppings
   Restaurant: Pizza Hut | Cuisine: Italian | City: Dhaka

2. [Similarity: 0.854]
   Review: Great pizza, arrived hot and delicious
   Restaurant: Dominos | Cuisine: Fast Food | City: Chittagong
   
...
--------------------------------------------------------------------------------
```

Type `quit` or `exit` to stop.

---

## What's Happening Under the Hood?

### 1. **Embeddings** (Vector Representations)
Each review is converted into a 384-dimensional vector that captures its semantic meaning. Similar reviews have similar vectors.

```
"delicious pizza" → [0.23, -0.45, 0.67, ...]
"tasty pie"       → [0.25, -0.43, 0.69, ...]  ← Very similar!
"slow delivery"   → [-0.12, 0.89, -0.34, ...] ← Very different
```

### 2. **HNSW Index** (Fast Search)
Instead of comparing your query against all 1,000 reviews (slow!), HNSW builds a graph structure that lets you jump directly to similar items.

- **Without HNSW:** Compare against all 1,000 reviews → ~10ms
- **With HNSW:** Navigate graph → ~0.5ms (20x faster!)
- Scales to millions of vectors with minimal slowdown

### 3. **Cosine Similarity** (Measuring Closeness)
We measure how similar two vectors are using cosine similarity:
- **1.0** = identical meaning
- **0.0** = completely unrelated
- Works great for text because it ignores length, only cares about direction

---

## What This Demo Shows

### Core GoFAISS Features

 * **HNSW Index**: Fast approximate nearest neighbor search  
 *  **Cosine Similarity**: Semantic similarity metric  
 * **Real-time Search**: Sub-millisecond query times  
 * **Vector Operations**: Efficient storage and retrieval  

### Real-World Use Cases

This same approach powers:
-  **Search engines** - Google, Bing semantic search
-  **Chatbots** - RAG (Retrieval Augmented Generation)
-  **Recommendation systems** - Netflix, Spotify
-  **Image search** - Pinterest, Google Images
-  **Document retrieval** - Legal, medical research

---

## Performance

**Dataset:** 1,000 reviews, 384 dimensions

| Operation | Time | Details |
|-----------|------|---------|
| Load embeddings | ~50ms | Read CSV file |
| Build HNSW index | ~200ms | M=16, efConstruction=200 |
| Search (per query) | ~0.5ms | Returns top 5 results |
| Generate query embedding | ~50ms | Via Python (one-time per query) |

**Total search time: ~50ms** (mostly embedding generation)

For comparison, a brute-force search would take ~10ms for 1,000 vectors. The HNSW advantage becomes dramatic with larger datasets (10,000+ vectors).

---

## Customization

### Search more results
Change `k` in the code:
```go
k := 10  // Get top 10 instead of top 5
results, err := index.Search(qEmb, k)
```

### Use different HNSW settings
```go
index, err := hnsw.New(dim, "cosine", hnsw.Config{
    M:              32,  // More neighbors = better accuracy, more memory
    EfConstruction: 400, // Higher = better build quality, slower build
    EfSearch:       200, // Higher = better search accuracy, slower search
})
```

### Use more reviews
Change `n = 1000` in `generate_data.py` to use more of the 334k dataset.

---

## Troubleshooting

**"Python not found"**
- Install Python 3.8+ and add to PATH
- Or use `python3` instead of `python` on Linux/Mac

**"No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**"embeddings_1k.csv not found"**
- Make sure you ran `generate_data.py` first
- Check the file is in `examples/kaggle_foodpanda_reviews/data/`

**"Dimension mismatch"**
- Embeddings must be 384-dimensional (from all-MiniLM-L6-v2)
- Don't mix embeddings from different models

**Search returns weird results**
- Make sure embeddings are normalized (use `normalize_embeddings=True`)
- Check that query embedding uses the same model

---

## Learn More

- **GoFAISS Documentation**: [github.com/tahcohcat/gofaiss](https://github.com/tahcohcat/gofaiss)
- **HNSW Algorithm**: [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
- **Sentence Transformers**: [sbert.net](https://www.sbert.net)
- **Vector Search Explained**: [pinecone.io/learn/vector-database](https://www.pinecone.io/learn/vector-database/)

---

## Next Steps

Want to build your own vector search application? Try:

1. **Use your own data**: Replace reviews with your documents, products, images
2. **Scale up**: Test with 10k, 100k, 1M vectors
3. **Add filtering**: Combine vector search with metadata filters (cuisine, city, rating)
4. **Try different indexes**: Compare HNSW vs IVF vs Flat for your use case
5. **Build an API**: Wrap this in a REST or gRPC server

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/tahcohcat/gofaiss/issues)

💙 **Happy searching!** 
