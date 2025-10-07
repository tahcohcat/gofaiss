# gofaiss Roadmap

This roadmap outlines the planned development path for **gofaiss**, from its current state toward a mature, production-ready Go-native vector search library.
The following are just ideas on what could be done in a loose logical progression. It is by no means set  in stone or guaranteed. 

---

## Phase 0: Foundation (current / immediate)

**Goals:**
- Solidify core index and metric implementations
- Provide basic persistence and benchmark scaffolding
- Launch initial version (v0.1)

**Planned Deliverables:**
- Flat (exact) index  
- IVF, IVFPQ, HNSW index types  
- Metrics: L2, Cosine, Inner Product  
- Serialization / Save & Load (gob, JSON, gzip)  
- Benchmark suite (simple datasets)  
- Basic CLI tool  
- README, docs, and project structure  
- Tag v0.1 release  

---

## Phase 1: Developer Experience & Stability (v0.2 → v0.3)

**Goals:**
- Make gofaiss easier to adopt, test, and contribute to  
- Harden the library for everyday use

**Key Items:**
- GitHub Actions CI / nightly builds  
- More benchmark comparisons vs popular alternatives (e.g. hnswlib-go, faiss-cgo)  
- Example applications / sample datasets  
- Interface stability & GoDoc enhancements  
- Better error handling and validation  
- Benchmarks with more realistic datasets (e.g. SIFT1M, GloVe)  
- “Good first issue” tags for new contributors  
- Release v0.2  

---

## Phase 2: Performance, Scaling & Concurrency (v0.3 → v0.4)

**Goals:**
- Enable larger datasets, faster queries, concurrent use  
- Reduce memory overheads, support persistence for large indexes

**Features & Enhancements:**
- Concurrency-safe `Search` / `Add` operations  
- Sharded or segmented “Flat” and “IVF” partitions  
- Memory-mapped backing (mmap) for large indexes  
- Batch query support  
- Hybrid index combinations (e.g. IVF + HNSW)  
- Parallelized search / probe for IVF  
- SIMD / vectorized distance computation optimizations  
- Incremental updates / deletions to index  
- Release v0.3  

---

## Phase 3: Serving, Ecosystem & Integration (v0.4 → v0.5)

**Goals:**
- Turn gofaiss into a building block in larger systems  
- Interoperate with other tools, serve remote clients

**Features & Integrations:**
- HTTP / gRPC API for query / index operations  
- Docker + Helm charts  
- Integration adapters (e.g. embedding pipelines, vector DBs)  
- Example embedding ingestion pipelines (e.g. CLIP, ONNX)  
- Monitoring, metrics endpoints (latency, memory usage)  
- Snapshotting, versioning, merging indexes  
- Release v0.4  

---

## Phase 4: Maturity & Community (v0.5 → v1.0)

**Goals:**
- Establish stable APIs, expand ecosystem, attract users & contributors  
- Ensure production robustness

**Key Milestones:**
- API stability / semantic versioning  
- Extended metric support (Manhattan, Hamming, Jaccard etc.)  
- Support for very large scale (10M+ vectors)  
- Distributed indexing / sharding across nodes  
- Community benchmark suite (contributors can add datasets)  
- Governance / contribution guidelines  
- Release v1.0  

---

## Ideas & Stretch Goals

- Disk-resident graph caching (for HNSW)  
- Hybrid quantization + indexing methods  
- Support GPU acceleration (via external bindings)  
- WebAssembly build target  
- Client libraries (Python / JS) for remote access  
- Auto-tuning of index parameters for a given dataset  

---

## How You Can Help

- Try using gofaiss and report issues or performance comparisons  
- Submit benchmark results or dataset support  
- Implement new metric / index types  
- Help writing docs, tutorials, examples  
- Review PRs and propose enhancements  

---

