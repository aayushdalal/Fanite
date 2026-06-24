# ArXiv Vector Search: A Bi-Encoder Semantic Search Engine

## Overview

A high-performance semantic search engine built to instantly query thousands of Machine Learning research papers from ArXiv. Unlike traditional keyword-based search (which relies on exact text matches), this engine uses Deep Learning (Sentence Transformers) to understand the context and intent behind a user's query, matching it with the semantic meaning of the research papers.

---

## 🛠️ Tech Stack

**Language:** Python

**NLP / Deep Learning:** sentence-transformers (Hugging Face)

**Vector Database Engine:** faiss-cpu (Meta/Facebook AI Similarity Search)

**Data Processing:** pandas, numpy, datasets

**Model:** all-MiniLM-L6-v2 (Bi-Encoder architecture, 384-dimensional dense vectors)

---

## 🧠 System Architecture

This project is separated into two distinct pipelines to simulate a production environment:

### 1. Data Pipeline (EDA.ipynb)

**Ingestion:** Extracts over 117,000 ML papers from the Hugging Face CShorten/ML-ArXiv-Papers dataset.

**Preprocessing:** Cleans structural artifacts (rogue `\n`, trailing whitespaces) and concatenates titles with abstracts to create a context-rich text feature.

**Vectorization:** Passes the cleaned text through the all-MiniLM-L6-v2 neural network in batches to prevent Out-Of-Memory (OOM) errors. Compresses the semantic meaning of each paper into a 384-dimensional dense vector.

**Storage:** Saves the cleaned text as a CSV and the mathematical embeddings as a highly compressed `.npy` binary file for instant loading.

---

### 2. Search & Inference Pipeline (Search_Engine.ipynb)

**FAISS Indexing:** Initializes an `IndexFlatIP` (Inner Product) FAISS index in memory.

**The "Math Hack" (L2 Normalization):** Because FAISS lacks a native Cosine Similarity function, the 384-dimensional document vectors and the user query vectors are transformed using L2 Normalization. By normalizing the vectors to a length of 1 and using FAISS's blazing-fast Inner Product engine, the system calculates mathematically perfect Cosine Similarity scores in sub-milliseconds.

**Retrieval:** Converts the user's natural language query into a vector, queries the FAISS index, and retrieves the top-K most semantically relevant research papers instantly.

---

## 🚀 How to Run Locally

### 1. Install Dependencies

```bash
pip install pandas numpy sentence-transformers faiss-cpu datasets scikit-learn
```

### 2. Generate the Embeddings (Data Pipeline)

Run all cells in `notebooks/EDA.ipynb`.

**Note:** This will download the dataset and the Hugging Face model. The batch encoding process will take a few minutes depending on your CPU/GPU. It will output `cleaned_arxiv_papers.csv` and `arxiv_embeddings.npy`.

### 3. Run the Search Engine (Inference)

Open `notebooks/Search_Engine.ipynb`.

This notebook loads the pre-computed embeddings instantly (skipping the heavy computation).

Modify the `query` variable in the final cell to test out different semantic searches (e.g., `"deep learning for medical image analysis"`).

---

## 💡 Key Learnings & Optimizations

### Bi-Encoders over Cross-Encoders

Used a Siamese network architecture (SentenceTransformers) to pre-compute document embeddings. This reduced search time from potential hours (via Cross-Encoder brute force) to milliseconds.

### FAISS Implementation

Transitioned from sklearn's `$O(N)$` exhaustive search to Meta's highly optimized FAISS library to ensure the system is scalable to millions of documents.

### Data Type Optimization

Utilized `float32` data types and `.npy` binary storage to balance mathematical precision with RAM efficiency.
