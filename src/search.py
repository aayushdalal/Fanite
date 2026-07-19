import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

class ArxivSearchEngine:
    def __init__(self, data_dir="../data"):
        print("Initializing Search Engine...")
        
        # 1. Load the dataset
        self.df = pd.read_csv(f"{data_dir}/cleaned_arxiv_papers.csv")
        
        # 2. Load the Embedding Model
        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # 3. Load the FAISS Index
        index_path = f"{data_dir}/paper_faiss.index"
        print("Loading FAISS index...")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run your EDA/Search notebook first.")
            
        print("Search Engine Ready!")

    def search(self, query, k=5):
        """
        Takes a text query, searches the FAISS index, and returns a list of dictionaries
        containing the score, title, and abstract of the top K results.
        """
        # Encode and normalize the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        D, I = self.index.search(query_embedding, k)
        
        # Package the results into a clean list of dictionaries for the LLM
        results = []
        for score, idx in zip(D[0], I[0]):
            paper_data = {
                "score": float(score),
                "title": self.df.iloc[idx]['title'],
                "abstract": self.df.iloc[idx]['abstract']
            }
            results.append(paper_data)
            
        return results