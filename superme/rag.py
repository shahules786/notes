import json
import os
import re
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import aiofiles
from pydantic import BaseModel, Field
import instructor
from langfuse.openai import AsyncOpenAI
from langfuse.decorators import observe


class AgenticResponseModel(BaseModel):
    """Response model for retrieval method."""
    text: str


class Document:
    """Class to represent a LinkedIn post as a document."""
    
    def __init__(self, id: str, content: str, url: str, date: str):
        self.id = id
        self.content = content
        self.url = url
        self.date = date
        self.embedding = None  # Will store the vector embedding
        
    def __str__(self):
        return f"Post {self.id}: {self.content[:50]}... (Posted: {self.date})"

class LinkedInPostsRAG:
    """RAG system for LinkedIn posts with both BM25 and vector search."""
    
    def __init__(self, data_file: str = "linkedin_posts.json"):
        # Initialize OpenAI client with Instructor
        openai_client = AsyncOpenAI()
        self.client = instructor.from_openai(openai_client)
        
        self.documents: List[Document] = []
        self.document_vectors: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.data_file = data_file
        self.vector_file = data_file.replace(".json", "_vectors.npy")
        
    async def load_data(self) -> None:
        """Load JSON data file with LinkedIn posts."""
        if os.path.exists(self.data_file):
            try:
                async with aiofiles.open(self.data_file, mode='r', encoding='utf-8') as file:
                    content = await file.read()
                    posts_data = json.loads(content)
                    
                for post_id, post_info in posts_data.items():
                    doc = Document(
                        id=post_id,
                        content=post_info.get("content", ""),
                        url=post_info.get("url", ""),
                        date=post_info.get("date", "unknown")
                    )
                    self.documents.append(doc)
                
                print(f"Loaded {len(self.documents)} LinkedIn posts")
            except Exception as e:
                print(f"Error loading data: {e}")
        else:
            print(f"Data file {self.data_file} not found")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better indexing."""
        # Convert to lowercase
        text = text.lower()
        # Replace newlines with spaces
        text = re.sub(r'\n+', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove special characters but keep spaces and alphanumerics
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    async def initialize_bm25(self) -> None:
        """Initialize BM25 index for text retrieval."""
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in self.documents:
            processed_text = self.preprocess_text(doc.content)
            tokenized_docs.append(processed_text.split())
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        print("BM25 index initialized")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the OpenAI Embeddings API."""
        
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        
        return response.data[0].embedding
    
    async def initialize_vector_search(self) -> None:
        """Initialize vector search by creating embeddings for all documents."""
        if not self.documents:
            print("No documents loaded. Please load data first.")
            return
        
        # Check if vectors are already saved
        if os.path.exists(self.vector_file):
            try:
                self.document_vectors = np.load(self.vector_file)
                print(f"Loaded existing document vectors from {self.vector_file}")
                # Assign vectors to documents
                for i, doc in enumerate(self.documents):
                    doc.embedding = self.document_vectors[i]
                return
            except Exception as e:
                print(f"Error loading vectors, will regenerate them: {e}")
        
        # Generate embeddings for all documents
        embeddings = []
        for i, doc in enumerate(self.documents):
            print(f"Generating embedding for document {i+1}/{len(self.documents)}...")
            embedding = await self.get_embedding(doc.content)
            doc.embedding = embedding
            embeddings.append(embedding)
        
        # Convert embeddings to numpy array for vector search
        self.document_vectors = np.array(embeddings)
        
        # Save embeddings to file
        np.save(self.vector_file, self.document_vectors)
        print(f"Saved document vectors to {self.vector_file}")
    
    async def initialize(self) -> None:
        """Initialize both retrieval methods."""
        await self.load_data()
        await self.initialize_bm25()
        await self.initialize_vector_search()
    
    def retrieve_with_bm25(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant documents using BM25."""
        if not self.bm25:
            raise Exception("BM25 index not initialized")
        
        # Preprocess and tokenize the query
        processed_query = self.preprocess_text(query)
        query_tokens = processed_query.split()
        
        # Get BM25 scores
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Get indices of top k documents
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]
    
    async def retrieve_with_vectors(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant documents using vector similarity."""
        if self.document_vectors is None:
            raise Exception("Vector index not initialized")
        
        # Get query embedding
        query_embedding = await self.get_embedding(query)
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarity
        similarities = np.dot(self.document_vectors, query_vector) / (
            np.linalg.norm(self.document_vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get indices of top k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]
    
    async def retrieve_combined(self, query: str, top_k: int = 3, weight_bm25: float = 0.5) -> List[Document]:
        """Retrieve documents using both BM25 and vector search with weighted combination."""
        # Get BM25 results
        if not self.bm25:
            raise Exception("BM25 index not initialized")
        
        processed_query = self.preprocess_text(query)
        query_tokens = processed_query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        normalized_bm25 = bm25_scores / (max(bm25_scores) if max(bm25_scores) > 0 else 1)
        
        # Get vector results
        query_embedding = await self.get_embedding(query)
        query_vector = np.array(query_embedding)
        
        similarities = np.dot(self.document_vectors, query_vector) / (
            np.linalg.norm(self.document_vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Combine scores with weights
        combined_scores = (weight_bm25 * normalized_bm25) + ((1 - weight_bm25) * similarities)
        
        # Get indices of top k documents
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # Return top documents
        return [self.documents[i] for i in top_indices]
    
    @observe()
    async def agentic_retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        
        data = json.load(open('/Users/shahules/Myprojects/notes/superme/data/shahules_profile.json'))
        response = await self.client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=AgenticResponseModel,
        messages=[
            {"role": "system", "content": f"Retrieve the chunk from the JSON file that is required to answer the question :{query}. If data is not useful for answering the query, return an empty str."},
            {"role": "user", "content": json.dumps(data)}
        ])
        chunks = [Document(id="0",content=f"PROFILE INFO\n{response.text}",date="",url="https://www.linkedin.com/in/shahules/")]
         
        relevant_docs = await self.retrieve_with_vectors(query, top_k)
        return  chunks + relevant_docs

    class RagResponse(BaseModel):
        """Response model for RAG system using Instructor."""
        answer: str = Field(description="The answer to the user's question based on the provided context")
        reasoning: str = Field(description="Step-by-step reasoning process to arrive at the answer")
        sources: List[str] = Field(description="URLs of LinkedIn posts used as sources for the answer")
    
    @observe()
    async def answer_query(
        self, 
        query: str, 
        retrieval_method: str = "combined", 
        top_k: int = 3,
        weight_bm25: float = 0.5
    ) -> RagResponse:
        """Answer a query using the RAG system with specified retrieval method."""
        # Retrieve relevant documents
        relevant_docs = await self.retrieve(query, retrieval_method, top_k, weight_bm25)
        
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Post {i+1} (Date: {doc.date}):\n{doc.content}\nURL: {doc.url}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Generate answer using instructor and Pydantic model
        system_message = "You are a helpful assistant that provides information based on given context. Do not add any additional information not present in the context."
        user_message = f"Here is the context:\n\n{context}\n\nBased on this information, please answer: {query}"
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            response_model=self.RagResponse,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        
        return response
    
    @observe()
    async def retrieve(self, query: str, retrieval_method: str = "combined", top_k: int = 3, weight_bm25: float = 0.5) -> List[Document]:
        
        
        if retrieval_method == "bm25":
            relevant_docs = self.retrieve_with_bm25(query, top_k)
        elif retrieval_method == "vector":
            relevant_docs = await self.retrieve_with_vectors(query, top_k)
        elif retrieval_method == "combined":
            relevant_docs = await self.retrieve_combined(query, top_k, weight_bm25)
        elif retrieval_method == "agentic":
            relevant_docs = await self.agentic_retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")
        
        return relevant_docs

async def main():
    """Main function for demo purposes."""
    # Initialize RAG system
    rag = LinkedInPostsRAG("/Users/shahules/Myprojects/notes/superme/data/shahules_posts.json")
    await rag.initialize()
    
    # Example queries
    queries = [
        "What are your thoughts on RAG?",
        "How do you evaluate AI systems?",
        "Tell me about the relationship between human feedback and AI systems.",
    ]
    
    # Demo different retrieval methods
    for query in queries:
        print(f"\n\n==== Query: {query} ====")
        
        # Try different retrieval methods
        for method in ["bm25", "vector", "combined"]:
            print(f"\n-- Using {method} retrieval --")
            response, docs = await rag.answer_query(query, retrieval_method=method, top_k=2)
            
            print("Retrieved documents:")
            for i, doc in enumerate(docs):
                print(f"{i+1}. {doc}")
            
            print("\nAnswer:")
            print(response.answer)
            
            print("\nReasoning:")
            print(response.reasoning)
            
            print("\nSources:")
            for source in response.sources:
                print(f"- {source}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv('/Users/shahules/Myprojects/notes/.envrc')
    
    asyncio.run(main())