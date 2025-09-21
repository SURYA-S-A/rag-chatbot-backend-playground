# 📘 Knowledge Retrieval & Agentic Chatbot with LangGraph  

This notebook implements a **two-layer architecture** for building intelligent knowledge assistants using **LangGraph + LangChain**:  

---

## 1. 🔎 RAG Pipeline  
- A retrieval-augmented generation (RAG) pipeline.  
- Takes a user query, fetches relevant documents from a vector store, and generates an answer using an LLM.  
- Stateless and one-shot (no memory, no conversation).  
- Supports **file-level filtering** via `selected_files` so users can restrict retrieval to specific uploaded documents.

---

## 2. 🤖 Knowledge Bot (Chatbot)  
- An **agentic AI chatbot** built on top of the `RAGPipeline`.  
- Maintains **conversation state** and **memory** across turns.  
- Uses **tools** (`rag_retrieval`, `calculator`, etc.) to answer user questions.  
- Always tries to search uploaded documents via the RAG pipeline before fallback responses.  
- Respects `selected_files` metadata passed in user queries, allowing focused Q&A on only those chosen documents.  

---

## 🔑 Key Concepts
- **Separation of Concerns**  
  - `RAG Pipeline` = retrieval + generation engine.  
  - `Knowledge Bot` = chat agent with memory, reasoning, and tool use.  

- **File Selection**  
  - The `selected_files` field gives users **fine-grained control** over which documents are searched.  
  - If not provided, the pipeline searches across the full collection.  

- **Extensibility**  
  - Tools can be added (e.g., web search, code execution).  
  - Streaming responses can be enabled for real-time chat experiences.  

- **Use Case**  
  - Ideal for applications where users upload documents and query them interactively.  
  - Example: knowledge assistants, enterprise search, personal document Q&A.  

---
