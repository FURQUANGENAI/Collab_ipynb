# NVIDIA AI Endpoints with LangChain

This repository contains a Jupyter notebook (`nvidia_ai_endpoints.ipynb`) that demonstrates how to use the `langchain-nvidia-ai-endpoints` package to interact with NVIDIA's NIM (NVIDIA Inference Microservices) for embedding models, specifically for retrieval-augmented generation (RAG) workflows. The notebook covers setup, embedding generation, similarity computation, input truncation, and a RAG pipeline using NVIDIA's cloud-hosted models.

## Overview

The notebook integrates LangChain with NVIDIA NIMs, which are optimized AI models (e.g., `NV-Embed-QA` for embeddings and `Mixtral-8x7B-Instruct` for chat) deployed as containers for high performance on NVIDIA hardware. It includes:
- Installation and setup of the `langchain-nvidia-ai-endpoints` package.
- Generating embeddings for queries and documents using `NVIDIAEmbeddings`.
- Computing and visualizing cosine similarity between query and document embeddings.
- Handling long inputs with truncation options.
- Building a RAG pipeline for question answering, including multilingual support.

## Prerequisites

1. **Python Environment**: Python 3.10 or later.
2. **NVIDIA API Key**:
   - Create an account on [NVIDIA API Catalog](https://build.nvidia.com/).
   - Select a retrieval model, generate an API key, and set it as the `NVIDIA_API_KEY` environment variable.
3. **Dependencies**:
   - Install required packages by running:
     ```bash
     pip install langchain-nvidia-ai-endpoints langchain faiss-cpu tiktoken langchain_community matplotlib scikit-learn
