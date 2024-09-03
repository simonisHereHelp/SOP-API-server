#!/usr/bin/env python
"""Example that shows how to upload and process PDF files in the server using PyMuPDFLoader.

This example uses PyMuPDFLoader from langchain_community.document_loaders.pdf to handle PDF files.

Uploaded PDFs are temporarily stored on disk, processed, and then deleted after processing.
"""
import os
import sys
import base64
import tempfile
from typing import List, Optional, Dict

from fastapi import FastAPI
from langchain.pydantic_v1 import Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders.pdf import PyMuPDFLoader

from langserve import CustomUserType, add_routes

# Ensure the utils package is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utils')
sys.path.append(utils_dir)

# Importing the create_vector_store function
from utils.vector_store import create_vector_store

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

# CustomUserType now allows for a list of files
class FileProcessingRequest(CustomUserType):
    """Request including a list of base64 encoded files and a question."""

    files: List[str] = Field(..., extra={"widget": {"type": "base64file"}})
    question: str = ""  # Field to handle user question

def _process_file(request: FileProcessingRequest) -> List[str]:
    """Process each uploaded PDF using OpenAI embeddings and FAISS retriever."""
    results = []

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Decode and load each file, then create a vector store
    for file_data in request.files:
        if file_data:  # Check if file data is not empty
            content = base64.b64decode(file_data.encode("utf-8"))
            
            # Temporarily save the file to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # Use PyMuPDFLoader to process the PDF content from the temporary file
                loader = PyMuPDFLoader(tmp_file_path)
                documents = loader.load()

                # Create a vector store
                uploaded_store = create_vector_store(documents)
                retriever = uploaded_store.as_retriever()

                # LLM model
                llm = ChatOpenAI(model="gpt-4", temperature=0)

                # Generate response based on the question using the retriever
                if request.question:
                    response = retriever.get_relevant_documents(request.question)
                    results.append(f"Response to '{request.question}': {response}")
                else:
                    results.append("No question provided.")
            finally:
                # Ensure the temporary file is deleted after processing
                os.remove(tmp_file_path)
        else:
            results.append("No PDF file provided.")

    return results

add_routes(
    app,
    RunnableLambda(_process_file).with_types(input_type=FileProcessingRequest),
    config_keys=["configurable"],
    path="/pdf",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
