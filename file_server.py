#!/usr/bin/env python
"""Example that shows how to upload files and process files in the server.

This example uses a very simple architecture for dealing with file uploads
and processing.

The main issue with this approach is that processing is done in
the same process rather than offloaded to a process pool. A smaller
issue is that the base64 encoding incurs an additional encoding/decoding
overhead.

This example also specifies a "base64file" widget, which will create a widget
allowing one to upload a binary file using the langserve playground UI.
"""
import base64
from typing import List

from fastapi import FastAPI
from langchain.pydantic_v1 import Field
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_core.document_loaders import Blob
from langchain_core.runnables import RunnableLambda

from langserve import CustomUserType, add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

# CustomUserType now allows for a list of files
class FileProcessingRequest(CustomUserType):
    """Request including a list of base64 encoded files and a question."""

    # The extra field is used to specify a widget for the playground UI.
    files: List[str] = Field(..., extra={"widget": {"type": "base64file"}})
    num_chars: int = 100
    question: str = ""  # Field to handle user question

def _process_file(request: FileProcessingRequest) -> List[str]:
    """Extract text from each uploaded PDF and optionally process a question."""
    results = []

    for file_data in request.files:
        if file_data:  # Check if file data is not empty
            content = base64.b64decode(file_data.encode("utf-8"))
            blob = Blob(data=content)
            documents = list(PDFMinerParser().lazy_parse(blob))
            extracted_text = documents[0].page_content[: request.num_chars]
        else:
            extracted_text = "No PDF file provided."

        if request.question:
            # Simple question-answering logic (could be expanded)
            if request.question.lower() in extracted_text.lower():
                results.append(
                    f"Found your query in the text: '{request.question}'\n\nExtracted text: {extracted_text}"
                )
            else:
                results.append(
                    f"Query '{request.question}' not found in the extracted text.\n\nExtracted text: {extracted_text}"
                )
        else:
            results.append(extracted_text)

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
