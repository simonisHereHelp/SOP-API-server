import os
import pickle
import sys

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the main directory to the Python path
sys.path.append(os.path.dirname(current_dir))

# Add the utils package to the Python path
utils_dir = os.path.join(current_dir, 'utils')
sys.path.append(utils_dir)
data_directory = os.path.join(current_dir, 'sop')

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Any, Union, Dict
from utils.vector_store import create_vector_store
from utils.grader import GraderUtils
from utils.graph import GraphState
from utils.generate_chain import create_generate_chain
from utils.nodes import GraphNodes
from utils.edges import EdgeGraph
from langgraph.graph import END, StateGraph
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # important line if cannot load api key
from typing import Optional
## Getting the api keys from the .env file

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# WebSearch tools
# os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')
# os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
# os.environ['GOOGLE_CSE_ID'] = os.getenv('GOOGLE_CSE_ID')
# os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

# Langsmith Tracing
# os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
# os.environ['LANGCHAIN_ENDPOINT'] = os.getenv('LANGCHAIN_ENDPOINT')
# os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Fire Crawl API
# os.environ['FIRE_API_KEY']=os.getenv('FIRE_API_KEY')

## Create Retriever

# embedding model
embedding_model = OpenAIEmbeddings()

def load_pdfs_from_directory(directory_path):
    all_documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
    return all_documents

# Directory containing the PDF file



# Load all PDFs from the directory
saved_docs = load_pdfs_from_directory(data_directory)

# create vector store
store = create_vector_store(saved_docs)

# creating retriever
retriever = store.as_retriever()

## LLM model
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Create the generate chain
generate_chain = create_generate_chain(llm)

## get the grader instances

# Create an instance of the GraderUtils class
grader = GraderUtils(llm)

# Get the retrieval grader
retrieval_grader = grader.create_retrieval_grader()

# Get the hallucination grader
hallucination_grader = grader.create_hallucination_grader()

# Get the code evaluator
code_evaluator = grader.create_code_evaluator()

# Get the question rewriter
question_rewriter = grader.create_question_rewriter()

## Creating the WorkFlow

# Initiating the Graph
workflow = StateGraph(GraphState)

# Create an instance of the GraphNodes class
graph_nodes = GraphNodes(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)

# Create an instance of the EdgeGraph class
edge_graph = EdgeGraph(hallucination_grader, code_evaluator)

# Define the nodes
workflow.add_node("retrieve", graph_nodes.retrieve) # retrieve documents
workflow.add_node("grade_documents", graph_nodes.grade_documents)  # grade documents
workflow.add_node("generate", graph_nodes.generate) # generate answers
workflow.add_node("transform_query", graph_nodes.transform_query)  # transform_query

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    edge_graph.decide_to_generate,
    {
        "transform_query": "transform_query", # "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    edge_graph.grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query", # "transform_query"
    },
)

# Compile
chain = workflow.compile()

## Create the FastAPI app

app = FastAPI(
    title="SOP Server",
    version="1.0",
    description="An API server to answer questions regarding SOP Docs"
    
)

# Fetch allowed origins from environment variables
origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the directory to save uploaded files
UPLOAD_DIRECTORY = "sop_temp"

# Ensure the directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class Input(BaseModel):
    input: str
    file_name: Optional[str] = None
    file_content: Optional[bytes] = None


class Output(BaseModel):
    output: dict
    
@app.post("/speckle_chat/")
async def handle_speckle_chat(
    input: str = Form(...),
    file_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    # Handle file processing if a file is uploaded
    file_content = await file.read() if file else None
    
    if file_content and file_name:
        # Save the file to the specified directory
        file_location = os.path.join(UPLOAD_DIRECTORY, file_name)
        with open(file_location, "wb") as f:
            f.write(file_content)

    # Construct the input data with file details
    input_data = Input(
        input=input,
        file_name=file_name,
        file_content=file_content
    )
    
    # Execute the chain with input data, automatically generating the response
    typed_chain = chain.with_types(input_type=Input, output_type=Output)
    response = typed_chain(input_data)

    # Return the output directly (handled by the chain)
    return response

# Add routes for the chain
add_routes(app, chain.with_types(input_type=Input, output_type=Output), path="/speckle_chat")


if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable if available, otherwise default to 8000
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
