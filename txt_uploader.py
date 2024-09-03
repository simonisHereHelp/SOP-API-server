import os
import sys
import tempfile
import streamlit as st
from utils.vector_store import create_vector_store
from utils.grader import GraderUtils
from utils.graph import GraphState
from utils.generate_chain import create_generate_chain
from utils.nodes import GraphNodes
from utils.edges import EdgeGraph
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# Set up environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Streamlit interface
st.title("US Personal Income Tax Code Query App")
st.write("Upload a text file and check if it's related to US Personal Income Tax codes.")

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose a file", type="txt")

if uploaded_file is not None:
        # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    loader = TextLoader(temp_file_path)

    document=loader.load()
    # Display the uploaded file content
    st.write("File content:")
    st.text(document[0].page_content if document else "No content found")

    # Initialize the LLM and other components
    embedding_model = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    

    # Create vector store and retriever
    store = create_vector_store(document)
    retriever = store.as_retriever()

    # Create the generate chain
    generate_chain = create_generate_chain(llm)

    # Set up the graders
    grader = GraderUtils(llm)
    retrieval_grader = grader.create_retrieval_grader()
    hallucination_grader = grader.create_hallucination_grader()
    code_evaluator = grader.create_code_evaluator()
    question_rewriter = grader.create_question_rewriter()

    # Initiate the Graph
    workflow = StateGraph(GraphState)
    graph_nodes = GraphNodes(llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter)
    edge_graph = EdgeGraph(hallucination_grader, code_evaluator)

    # Define the nodes
    workflow.add_node("retrieve", graph_nodes.retrieve)
    workflow.add_node("grade_documents", graph_nodes.grade_documents)
    workflow.add_node("generate", graph_nodes.generate)
    workflow.add_node("transform_query", graph_nodes.transform_query)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        edge_graph.decide_to_generate,
        {
            "transform_query": "transform_query",
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
            "not useful": "transform_query",
        },
    )

    # Compile the chain
    chain = workflow.compile()

    # Query input from the user
    query = "Is the uploaded file related to US Personal Income Tax codes?"

    if st.button("Submit Query"):
        # Run the workflow with the input query
        input_data = {"input": query}
        response = chain(input_data)
        
        # Display the response
        st.write("Query:", query)
        st.write("Response:", response["output"])

